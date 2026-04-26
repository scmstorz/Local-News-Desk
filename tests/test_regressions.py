import json
import tempfile
import unittest
from pathlib import Path

import local_news_backend as backend


class LocalNewsRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.tmpdir.name)
        self.original_db_path = backend.DB_PATH
        self.original_model_dir = backend.MODEL_DIR
        self.original_compare_export_dir = backend.COMPARE_EXPORT_DIR

        backend.DB_PATH = self.base_path / "test-news.db"
        backend.MODEL_DIR = self.base_path / "models"
        backend.COMPARE_EXPORT_DIR = self.base_path / "compare_exports"
        backend.STATE.stop_event.clear()
        backend.STATE.summary_event.clear()
        backend.STATE.feed_similarity_snapshot = {}
        backend.STATE.feed_similarity_dirty = True
        backend.STATE.feed_similarity_building = False
        with backend.STATE.model_lock:
            backend.STATE.models.clear()

        backend.init_db()
        self.client = backend.APP.test_client()

    def tearDown(self) -> None:
        with backend.STATE.model_lock:
            backend.STATE.models.clear()
        backend.DB_PATH = self.original_db_path
        backend.MODEL_DIR = self.original_model_dir
        backend.COMPARE_EXPORT_DIR = self.original_compare_export_dir
        backend.STATE.summary_event.clear()
        self.tmpdir.cleanup()

    def insert_article(self, **overrides):
        now = backend.utc_now_iso()
        values = {
            "guid": overrides.pop("guid", f"guid-{now}-{len(overrides)}"),
            "title": overrides.pop("title", "OpenAI launches a useful local test article"),
            "link_to_article": overrides.pop("link_to_article", "https://example.com/article"),
            "rss_source_url": overrides.pop("rss_source_url", "https://example.com/rss-item"),
            "source_url": overrides.pop("source_url", "https://example.com"),
            "source_label": overrides.pop("source_label", "example.com"),
            "source_feed": overrides.pop("source_feed", "test-feed"),
            "published_at": overrides.pop("published_at", now),
            "created_at": overrides.pop("created_at", now),
            "updated_at": overrides.pop("updated_at", now),
            "feed_decision": overrides.pop("feed_decision", "pending"),
            "summary_status": overrides.pop("summary_status", "not_requested"),
            "predicted_recommendation": overrides.pop("predicted_recommendation", None),
            "predicted_probability": overrides.pop("predicted_probability", None),
            "prediction_model_run_id": overrides.pop("prediction_model_run_id", None),
            "prediction_generated_at": overrides.pop("prediction_generated_at", None),
        }
        if overrides:
            raise AssertionError(f"Unused article overrides: {sorted(overrides)}")

        with backend.db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO articles(
                    guid, title, link_to_article, rss_source_url, source_url, source_label,
                    source_feed, published_at, created_at, updated_at, feed_decision,
                    summary_status, predicted_recommendation, predicted_probability,
                    prediction_model_run_id, prediction_generated_at
                )
                VALUES (
                    :guid, :title, :link_to_article, :rss_source_url, :source_url, :source_label,
                    :source_feed, :published_at, :created_at, :updated_at, :feed_decision,
                    :summary_status, :predicted_recommendation, :predicted_probability,
                    :prediction_model_run_id, :prediction_generated_at
                )
                """,
                values,
            )
            return int(cursor.lastrowid)

    def article_row(self, article_id):
        with backend.db_connection() as conn:
            return backend.row_to_dict(conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone())

    def event_rows(self, article_id):
        with backend.db_connection() as conn:
            return [
                backend.row_to_dict(row)
                for row in conn.execute(
                    """
                    SELECT *
                    FROM article_events
                    WHERE article_id = ?
                    ORDER BY id
                    """,
                    (article_id,),
                ).fetchall()
            ]

    def test_summarize_endpoint_queues_article_and_wakes_worker(self):
        article_id = self.insert_article()

        response = self.client.post(f"/api/articles/{article_id}/summarize")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["decision"], "summarize")
        self.assertTrue(backend.STATE.summary_event.is_set())

        article = self.article_row(article_id)
        self.assertEqual(article["feed_decision"], "summarize")
        self.assertEqual(article["summary_status"], "queued")
        self.assertIsNotNone(article["summary_requested_at"])

        events = self.event_rows(article_id)
        self.assertEqual([event["event_type"] for event in events], ["summary_requested"])

    def test_skip_endpoint_marks_article_and_logs_event(self):
        article_id = self.insert_article()

        response = self.client.post(f"/api/articles/{article_id}/skip")

        self.assertEqual(response.status_code, 200)
        article = self.article_row(article_id)
        self.assertEqual(article["feed_decision"], "skip")
        self.assertEqual(article["summary_status"], "not_requested")
        events = self.event_rows(article_id)
        self.assertEqual([event["event_type"] for event in events], ["article_skipped"])

    def test_feed_recommended_and_maybe_modes_are_disjoint(self):
        run_id = 42
        with backend.STATE.model_lock:
            backend.STATE.models["feed_recommendation"] = {
                "run_id": run_id,
                "threshold": 0.7,
                "maybe_threshold": 0.4,
                "pipeline": None,
            }

        recommended_id = self.insert_article(
            guid="recommended",
            title="Strong recommended article",
            predicted_recommendation=1,
            predicted_probability=0.9,
            prediction_model_run_id=run_id,
        )
        maybe_id = self.insert_article(
            guid="maybe",
            title="Borderline maybe article",
            predicted_recommendation=0,
            predicted_probability=0.5,
            prediction_model_run_id=run_id,
        )
        no_id = self.insert_article(
            guid="no",
            title="Low score article",
            predicted_recommendation=0,
            predicted_probability=0.2,
            prediction_model_run_id=run_id,
        )

        recommended_response = self.client.get("/api/feed?mode=recommended")
        maybe_response = self.client.get("/api/feed?mode=maybe")
        all_response = self.client.get("/api/feed?mode=all")

        self.assertEqual(recommended_response.status_code, 200)
        self.assertEqual(maybe_response.status_code, 200)
        self.assertEqual(all_response.status_code, 200)

        recommended_ids = {item["id"] for item in recommended_response.get_json()["items"]}
        maybe_ids = {item["id"] for item in maybe_response.get_json()["items"]}
        all_ids = {item["id"] for item in all_response.get_json()["items"]}

        self.assertEqual(recommended_ids, {recommended_id})
        self.assertEqual(maybe_ids, {maybe_id})
        self.assertEqual(all_ids, {recommended_id, maybe_id, no_id})
        self.assertFalse(recommended_ids & maybe_ids)

    def test_feed_mode_filtering_is_explicit_and_keeps_legacy_maybe_plus(self):
        rows = [
            {"id": 1, "prediction": {"recommended": True, "tier": "recommended"}},
            {"id": 2, "prediction": {"recommended": False, "tier": "maybe"}},
            {"id": 3, "prediction": {"recommended": False, "tier": "no"}},
            {"id": 4, "prediction": {"recommended": None, "tier": None}},
        ]

        self.assertEqual(
            [row["id"] for row in backend.filter_predicted_feed_rows(rows, "recommended")],
            [1],
        )
        self.assertEqual(
            [row["id"] for row in backend.filter_predicted_feed_rows(rows, "maybe")],
            [2],
        )
        self.assertEqual(
            [row["id"] for row in backend.filter_predicted_feed_rows(rows, "maybe_plus")],
            [1, 2],
        )
        self.assertEqual(
            [row["id"] for row in backend.filter_predicted_feed_rows(rows, "all")],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            [row["id"] for row in backend.filter_predicted_feed_rows(rows, "unknown")],
            [1, 2, 3, 4],
        )

    def test_feed_prediction_helpers_build_consistent_tiers(self):
        self.assertEqual(
            backend.build_unavailable_feed_prediction(),
            {
                "available": False,
                "recommended": None,
                "maybe": None,
                "tier": None,
                "probability": None,
                "run_id": None,
            },
        )
        self.assertEqual(
            backend.build_threshold_feed_prediction(0.9, 0.7, 0.4, 42),
            {
                "available": True,
                "recommended": True,
                "maybe": True,
                "tier": "recommended",
                "probability": 0.9,
                "run_id": 42,
            },
        )
        self.assertEqual(
            backend.build_threshold_feed_prediction(0.5, 0.7, 0.4, 42)["tier"],
            "maybe",
        )
        self.assertEqual(
            backend.build_threshold_feed_prediction(0.2, 0.7, 0.4, 42)["tier"],
            "no",
        )
        self.assertEqual(
            backend.build_feed_prediction(False, True, 0.45678, 42),
            {
                "available": True,
                "recommended": False,
                "maybe": True,
                "tier": "maybe",
                "probability": 0.4568,
                "run_id": 42,
            },
        )

    def test_embedding_input_preserves_unicode_title_fallback(self):
        article = {
            "title": "Сэм Альтман: «Хатоларимиз учун аҳолидан чуқур узр сўрайман» - Zamin.uz"
        }

        input_text = backend.build_embedding_input_text(article)

        self.assertEqual(input_text, "Сэм Альтман: «Хатоларимиз учун аҳолидан чуқур узр сўрайман»")
        self.assertNotEqual(input_text, "")

    def test_embedding_response_parser_accepts_common_shapes(self):
        self.assertEqual(backend.extract_ollama_embedding_vector({"embedding": [0.1, 0.2]}), [0.1, 0.2])
        self.assertEqual(backend.extract_ollama_embedding_vector({"embeddings": [[0.1, 0.2]]}), [0.1, 0.2])
        self.assertEqual(backend.extract_ollama_embedding_vector({"embeddings": [0.1, 0.2]}), [0.1, 0.2])
        self.assertEqual(backend.extract_ollama_embedding_vector({"data": [{"embedding": [0.1, 0.2]}]}), [0.1, 0.2])
        self.assertIsNone(backend.extract_ollama_embedding_vector({"embeddings": []}))

    def test_summary_work_pending_tracks_queued_and_processing_jobs(self):
        self.assertFalse(backend.summary_work_pending())

        processing_id = self.insert_article(guid="processing", summary_status="processing")
        self.assertTrue(backend.summary_work_pending())

        with backend.db_connection() as conn:
            conn.execute("DELETE FROM articles WHERE id = ?", (processing_id,))
        self.assertFalse(backend.summary_work_pending())

        self.insert_article(guid="queued", summary_status="queued")
        self.assertTrue(backend.summary_work_pending())

    def test_frontend_s_shortcut_is_bound_in_capture_phase(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("document.addEventListener('keydown', handleKeyboard, true)", html)
        self.assertIn("isPlainShortcut(event, 's', 'KeyS')", html)
        self.assertIn("showToast('Summary vorgemerkt.", html)


if __name__ == "__main__":
    unittest.main()
