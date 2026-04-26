import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def event_payloads(self, article_id):
        return [json.loads(event["event_payload"]) for event in self.event_rows(article_id)]

    def store_embedding(self, article_id, input_hash=None, model=None):
        now = backend.utc_now_iso()
        article = self.article_row(article_id)
        input_hash = input_hash if input_hash is not None else backend.build_embedding_input_hash(article)
        model = model if model is not None else backend.get_embedding_model()
        with backend.db_connection() as conn:
            conn.execute(
                """
                INSERT INTO article_embeddings(
                    article_id, embedding_model, embedding_input_hash, embedding_json, generated_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id) DO UPDATE SET
                    embedding_model = excluded.embedding_model,
                    embedding_input_hash = excluded.embedding_input_hash,
                    embedding_json = excluded.embedding_json,
                    generated_at = excluded.generated_at,
                    updated_at = excluded.updated_at
                """,
                (article_id, model, input_hash, json.dumps([0.1, 0.2]), now, now),
            )

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
        payload = json.loads(events[0]["event_payload"])
        self.assertEqual(payload["previous_feed_decision"], "pending")
        self.assertEqual(payload["new_feed_decision"], "summarize")
        self.assertIn("prediction_snapshot", payload)

    def test_summarize_endpoint_requeues_failed_article_and_clears_error(self):
        article_id = self.insert_article(summary_status="failed")
        with backend.db_connection() as conn:
            conn.execute("UPDATE articles SET last_error = ? WHERE id = ?", ("previous failure", article_id))

        response = self.client.post(f"/api/articles/{article_id}/summarize")

        self.assertEqual(response.status_code, 200)
        article = self.article_row(article_id)
        self.assertEqual(article["feed_decision"], "summarize")
        self.assertEqual(article["summary_status"], "queued")
        self.assertEqual(article["last_error"], "")
        self.assertIsNotNone(article["summary_requested_at"])

    def test_skip_endpoint_rejects_article_already_in_summary_flow(self):
        article_id = self.insert_article(summary_status="queued")

        response = self.client.post(f"/api/articles/{article_id}/skip")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["message"], "Article is already in summary flow")
        self.assertEqual(self.article_row(article_id)["feed_decision"], "pending")

    def test_summary_job_transitions_from_queued_to_processing_to_ready(self):
        article_id = self.insert_article(feed_decision="summarize", summary_status="queued")
        with backend.db_connection() as conn:
            conn.execute(
                "UPDATE articles SET summary_requested_at = ? WHERE id = ?",
                (backend.utc_now_iso(), article_id),
            )

        job = backend.claim_next_summary_job()

        self.assertIsNotNone(job)
        self.assertEqual(job["id"], article_id)
        self.assertEqual(self.article_row(article_id)["summary_status"], "processing")

        with mock.patch(
            "local_news_backend.fetch_best_article_text",
            return_value=("Article body " * 40, "https://example.com/final"),
        ), mock.patch(
            "local_news_backend.ollama_generate_summary",
            return_value=("Generated title", "Generated summary"),
        ), mock.patch("local_news_backend.get_compare_enabled", return_value=False):
            backend.process_summary_job(job)

        article = self.article_row(article_id)
        self.assertEqual(article["summary_status"], "ready")
        self.assertEqual(article["summary_title"], "Generated title")
        self.assertEqual(article["summary_text"], "Generated summary")
        self.assertEqual(article["link_to_article"], "https://example.com/final")
        self.assertEqual(article["last_error"], "")
        self.assertIsNotNone(article["summarized_at"])
        self.assertEqual(
            [event["event_type"] for event in self.event_rows(article_id)],
            ["summary_processing_started", "summary_generated"],
        )
        payloads = self.event_payloads(article_id)
        self.assertEqual(payloads[0], {})
        self.assertEqual(payloads[1]["model"], backend.OLLAMA_MODEL)
        self.assertEqual(payloads[1]["final_url"], "https://example.com/final")
        self.assertFalse(payloads[1]["extraction_fallback"])

    def test_summary_job_uses_metadata_fallback_when_extraction_is_too_short(self):
        article_id = self.insert_article(
            feed_decision="summarize",
            summary_status="processing",
            title="Important AI headline from feed",
            source_label="example.com",
            source_url="https://example.com",
        )
        job = self.article_row(article_id)

        def summarize_stub(title, text):
            self.assertEqual(title, "Important AI headline from feed")
            self.assertIn("Volltext konnte nicht zuverlässig extrahiert werden", text)
            self.assertIn("Titel: Important AI headline from feed", text)
            self.assertIn("Vorhandener Textauszug: Too short", text)
            return "Fallback title", "Fallback summary"

        with mock.patch(
            "local_news_backend.fetch_best_article_text",
            return_value=("Too short", "https://example.com/final"),
        ), mock.patch("local_news_backend.ollama_generate_summary", side_effect=summarize_stub), mock.patch(
            "local_news_backend.get_compare_enabled", return_value=False
        ):
            backend.process_summary_job(job)

        article = self.article_row(article_id)
        self.assertEqual(article["summary_status"], "ready")
        self.assertEqual(article["summary_title"], "Fallback title")
        self.assertEqual(article["summary_text"], "Fallback summary")
        self.assertIn("Volltext konnte nicht zuverlässig extrahiert werden", article["article_text"])
        self.assertEqual([event["event_type"] for event in self.event_rows(article_id)], ["summary_generated"])
        self.assertTrue(self.event_payloads(article_id)[0]["extraction_fallback"])

    def test_article_fetch_url_candidates_decode_and_skip_source_homepage(self):
        article = {
            "link_to_article": "https://news.google.com/rss/articles/token",
            "rss_source_url": "https://news.google.com/rss/articles/token",
            "source_url": "https://example.com",
        }

        with mock.patch("local_news_backend.decode_google_news_url", return_value="https://example.com/story"):
            candidates = backend.article_fetch_url_candidates(article)

        self.assertEqual(
            candidates,
            ["https://example.com/story", "https://news.google.com/rss/articles/token"],
        )

    def test_fetch_best_article_text_tries_candidates_until_enough_text(self):
        article = {
            "link_to_article": "https://news.google.com/rss/articles/token",
            "rss_source_url": "https://news.google.com/rss/articles/token",
            "source_url": "https://publisher.example/story",
        }

        def fetch_stub(url):
            if url == "https://news.google.com/rss/articles/token":
                return "short", url
            if url == "https://publisher.example/story":
                return "Long article text " * 30, url
            raise AssertionError(f"Unexpected URL: {url}")

        with mock.patch("local_news_backend.decode_google_news_url", side_effect=lambda url: url), mock.patch(
            "local_news_backend.fetch_and_extract_article_text", side_effect=fetch_stub
        ) as fetch:
            text, final_url = backend.fetch_best_article_text(article)

        self.assertEqual(final_url, "https://publisher.example/story")
        self.assertGreaterEqual(len(text), backend.MIN_EXTRACTED_ARTICLE_CHARS)
        self.assertEqual(
            [call.args[0] for call in fetch.call_args_list],
            ["https://news.google.com/rss/articles/token", "https://publisher.example/story"],
        )

    def test_fetch_best_article_text_returns_empty_text_for_metadata_fallback(self):
        article = {
            "link_to_article": "https://news.google.com/rss/articles/token",
            "rss_source_url": "",
            "source_url": "https://example.com",
        }

        with mock.patch("local_news_backend.decode_google_news_url", side_effect=lambda url: url), mock.patch(
            "local_news_backend.fetch_and_extract_article_text",
            return_value=("", "https://news.google.com/rss/articles/token"),
        ):
            text, final_url = backend.fetch_best_article_text(article)

        self.assertEqual(text, "")
        self.assertEqual(final_url, "https://news.google.com/rss/articles/token")

    def test_mark_summary_processing_updates_status_and_logs_event(self):
        article_id = self.insert_article(feed_decision="summarize", summary_status="queued")

        with backend.db_connection() as conn:
            backend.mark_summary_processing(conn, article_id)

        self.assertEqual(self.article_row(article_id)["summary_status"], "processing")
        self.assertEqual([event["event_type"] for event in self.event_rows(article_id)], ["summary_processing_started"])
        self.assertEqual(self.event_payloads(article_id), [{}])

    def test_mark_summary_ready_updates_article_and_logs_event(self):
        article_id = self.insert_article(feed_decision="summarize", summary_status="processing")

        with backend.db_connection() as conn:
            backend.mark_summary_ready(
                conn,
                article_id,
                "Article body",
                "https://example.com/final",
                "Ready title",
                "Ready summary",
            )

        article = self.article_row(article_id)
        self.assertEqual(article["summary_status"], "ready")
        self.assertEqual(article["article_text"], "Article body")
        self.assertEqual(article["link_to_article"], "https://example.com/final")
        self.assertEqual(article["summary_title"], "Ready title")
        self.assertEqual(article["summary_text"], "Ready summary")
        self.assertEqual(article["summary_model"], backend.OLLAMA_MODEL)
        self.assertEqual(article["last_error"], "")
        self.assertIsNotNone(article["summarized_at"])
        self.assertEqual([event["event_type"] for event in self.event_rows(article_id)], ["summary_generated"])
        self.assertEqual(
            self.event_payloads(article_id),
            [{"model": backend.OLLAMA_MODEL, "final_url": "https://example.com/final", "extraction_fallback": False}],
        )

    def test_legacy_summary_job_getter_delegates_to_claim_function(self):
        with mock.patch("local_news_backend.claim_next_summary_job", return_value={"id": 123}) as claim:
            self.assertEqual(backend.get_next_summary_job(), {"id": 123})
        claim.assert_called_once_with()

    def test_summary_job_failure_marks_failed_and_logs_error(self):
        article_id = self.insert_article(feed_decision="summarize", summary_status="processing")
        job = self.article_row(article_id)

        with mock.patch(
            "local_news_backend.fetch_best_article_text",
            side_effect=RuntimeError("extraction failed"),
        ), mock.patch("local_news_backend.LOGGER.warning"):
            backend.process_summary_job(job)

        article = self.article_row(article_id)
        self.assertEqual(article["summary_status"], "failed")
        self.assertEqual(article["last_error"], "extraction failed")
        self.assertEqual([event["event_type"] for event in self.event_rows(article_id)], ["summary_failed"])
        self.assertEqual(self.event_payloads(article_id), [{"error": "extraction failed"}])

    def test_stale_processing_summary_job_is_recovered_to_failed(self):
        old_timestamp = (
            backend.utc_now() - backend.timedelta(minutes=backend.SUMMARY_PROCESSING_STALE_MINUTES + 1)
        ).isoformat()
        article_id = self.insert_article(summary_status="processing", updated_at=old_timestamp)

        recovered = backend.recover_stale_processing_jobs()

        self.assertEqual(recovered, 1)
        article = self.article_row(article_id)
        self.assertEqual(article["summary_status"], "failed")
        self.assertIn("recovery timeout", article["last_error"])
        self.assertEqual(
            [event["event_type"] for event in self.event_rows(article_id)],
            ["summary_processing_recovered"],
        )
        self.assertEqual(
            self.event_payloads(article_id),
            [{"recovery": "stale_to_failed", "timeout_minutes": backend.SUMMARY_PROCESSING_STALE_MINUTES}],
        )

    def test_skip_endpoint_marks_article_and_logs_event(self):
        article_id = self.insert_article()

        response = self.client.post(f"/api/articles/{article_id}/skip")

        self.assertEqual(response.status_code, 200)
        article = self.article_row(article_id)
        self.assertEqual(article["feed_decision"], "skip")
        self.assertEqual(article["summary_status"], "not_requested")
        events = self.event_rows(article_id)
        self.assertEqual([event["event_type"] for event in events], ["article_skipped"])
        payload = json.loads(events[0]["event_payload"])
        self.assertEqual(payload["previous_feed_decision"], "pending")
        self.assertEqual(payload["new_feed_decision"], "skip")
        self.assertIn("prediction_snapshot", payload)

    def test_summary_feedback_endpoint_updates_article_and_logs_payload(self):
        article_id = self.insert_article(summary_status="ready")

        response = self.client.post(
            f"/api/summaries/{article_id}/feedback",
            json={"feedback": "interesting"},
        )

        self.assertEqual(response.status_code, 200)
        article = self.article_row(article_id)
        self.assertEqual(article["summary_feedback"], "interesting")
        self.assertIsNotNone(article["summary_feedback_at"])
        self.assertEqual([event["event_type"] for event in self.event_rows(article_id)], ["summary_feedback"])
        self.assertEqual(self.event_payloads(article_id), [{"feedback": "interesting"}])

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

    def test_embedding_selection_skips_empty_input_titles(self):
        empty_id = self.insert_article(guid="empty-input", title="\u200b")
        valid_id = self.insert_article(guid="valid-input", title="Valid article title for embedding")

        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], valid_id)
        self.assertNotEqual(selected["id"], empty_id)

    def test_embedding_selection_skips_current_matching_embedding(self):
        article_id = self.insert_article(guid="already-embedded", title="Already embedded article title")
        self.store_embedding(article_id)

        selected = backend.select_article_for_embedding()

        self.assertIsNone(selected)

    def test_embedding_selection_reselects_article_when_hash_changed(self):
        article_id = self.insert_article(guid="stale-hash", title="Article with changed embedding input")
        self.store_embedding(article_id, input_hash="old-hash")

        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], article_id)
        self.assertEqual(selected["expected_embedding_hash"], backend.build_embedding_input_hash(selected))

    def test_embedding_selection_reselects_article_when_model_changed(self):
        article_id = self.insert_article(guid="stale-model", title="Article embedded with old model")
        self.store_embedding(article_id, model="previous-embedding-model")

        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], article_id)
        self.assertEqual(selected["embedding_model"], None)

    def test_embedding_selection_prioritizes_pending_then_summarize_then_skip(self):
        now = backend.utc_now()
        skip_id = self.insert_article(
            guid="skip-priority",
            title="Newest skipped article",
            feed_decision="skip",
            published_at=now.isoformat(),
        )
        summarize_id = self.insert_article(
            guid="summarize-priority",
            title="Middle summarized article",
            feed_decision="summarize",
            published_at=(now - backend.timedelta(minutes=1)).isoformat(),
        )
        pending_id = self.insert_article(
            guid="pending-priority",
            title="Oldest pending article",
            feed_decision="pending",
            published_at=(now - backend.timedelta(minutes=2)).isoformat(),
        )

        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], pending_id)

        self.store_embedding(pending_id)
        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], summarize_id)

        self.store_embedding(summarize_id)
        selected = backend.select_article_for_embedding()

        self.assertIsNotNone(selected)
        self.assertEqual(selected["id"], skip_id)

    def test_embedding_selection_ignores_articles_outside_recent_window(self):
        old_timestamp = (backend.utc_now() - backend.timedelta(hours=73)).isoformat()
        self.insert_article(guid="old-article", title="Old article outside embedding window", published_at=old_timestamp)

        selected = backend.select_article_for_embedding()

        self.assertIsNone(selected)

    def test_summary_work_pending_tracks_queued_and_processing_jobs(self):
        self.assertFalse(backend.summary_work_pending())

        processing_id = self.insert_article(guid="processing", summary_status="processing")
        self.assertTrue(backend.summary_work_pending())

        with backend.db_connection() as conn:
            conn.execute("DELETE FROM articles WHERE id = ?", (processing_id,))
        self.assertFalse(backend.summary_work_pending())

        self.insert_article(guid="queued", summary_status="queued")
        self.assertTrue(backend.summary_work_pending())

    def test_embedding_worker_step_pauses_when_summary_work_is_pending(self):
        with mock.patch("local_news_backend.summary_work_pending", return_value=True), mock.patch(
            "local_news_backend.select_article_for_embedding"
        ) as select_article, mock.patch.object(backend.STATE.stop_event, "wait", return_value=False) as wait:
            result = backend.run_embedding_worker_once()

        self.assertEqual(result, "summary_pending")
        select_article.assert_not_called()
        wait.assert_called_once()

    def test_frontend_s_shortcut_is_bound_in_capture_phase(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("document.addEventListener('keydown', handleKeyboard, true)", html)
        self.assertIn("isPlainShortcut(event, 's', 'KeyS')", html)
        self.assertIn("showToast('Summary vorgemerkt.", html)

    def test_frontend_empty_ranked_feed_switches_to_all_without_reload(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("function switchEmptyFeedModeToAll()", html)
        self.assertIn("function syncFeedCountsFromLocalItems()", html)
        self.assertIn("Keine Treffer mehr in diesem Modus. Zeige alle offenen Feed-Einträge.", html)
        self.assertIn("switchEmptyFeedModeToAll();", html)
        self.assertNotIn("else if (!state.feedItems.length)", html)


if __name__ == "__main__":
    unittest.main()
