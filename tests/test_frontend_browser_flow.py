import threading
import unittest
from pathlib import Path

from flask import Flask, jsonify, send_file
from werkzeug.serving import make_server


try:
    from playwright.sync_api import sync_playwright
except ModuleNotFoundError:  # pragma: no cover - depends on optional browser test dependency
    sync_playwright = None


class FrontendBrowserFlowTests(unittest.TestCase):
    @unittest.skipIf(sync_playwright is None, "playwright is not installed")
    def test_summary_shortcut_reports_success_only_after_backend_success(self):
        app = Flask(__name__)
        requests = []
        allow_summary_response = threading.Event()

        feed_item = {
            "id": 123,
            "title": "Browser flow test article",
            "source_label": "example.com",
            "source_url": "https://example.com",
            "source_feed": "test-feed",
            "published_at": "2026-05-22T10:00:00+00:00",
            "link_to_article": "https://example.com/article",
            "feed_decision": "pending",
            "similar_count": 0,
            "prediction": {
                "available": True,
                "recommended": False,
                "maybe": True,
                "tier": "maybe",
                "probability": 0.7,
                "run_id": 1,
            },
        }

        @app.get("/")
        def index():
            return send_file(Path("local-news-app.html").resolve())

        @app.get("/api/status")
        def status():
            return jsonify(
                {
                    "feed": {"pending": 1},
                    "summaries": {"queued": 0, "processing": 0, "ready": 0, "failed": 0, "review_total": 0},
                    "models": {},
                    "llm_compare": {"enabled": False},
                    "ollama": {"reachable": True},
                    "last_feed_refresh": {},
                    "config": {},
                }
            )

        @app.get("/api/feed")
        def feed():
            return jsonify(
                {
                    "mode": "all",
                    "items": [feed_item],
                    "counts": {
                        "total_pending": 1,
                        "recommended_pending": 0,
                        "maybe_pending": 1,
                        "similar_group_count": 0,
                        "similar_hidden_count": 0,
                    },
                }
            )

        @app.post("/api/articles/<int:article_id>/summarize")
        def summarize(article_id):
            requests.append(article_id)
            allow_summary_response.wait(timeout=5)
            return jsonify({"status": "ok", "article_id": article_id, "decision": "summarize", "deduplicated_count": 0})

        server = make_server("127.0.0.1", 0, app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"

        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                page = browser.new_page()
                try:
                    page.goto(base_url)
                    page.get_by_text("Browser flow test article").wait_for(timeout=5000)
                    page.keyboard.press("s")

                    page.get_by_text("Summary wird vorgemerkt", exact=False).wait_for(timeout=5000)
                    self.assertEqual(requests, [123])
                    self.assertEqual(page.get_by_text("Summary vorgemerkt. Läuft im Hintergrund", exact=False).count(), 0)

                    allow_summary_response.set()
                    page.get_by_text("Summary vorgemerkt. Läuft im Hintergrund", exact=False).wait_for(timeout=5000)
                finally:
                    browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
