import re
import unittest
from pathlib import Path

import local_news_backend as backend


class UiHttpContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = backend.APP.test_client()

    def test_backend_serves_frontend_from_root_and_app_routes(self):
        for path in ("/", "/app"):
            with self.subTest(path=path):
                response = self.client.get(path)
                data = response.get_data()
                response.close()

                self.assertEqual(response.status_code, 200)
                self.assertIn("text/html", response.content_type)
                self.assertIn(b"<title>Local News Desk</title>", data)

    def test_frontend_uses_same_origin_api_when_served_over_http(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("function resolveApiBase()", html)
        self.assertIn("window.location.origin}/api", html)
        self.assertRegex(html, re.compile(r"const\s+API_BASE\s*=\s*resolveApiBase\(\);"))

    def test_frontend_keeps_file_protocol_fallback_for_direct_opening(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("window.location.protocol === 'http:'", html)
        self.assertIn("http://127.0.0.1:8765/api", html)

    def test_frontend_api_base_can_be_overridden_before_script_loads(self):
        html = Path("local-news-app.html").read_text(encoding="utf-8")

        self.assertIn("window.LOCAL_NEWS_API_BASE", html)
        self.assertIn("replace(/\\/$/, '')", html)


if __name__ == "__main__":
    unittest.main()
