#!/usr/bin/env python3
"""Local backend for the News App.

Replaces the old Firebase + Cloud Function stack with:
- SQLite for storage
- RSS polling
- article extraction
- Ollama-based summarization
- simple local classifiers with model metrics

The frontend can be opened directly as a local HTML file and talks to this
backend via http://localhost:8765.
"""

from __future__ import annotations

import html as html_lib
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import feedparser
import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from googlenewsdecoder import gnewsdecoder
except ImportError:  # pragma: no cover - optional dependency at runtime
    gnewsdecoder = None

try:
    import trafilatura
except ImportError:  # pragma: no cover - optional dependency at runtime
    trafilatura = None


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("LOCAL_NEWS_CONFIG_PATH", BASE_DIR / "local_config.json"))
DEFAULT_SETTINGS = {
    "server": {
        "host": "127.0.0.1",
        "port": 8765,
    },
    "storage": {
        "db_path": "local-news.db",
        "model_dir": "models",
    },
    "ollama": {
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen3.5:35b",
    },
    "timing": {
        "feed_refresh_seconds": 300,
        "summary_poll_seconds": 3,
        "request_timeout_seconds": 20,
    },
    "feeds": [
        "https://news.google.com/rss/search?q=%22generative%20ai%22%20OR%20llm%20when%3A1h&hl=en-US&gl=US&ceid=US%3Aen",
        "https://news.google.com/rss/search?q=artificial%20intelligence%20when%3A1h&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=anthropic%20OR%20openai%20OR%20%22google%20gemini%22%20OR%20%22open%20source%20llm%22%20OR%20nvidia%20when%3A1h&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=machine+learning+when:2d&hl=en-US&gl=US&ceid=US:en",
    ],
}


def merge_settings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict[str, Any]:
    settings = dict(DEFAULT_SETTINGS)
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            file_settings = json.load(handle)
        settings = merge_settings(settings, file_settings)
    return settings


def resolve_local_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


SETTINGS = load_settings()
DB_PATH = resolve_local_path(os.environ.get("LOCAL_NEWS_DB_PATH", SETTINGS["storage"]["db_path"]))
MODEL_DIR = resolve_local_path(os.environ.get("LOCAL_NEWS_MODEL_DIR", SETTINGS["storage"]["model_dir"]))
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", SETTINGS["ollama"]["base_url"]).rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", SETTINGS["ollama"]["model"])
HOST = os.environ.get("LOCAL_NEWS_HOST", SETTINGS["server"]["host"])
PORT = int(os.environ.get("LOCAL_NEWS_PORT", str(SETTINGS["server"]["port"])))
FEED_REFRESH_SECONDS = int(
    os.environ.get("LOCAL_NEWS_REFRESH_SECONDS", str(SETTINGS["timing"]["feed_refresh_seconds"]))
)
SUMMARY_POLL_SECONDS = float(
    os.environ.get("LOCAL_NEWS_SUMMARY_POLL_SECONDS", str(SETTINGS["timing"]["summary_poll_seconds"]))
)
REQUEST_TIMEOUT_SECONDS = int(
    os.environ.get("LOCAL_NEWS_REQUEST_TIMEOUT_SECONDS", str(SETTINGS["timing"]["request_timeout_seconds"]))
)
RSS_FEED_URLS = list(SETTINGS["feeds"])

LOGGER = logging.getLogger("local_news_backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    guid TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    link_to_article TEXT NOT NULL DEFAULT '',
    rss_source_url TEXT NOT NULL DEFAULT '',
    source_url TEXT NOT NULL DEFAULT '',
    source_label TEXT NOT NULL DEFAULT '',
    source_feed TEXT NOT NULL DEFAULT '',
    published_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    article_text TEXT NOT NULL DEFAULT '',
    article_text_extracted_at TEXT,
    feed_decision TEXT NOT NULL DEFAULT 'pending',
    feed_decision_at TEXT,
    summary_status TEXT NOT NULL DEFAULT 'not_requested',
    summary_title TEXT NOT NULL DEFAULT '',
    summary_text TEXT NOT NULL DEFAULT '',
    summary_model TEXT NOT NULL DEFAULT '',
    summary_requested_at TEXT,
    summarized_at TEXT,
    summary_feedback TEXT NOT NULL DEFAULT 'unreviewed',
    summary_feedback_at TEXT,
    predicted_recommendation INTEGER,
    predicted_probability REAL,
    prediction_model_run_id INTEGER,
    prediction_generated_at TEXT,
    last_error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS article_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    event_payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target TEXT NOT NULL,
    model_path TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    labels_used INTEGER NOT NULL,
    train_size INTEGER NOT NULL,
    test_size INTEGER NOT NULL,
    positive_labels INTEGER NOT NULL,
    negative_labels INTEGER NOT NULL,
    accuracy REAL NOT NULL,
    precision REAL NOT NULL,
    recall REAL NOT NULL,
    f1 REAL NOT NULL,
    threshold_value REAL NOT NULL,
    confusion_matrix_json TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'trained'
);

CREATE TABLE IF NOT EXISTS app_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_articles_feed_queue
    ON articles(feed_decision, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_summary_queue
    ON articles(summary_status, summary_feedback, summary_requested_at);
CREATE INDEX IF NOT EXISTS idx_events_article_time
    ON article_events(article_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_model_runs_target_time
    ON model_runs(target, trained_at DESC);
"""

TARGET_CONFIG = {
    "feed_recommendation": {
        "artifact_name": "feed_recommendation.joblib",
        "positive_label": "summarize",
        "query": """
            SELECT id, title, source_label, source_feed, published_at, feed_decision
            FROM articles
            WHERE feed_decision IN ('skip', 'summarize')
        """,
        "min_total": 20,
        "min_per_class": 5,
        "timestamp_column": "feed_decision_at",
    },
    "summary_interest": {
        "artifact_name": "summary_interest.joblib",
        "positive_label": "interesting",
        "query": """
            SELECT id, summary_title, summary_text, source_label, source_feed, published_at, summary_feedback
            FROM articles
            WHERE summary_feedback IN ('interesting', 'not_interesting')
              AND LENGTH(COALESCE(summary_text, '')) > 0
        """,
        "min_total": 20,
        "min_per_class": 5,
        "timestamp_column": "summary_feedback_at",
    },
}


class RuntimeState:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.refresh_lock = threading.Lock()
        self.training_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.models: dict[str, dict[str, Any]] = {}


STATE = RuntimeState()
APP = Flask(__name__)
APP.logger.setLevel(logging.WARNING)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def ensure_dirs() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    ensure_dirs()
    with db_connection() as conn:
        conn.executescript(SCHEMA_SQL)


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def parse_datetime(value: Any) -> datetime:
    if not value:
        return utc_now()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if hasattr(value, "tm_year"):
        return datetime(*value[:6], tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).astimezone(timezone.utc)
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except (TypeError, ValueError):
                return utc_now()
    return utc_now()


def format_source_label(url: str) -> str:
    if not url:
        return "Unknown source"
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    if host.startswith("www."):
        host = host[4:]
    return host or "Unknown source"


def parse_legacy_timestamp(value: Any) -> str:
    if isinstance(value, dict) and "seconds" in value:
        try:
            seconds = float(value["seconds"])
            return datetime.fromtimestamp(seconds, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OSError):
            return utc_now_iso()
    parsed = parse_datetime(value)
    return parsed.isoformat()


def normalize_legacy_entry(raw: dict[str, Any]) -> Optional[dict[str, Any]]:
    preference = raw.get("_preference") or raw.get("preference")
    if preference not in {"interesting", "not_interesting"}:
        return None

    title = (raw.get("title") or raw.get("summary_title") or "").strip()
    if not title:
        return None

    source_url = (raw.get("source_url") or "").strip()
    link_to_article = (raw.get("rss_source_url") or raw.get("link_to_article") or "").strip()
    tracked_at = parse_legacy_timestamp(raw.get("_trackedAt") or raw.get("tracked_at") or raw.get("feed_decision_at"))
    published_at = parse_legacy_timestamp(raw.get("published_at") or tracked_at)
    legacy_id = str(raw.get("_legacy_id") or raw.get("id") or raw.get("guid") or "").strip()

    source_label = (raw.get("source_label") or "").strip() or format_source_label(source_url)
    source_feed = (raw.get("source_feed") or "").strip() or "legacy_import"

    return {
        "legacy_id": legacy_id,
        "title": title,
        "source_url": source_url,
        "source_label": source_label,
        "source_feed": source_feed,
        "link_to_article": link_to_article,
        "published_at": published_at,
        "tracked_at": tracked_at,
        "preference": preference,
    }


def parse_legacy_json_entries(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    records: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("entries"), list):
            raw_entries = payload["entries"]
        else:
            raw_entries = []
            for key, value in payload.items():
                if isinstance(value, dict):
                    item = dict(value)
                    item.setdefault("_legacy_id", key)
                    raw_entries.append(item)
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raw_entries = []

    for raw in raw_entries:
        if not isinstance(raw, dict):
            continue
        normalized = normalize_legacy_entry(raw)
        if normalized:
            records.append(normalized)

    return records


def parse_legacy_markdown_entries(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    current_section: Optional[str] = None
    current_record: Optional[dict[str, Any]] = None
    bullet_re = re.compile(r"^- \*\*(.+?):\*\* ?(.*)$")

    def flush_record() -> None:
        nonlocal current_record
        if not current_record:
            return
        normalized = normalize_legacy_entry(current_record)
        if normalized:
            records.append(normalized)
        current_record = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line == "## Interessant":
            flush_record()
            current_section = "interesting"
            continue
        if line == "## Nicht interessant":
            flush_record()
            current_section = "not_interesting"
            continue
        if line.startswith("### "):
            flush_record()
            current_record = {
                "title": line[4:].strip(),
                "_preference": current_section,
            }
            continue

        if current_record is None:
            continue

        match = bullet_re.match(line)
        if not match:
            continue

        key = match.group(1).strip()
        value = match.group(2).strip()
        if key == "Getrackt":
            current_record["_trackedAt"] = value
        else:
            current_record[key] = value

    flush_record()
    return records


def parse_legacy_entries(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    try:
        return parse_legacy_json_entries(stripped)
    except json.JSONDecodeError:
        return parse_legacy_markdown_entries(stripped)


def build_legacy_guid(entry: dict[str, Any]) -> str:
    if entry["legacy_id"]:
        return f"legacy:{entry['legacy_id']}"
    fingerprint = "|".join(
        [
            entry["title"],
            entry["source_url"],
            entry["link_to_article"],
            entry["published_at"],
            entry["preference"],
        ]
    )
    digest = hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()
    return f"legacy:{digest}"


def find_matching_article(conn: sqlite3.Connection, entry: dict[str, Any]) -> Optional[sqlite3.Row]:
    if entry["link_to_article"]:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE link_to_article = ? OR rss_source_url = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (entry["link_to_article"], entry["link_to_article"]),
        ).fetchone()
        if row:
            return row

    if entry["source_url"]:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE title = ? AND source_url = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (entry["title"], entry["source_url"]),
        ).fetchone()
        if row:
            return row

    if entry["source_label"]:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE title = ? AND source_label = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (entry["title"], entry["source_label"]),
        ).fetchone()
        if row:
            return row

    return conn.execute(
        """
        SELECT *
        FROM articles
        WHERE title = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (entry["title"],),
    ).fetchone()


def import_legacy_preferences(raw_text: str, auto_train: bool = True) -> dict[str, Any]:
    entries = parse_legacy_entries(raw_text)
    if not entries:
        return {
            "status": "error",
            "message": "No importable legacy labels found",
            "imported": 0,
        }

    inserted = 0
    updated = 0
    matched = 0

    with db_connection() as conn:
        for entry in entries:
            decision = "summarize" if entry["preference"] == "interesting" else "skip"
            row = find_matching_article(conn, entry)
            if row:
                matched += 1
                conn.execute(
                    """
                    UPDATE articles
                    SET feed_decision = ?,
                        feed_decision_at = ?,
                        updated_at = ?,
                        source_url = CASE
                            WHEN source_url = '' THEN ?
                            ELSE source_url
                        END,
                        source_label = CASE
                            WHEN source_label = '' THEN ?
                            ELSE source_label
                        END,
                        source_feed = CASE
                            WHEN source_feed = '' THEN ?
                            ELSE source_feed
                        END
                    WHERE id = ?
                    """,
                    (
                        decision,
                        entry["tracked_at"],
                        utc_now_iso(),
                        entry["source_url"],
                        entry["source_label"],
                        entry["source_feed"],
                        row["id"],
                    ),
                )
                log_event(
                    conn,
                    int(row["id"]),
                    "legacy_feed_label_imported",
                    {
                        "decision": decision,
                        "legacy_id": entry["legacy_id"],
                        "tracked_at": entry["tracked_at"],
                    },
                )
                updated += 1
                continue

            guid = build_legacy_guid(entry)
            now = utc_now_iso()
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO articles(
                    guid, title, link_to_article, rss_source_url, source_url,
                    source_label, source_feed, published_at, created_at, updated_at,
                    feed_decision, feed_decision_at, summary_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guid,
                    entry["title"],
                    entry["link_to_article"],
                    entry["link_to_article"],
                    entry["source_url"],
                    entry["source_label"],
                    entry["source_feed"],
                    entry["published_at"],
                    entry["tracked_at"] or now,
                    now,
                    decision,
                    entry["tracked_at"],
                    "not_requested",
                ),
            )
            article_id = cursor.lastrowid
            if article_id:
                inserted += 1
                log_event(
                    conn,
                    int(article_id),
                    "legacy_feed_label_imported",
                    {
                        "decision": decision,
                        "legacy_id": entry["legacy_id"],
                        "tracked_at": entry["tracked_at"],
                    },
                )

    train_results: list[dict[str, Any]] = []
    if auto_train:
        train_results = train_targets(["feed_recommendation"])

    return {
        "status": "ok",
        "imported": len(entries),
        "matched_existing": matched,
        "updated_existing": updated,
        "inserted_new": inserted,
        "train_results": train_results,
    }


def update_app_state(key: str, value: Any) -> None:
    with db_connection() as conn:
        conn.execute(
            """
            INSERT INTO app_state(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (key, json.dumps(value), utc_now_iso()),
        )


def get_app_state(key: str, default: Any = None) -> Any:
    with db_connection() as conn:
        row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except json.JSONDecodeError:
        return default


def log_event(
    conn: sqlite3.Connection,
    article_id: int,
    event_type: str,
    payload: Optional[dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO article_events(article_id, event_type, event_payload, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            article_id,
            event_type,
            json.dumps(payload or {}, ensure_ascii=True),
            utc_now_iso(),
        ),
    )


def get_source_url_from_entry(entry: Any) -> str:
    if hasattr(entry, "source") and hasattr(entry.source, "href"):
        return entry.source.href or ""
    return ""


def parse_entry_published_at(entry: Any) -> str:
    source = None
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        source = entry.published_parsed
    elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
        source = entry.updated_parsed
    elif hasattr(entry, "published") and entry.published:
        source = entry.published
    elif hasattr(entry, "updated") and entry.updated:
        source = entry.updated
    return parse_datetime(source).isoformat()


def decode_google_news_url(url: str) -> str:
    if not url or "news.google.com" not in url or gnewsdecoder is None:
        return url
    try:
        decoded = gnewsdecoder(url, interval=1)
        if decoded.get("status") and decoded.get("decoded_url"):
            return decoded["decoded_url"]
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        LOGGER.warning("Could not decode Google News URL %s: %s", url, exc)
    return url


def fallback_extract_text(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    pieces = []
    for paragraph in soup.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if text:
            pieces.append(text)
    return "\n\n".join(pieces)


def fetch_and_extract_article_text(initial_url: str) -> tuple[str, str]:
    resolved_input = decode_google_news_url(initial_url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    response = requests.get(
        resolved_input,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers=headers,
        allow_redirects=True,
    )
    response.raise_for_status()
    final_url = response.url
    extracted = None
    if trafilatura is not None:
        extracted = trafilatura.extract(
            response.text,
            url=final_url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    if not extracted or len(extracted.strip()) < 400:
        extracted = fallback_extract_text(response.text)
    cleaned = (extracted or "").strip()
    return cleaned[:24000], final_url


def extract_json_blob(text: str) -> Optional[dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def parse_summary_response(text: str) -> tuple[str, str]:
    parsed = extract_json_blob(text)
    if parsed:
        title = str(parsed.get("title", "")).strip()
        summary = str(parsed.get("summary", "")).strip()
        if title and summary:
            return title, summary

    title = ""
    summary = ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        lower = line.lower()
        if lower.startswith("titel:") or lower.startswith("title:"):
            title = line.split(":", 1)[1].strip()
        if lower.startswith("zusammenfassung:") or lower.startswith("summary:"):
            summary = line.split(":", 1)[1].strip()
            if index + 1 < len(lines):
                rest = "\n".join(lines[index + 1 :]).strip()
                if rest:
                    summary = f"{summary}\n{rest}".strip()
            break

    if not title and lines:
        title = lines[0][:160].strip()
    if not summary and len(lines) > 1:
        summary = "\n".join(lines[1:]).strip()
    if not summary:
        summary = text.strip()
    return title, summary


def ollama_generate_summary(article_title: str, article_text: str) -> tuple[str, str]:
    prompt = f"""
Du bist ein präziser Nachrichtenassistent.
Erzeuge einen deutschen Titel und eine deutsche Zusammenfassung.

Regeln:
- sachlich, knapp, klar
- keine Einleitung wie "Der Artikel beschreibt"
- kein Verweis auf Zeitung, Portal oder Quelle
- der Titel soll wie eine kurze News-Headline klingen
- die Zusammenfassung soll 3 bis 5 Sätze lang sein
- liefere nur JSON

JSON-Format:
{{
  "title": "...",
  "summary": "..."
}}

ARTIKELTITEL:
{article_title}

ARTIKELTEXT:
{article_text}
""".strip()

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        timeout=180,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
            },
        },
    )
    response.raise_for_status()
    body = response.json()
    title, summary = parse_summary_response(body.get("response", ""))
    if not title or not summary:
        raise RuntimeError("Ollama response could not be parsed into title and summary")
    return title[:220], summary[:6000]


def build_feature_text(target: str, row: sqlite3.Row | dict[str, Any]) -> str:
    record = row if isinstance(row, dict) else row_to_dict(row)
    published = parse_datetime(record.get("published_at"))
    if target == "feed_recommendation":
        return "\n".join(
            [
                f"title {record.get('title', '')}",
                f"source {record.get('source_label', '')}",
                f"feed {record.get('source_feed', '')}",
                f"weekday {published.weekday()}",
                f"hour {published.hour}",
            ]
        )
    return "\n".join(
        [
            f"title {record.get('summary_title', '')}",
            f"summary {record.get('summary_text', '')}",
            f"source {record.get('source_label', '')}",
            f"feed {record.get('source_feed', '')}",
            f"weekday {published.weekday()}",
            f"hour {published.hour}",
        ]
    )


def latest_model_run(target: str) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM model_runs
            WHERE target = ?
            ORDER BY datetime(trained_at) DESC, id DESC
            LIMIT 1
            """,
            (target,),
        ).fetchone()
    return row_to_dict(row) if row else None


def load_persisted_models() -> None:
    with STATE.model_lock:
        STATE.models.clear()
        for target, config in TARGET_CONFIG.items():
            path = MODEL_DIR / config["artifact_name"]
            if not path.exists():
                continue
            try:
                artifact = joblib.load(path)
            except Exception as exc:  # pragma: no cover - runtime defensive
                LOGGER.warning("Could not load model %s: %s", path, exc)
                continue
            STATE.models[target] = artifact


def get_loaded_model(target: str) -> Optional[dict[str, Any]]:
    with STATE.model_lock:
        return STATE.models.get(target)


def predict_feed_rows(rows: list[sqlite3.Row]) -> tuple[list[dict[str, Any]], Optional[int]]:
    if not rows:
        return [], None

    artifact = get_loaded_model("feed_recommendation")
    if not artifact:
        items = []
        for row in rows:
            item = row_to_dict(row)
            item["prediction"] = {
                "available": False,
                "recommended": None,
                "probability": None,
                "run_id": None,
            }
            items.append(item)
        return items, None

    pipeline: Pipeline = artifact["pipeline"]
    features = [build_feature_text("feed_recommendation", row) for row in rows]
    probabilities = pipeline.predict_proba(features)[:, 1]
    items = []
    for row, probability in zip(rows, probabilities):
        item = row_to_dict(row)
        recommended = bool(probability >= artifact["threshold"])
        item["prediction"] = {
            "available": True,
            "recommended": recommended,
            "probability": round(float(probability), 4),
            "run_id": artifact["run_id"],
        }
        items.append(item)
    return items, artifact["run_id"]


def update_cached_feed_predictions(rows: list[dict[str, Any]], run_id: Optional[int]) -> None:
    if not rows or run_id is None:
        return
    with db_connection() as conn:
        for item in rows:
            prediction = item["prediction"]
            conn.execute(
                """
                UPDATE articles
                SET predicted_recommendation = ?,
                    predicted_probability = ?,
                    prediction_model_run_id = ?,
                    prediction_generated_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    1 if prediction["recommended"] else 0,
                    prediction["probability"],
                    run_id,
                    utc_now_iso(),
                    utc_now_iso(),
                    item["id"],
                ),
            )


def refresh_feeds() -> dict[str, Any]:
    if not STATE.refresh_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Feed refresh already running"}

    inserted = 0
    updated = 0
    feed_errors: list[str] = []
    started_at = utc_now_iso()
    try:
        LOGGER.info("Starting RSS refresh")
        with db_connection() as conn:
            for feed_url in RSS_FEED_URLS:
                try:
                    feed = feedparser.parse(feed_url)
                except Exception as exc:
                    feed_errors.append(f"{feed_url}: {exc}")
                    LOGGER.warning("Could not parse feed %s: %s", feed_url, exc)
                    continue

                for entry in feed.entries:
                    guid = entry.get("id") or entry.get("guid") or entry.get("link")
                    if not guid:
                        continue
                    title = html_lib.unescape(entry.get("title", "No title")).strip() or "No title"
                    link = entry.get("link", "") or ""
                    source_url = get_source_url_from_entry(entry) or link
                    published_at = parse_entry_published_at(entry)
                    now = utc_now_iso()
                    existing = conn.execute(
                        "SELECT id FROM articles WHERE guid = ?",
                        (guid,),
                    ).fetchone()
                    if existing:
                        conn.execute(
                            """
                            UPDATE articles
                            SET title = ?,
                                link_to_article = ?,
                                rss_source_url = ?,
                                source_url = ?,
                                source_label = ?,
                                source_feed = ?,
                                published_at = ?,
                                updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                title,
                                link,
                                link,
                                source_url,
                                format_source_label(source_url),
                                feed_url,
                                published_at,
                                now,
                                existing["id"],
                            ),
                        )
                        updated += 1
                        continue

                    cursor = conn.execute(
                        """
                        INSERT INTO articles(
                            guid, title, link_to_article, rss_source_url, source_url,
                            source_label, source_feed, published_at, created_at, updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            guid,
                            title,
                            link,
                            link,
                            source_url,
                            format_source_label(source_url),
                            feed_url,
                            published_at,
                            now,
                            now,
                        ),
                    )
                    article_id = cursor.lastrowid
                    log_event(
                        conn,
                        int(article_id),
                        "article_ingested",
                        {"feed_url": feed_url, "guid": guid},
                    )
                    inserted += 1

        result = {
            "status": "ok",
            "inserted": inserted,
            "updated": updated,
            "errors": feed_errors,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        }
        update_app_state("last_feed_refresh", result)
        LOGGER.info("RSS refresh finished: %s inserted, %s updated", inserted, updated)
        return result
    finally:
        STATE.refresh_lock.release()


def set_feed_decision(article_id: int, decision: str) -> dict[str, Any]:
    if decision not in {"skip", "summarize"}:
        raise ValueError("Unsupported feed decision")
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if not row:
            raise LookupError("Article not found")

        row_dict = row_to_dict(row)
        now = utc_now_iso()
        summary_status = row_dict["summary_status"]
        summary_requested_at = row_dict["summary_requested_at"]

        if decision == "skip" and summary_status in {"queued", "processing", "ready"}:
            raise RuntimeError("Article is already in summary flow")

        if decision == "summarize":
            if summary_status in {"not_requested", "failed"}:
                summary_status = "queued"
                summary_requested_at = now
            event_name = "summary_requested"
        else:
            event_name = "article_skipped"

        conn.execute(
            """
            UPDATE articles
            SET feed_decision = ?,
                feed_decision_at = ?,
                summary_status = ?,
                summary_requested_at = ?,
                updated_at = ?,
                last_error = CASE WHEN ? = 'summarize' THEN '' ELSE last_error END
            WHERE id = ?
            """,
            (
                decision,
                now,
                summary_status,
                summary_requested_at,
                now,
                decision,
                article_id,
            ),
        )
        log_event(
            conn,
            article_id,
            event_name,
            {
                "previous_feed_decision": row_dict["feed_decision"],
                "new_feed_decision": decision,
            },
        )
    return {"status": "ok", "article_id": article_id, "decision": decision}


def archive_pending_feed() -> dict[str, Any]:
    now = utc_now_iso()
    archived_ids: list[int] = []

    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM articles
            WHERE feed_decision = 'pending'
            ORDER BY id
            """
        ).fetchall()
        archived_ids = [int(row["id"]) for row in rows]

        if not archived_ids:
            result = {
                "status": "ok",
                "archived_count": 0,
                "archived_at": now,
            }
            update_app_state("last_feed_reset", result)
            return result

        conn.execute(
            """
            UPDATE articles
            SET feed_decision = 'archived',
                feed_decision_at = ?,
                updated_at = ?
            WHERE feed_decision = 'pending'
            """,
            (now, now),
        )

        for article_id in archived_ids:
            log_event(
                conn,
                article_id,
                "feed_archived",
                {"archived_at": now},
            )

    result = {
        "status": "ok",
        "archived_count": len(archived_ids),
        "archived_at": now,
    }
    update_app_state("last_feed_reset", result)
    return result


def set_summary_feedback(article_id: int, feedback: str) -> dict[str, Any]:
    if feedback not in {"interesting", "not_interesting"}:
        raise ValueError("Unsupported summary feedback")
    with db_connection() as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if not row:
            raise LookupError("Article not found")
        now = utc_now_iso()
        conn.execute(
            """
            UPDATE articles
            SET summary_feedback = ?,
                summary_feedback_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (feedback, now, now, article_id),
        )
        log_event(conn, article_id, "summary_feedback", {"feedback": feedback})
    return {"status": "ok", "article_id": article_id, "feedback": feedback}


def get_next_summary_job() -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE summary_status = 'queued'
            ORDER BY datetime(summary_requested_at) ASC, id ASC
            LIMIT 1
            """,
        ).fetchone()
        if not row:
            return None
        row_dict = row_to_dict(row)
        conn.execute(
            """
            UPDATE articles
            SET summary_status = 'processing',
                updated_at = ?
            WHERE id = ?
            """,
            (utc_now_iso(), row_dict["id"]),
        )
        log_event(conn, row_dict["id"], "summary_processing_started", {})
    return row_dict


def process_summary_job(job: dict[str, Any]) -> None:
    article_id = int(job["id"])
    try:
        article_text, final_url = fetch_and_extract_article_text(job["rss_source_url"] or job["link_to_article"])
        if not article_text or len(article_text.strip()) < 300:
            raise RuntimeError("Article extraction returned too little text")

        summary_title, summary_text = ollama_generate_summary(job["title"], article_text)
        now = utc_now_iso()
        with db_connection() as conn:
            conn.execute(
                """
                UPDATE articles
                SET link_to_article = ?,
                    article_text = ?,
                    article_text_extracted_at = ?,
                    summary_title = ?,
                    summary_text = ?,
                    summary_model = ?,
                    summary_status = 'ready',
                    summarized_at = ?,
                    updated_at = ?,
                    last_error = ''
                WHERE id = ?
                """,
                (
                    final_url,
                    article_text,
                    now,
                    summary_title,
                    summary_text,
                    OLLAMA_MODEL,
                    now,
                    now,
                    article_id,
                ),
            )
            log_event(
                conn,
                article_id,
                "summary_generated",
                {"model": OLLAMA_MODEL, "final_url": final_url},
            )
    except Exception as exc:  # pragma: no cover - runtime defensive
        LOGGER.warning("Summary job failed for article %s: %s", article_id, exc)
        with db_connection() as conn:
            conn.execute(
                """
                UPDATE articles
                SET summary_status = 'failed',
                    updated_at = ?,
                    last_error = ?
                WHERE id = ?
                """,
                (utc_now_iso(), str(exc), article_id),
            )
            log_event(
                conn,
                article_id,
                "summary_failed",
                {"error": str(exc)},
            )


def summary_worker() -> None:
    LOGGER.info("Summary worker started")
    while not STATE.stop_event.is_set():
        job = get_next_summary_job()
        if job is None:
            STATE.stop_event.wait(SUMMARY_POLL_SECONDS)
            continue
        process_summary_job(job)


def feed_refresh_worker() -> None:
    LOGGER.info("Feed refresh worker started")
    refresh_feeds()
    while not STATE.stop_event.wait(FEED_REFRESH_SECONDS):
        try:
            refresh_feeds()
        except Exception as exc:  # pragma: no cover - runtime defensive
            LOGGER.warning("Periodic feed refresh failed: %s", exc)


def count_rows(where_sql: str, params: tuple[Any, ...] = ()) -> int:
    with db_connection() as conn:
        row = conn.execute(f"SELECT COUNT(*) AS total FROM articles WHERE {where_sql}", params).fetchone()
    return int(row["total"])


def fetch_pending_feed_articles() -> list[sqlite3.Row]:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE feed_decision = 'pending'
            ORDER BY datetime(published_at) DESC, id DESC
            """
        ).fetchall()
    return list(rows)


def fetch_ready_summaries() -> list[dict[str, Any]]:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE summary_status = 'ready'
              AND summary_feedback = 'unreviewed'
            ORDER BY datetime(summarized_at) DESC, id DESC
            """
        ).fetchall()
    return [row_to_dict(row) for row in rows]


def latest_labels_count(target: str) -> dict[str, Any]:
    config = TARGET_CONFIG[target]
    with db_connection() as conn:
        rows = conn.execute(config["query"]).fetchall()
    if target == "feed_recommendation":
        positives = sum(1 for row in rows if row["feed_decision"] == "summarize")
    else:
        positives = sum(1 for row in rows if row["summary_feedback"] == "interesting")
    negatives = len(rows) - positives
    return {"total": len(rows), "positive": positives, "negative": negatives}


def new_labels_since_training(target: str, trained_at: Optional[str]) -> int:
    if not trained_at:
        return latest_labels_count(target)["total"]
    timestamp_column = TARGET_CONFIG[target]["timestamp_column"]
    if target == "feed_recommendation":
        where_sql = f"""
            {timestamp_column} IS NOT NULL
            AND datetime({timestamp_column}) > datetime(?)
            AND feed_decision IN ('skip', 'summarize')
        """
    else:
        where_sql = f"""
            {timestamp_column} IS NOT NULL
            AND datetime({timestamp_column}) > datetime(?)
            AND summary_feedback IN ('interesting', 'not_interesting')
        """
    return count_rows(where_sql, (trained_at,))


def retraining_recommendation(target: str, latest_run: Optional[dict[str, Any]]) -> dict[str, Any]:
    counts = latest_labels_count(target)
    if counts["total"] == 0:
        return {
            "recommended": False,
            "reason": "no_labels",
            "headline": "Noch keine Labels vorhanden",
            "detail": "Sobald du erste Entscheidungen gesammelt hast, lohnt sich ein erstes Training.",
        }

    if latest_run is None:
        ready = (
            counts["total"] >= TARGET_CONFIG[target]["min_total"]
            and counts["positive"] >= TARGET_CONFIG[target]["min_per_class"]
            and counts["negative"] >= TARGET_CONFIG[target]["min_per_class"]
        )
        return {
            "recommended": ready,
            "reason": "initial_training_ready" if ready else "not_enough_labels",
            "headline": "Erstes Training lohnt sich jetzt" if ready else "Für ein erstes Training noch warten",
            "detail": (
                "Es sind genug positive und negative Labels vorhanden."
                if ready
                else "Für ein stabiles erstes Modell braucht es noch mehr Daten in beiden Klassen."
            ),
        }

    delta = new_labels_since_training(target, latest_run["trained_at"])
    threshold = max(10, int(latest_run["labels_used"] * 0.15))
    return {
        "recommended": delta >= threshold,
        "reason": "enough_new_labels" if delta >= threshold else "delta_too_small",
        "headline": (
            "Neues Training lohnt sich"
            if delta >= threshold
            else "Mit Retraining noch warten"
        ),
        "detail": (
            f"Seit dem letzten Training kamen {delta} neue Labels dazu. Ab etwa {threshold} lohnt sich der nächste Lauf."
        ),
    }


def model_quality_assessment(target: str, latest_run: Optional[dict[str, Any]]) -> dict[str, Any]:
    if latest_run is None:
        return {
            "level": "none",
            "headline": "Noch kein Modell trainiert",
            "detail": "Aktuell gibt es nur rohe Labels, aber noch keine Modellgüte.",
        }

    accuracy = float(latest_run.get("accuracy", 0.0))
    precision = float(latest_run.get("precision", 0.0))
    recall = float(latest_run.get("recall", 0.0))
    f1 = float(latest_run.get("f1", 0.0))
    labels_used = int(latest_run.get("labels_used", 0))
    positive_labels = int(latest_run.get("positive_labels", 0))

    if target == "feed_recommendation":
        if labels_used < 400 or positive_labels < 200:
            return {
                "level": "early",
                "headline": "Nur als Ranking-Signal nutzen",
                "detail": "Es gibt noch zu wenig aktuelle positive Beispiele für verlässliches hartes Filtering.",
            }
        if f1 < 0.35 or precision < 0.35 or recall < 0.35:
            return {
                "level": "weak",
                "headline": "Noch nicht für hartes Filtering geeignet",
                "detail": "Die Trefferqualität ist noch zu schwach. Im Feed besser weiterhin alle Einträge sehen und den Score nur zur Priorisierung nutzen.",
            }
        if f1 < 0.55:
            return {
                "level": "usable",
                "headline": "Vorsichtig nutzbar",
                "detail": "Das Modell taugt als Priorisierung und für experimentelles Filtern, aber noch nicht für aggressive Automatik.",
            }
        return {
            "level": "strong",
            "headline": "Für selektiveres Filtering brauchbar",
            "detail": "Das Modell zeigt inzwischen genug Balance zwischen Precision und Recall für vorsichtigeren produktiven Einsatz.",
        }

    if labels_used < 100:
        return {
            "level": "early",
            "headline": "Noch frühe Einschätzung",
            "detail": "Für das Summary-Modell sind noch relativ wenig gelabelte Beispiele vorhanden.",
        }
    if f1 < 0.4:
        return {
            "level": "weak",
            "headline": "Noch instabil",
            "detail": "Die Summary-Relevanz ist noch nicht stabil genug modelliert.",
        }
    return {
        "level": "usable",
        "headline": "Brauchbare erste Orientierung",
        "detail": "Das Summary-Modell liefert eine erste nutzbare Signalqualität.",
    }


def train_model(target: str) -> dict[str, Any]:
    if target not in TARGET_CONFIG:
        raise ValueError(f"Unsupported target {target}")

    config = TARGET_CONFIG[target]
    with db_connection() as conn:
        rows = conn.execute(config["query"]).fetchall()

    if target == "feed_recommendation":
        labels = [1 if row["feed_decision"] == "summarize" else 0 for row in rows]
    else:
        labels = [1 if row["summary_feedback"] == "interesting" else 0 for row in rows]

    positives = sum(labels)
    negatives = len(labels) - positives
    if (
        len(labels) < config["min_total"]
        or positives < config["min_per_class"]
        or negatives < config["min_per_class"]
    ):
        return {
            "target": target,
            "status": "insufficient_labels",
            "labels_used": len(labels),
            "positive_labels": positives,
            "negative_labels": negatives,
        }

    feature_texts = [build_feature_text(target, row) for row in rows]
    train_x, test_x, train_y, test_y = train_test_split(
        feature_texts,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=6000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(train_x, train_y)
    predicted = pipeline.predict(test_x)
    matrix = confusion_matrix(test_y, predicted, labels=[0, 1]).tolist()
    accuracy = float(accuracy_score(test_y, predicted))
    precision = float(precision_score(test_y, predicted, zero_division=0))
    recall = float(recall_score(test_y, predicted, zero_division=0))
    f1 = float(f1_score(test_y, predicted, zero_division=0))

    trained_at = utc_now_iso()
    model_path = MODEL_DIR / config["artifact_name"]
    with db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO model_runs(
                target, model_path, trained_at, labels_used, train_size, test_size,
                positive_labels, negative_labels, accuracy, precision, recall, f1,
                threshold_value, confusion_matrix_json, notes, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                target,
                str(model_path),
                trained_at,
                len(labels),
                len(train_y),
                len(test_y),
                positives,
                negatives,
                accuracy,
                precision,
                recall,
                f1,
                0.5,
                json.dumps(matrix),
                "",
                "trained",
            ),
        )
        run_id = int(cursor.lastrowid)

    artifact = {
        "target": target,
        "run_id": run_id,
        "trained_at": trained_at,
        "threshold": 0.5,
        "pipeline": pipeline,
    }
    joblib.dump(artifact, model_path)

    with STATE.model_lock:
        STATE.models[target] = artifact

    if target == "feed_recommendation":
        rows_for_prediction = fetch_pending_feed_articles()
        predicted_rows, used_run_id = predict_feed_rows(rows_for_prediction)
        update_cached_feed_predictions(predicted_rows, used_run_id)

    return {
        "target": target,
        "status": "trained",
        "run_id": run_id,
        "trained_at": trained_at,
        "labels_used": len(labels),
        "positive_labels": positives,
        "negative_labels": negatives,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion_matrix": matrix,
    }


def train_targets(targets: list[str]) -> list[dict[str, Any]]:
    results = []
    with STATE.training_lock:
        for target in targets:
            results.append(train_model(target))
    return results


def serialize_article_for_feed(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "title": item["title"],
        "source_label": item["source_label"],
        "source_url": item["source_url"],
        "source_feed": item["source_feed"],
        "published_at": item["published_at"],
        "link_to_article": item["link_to_article"],
        "feed_decision": item["feed_decision"],
        "prediction": item["prediction"],
    }


def serialize_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "title": item["title"],
        "summary_title": item["summary_title"],
        "summary_text": item["summary_text"],
        "source_label": item["source_label"],
        "source_url": item["source_url"],
        "published_at": item["published_at"],
        "link_to_article": item["link_to_article"],
        "summary_model": item["summary_model"],
        "summary_feedback": item["summary_feedback"],
    }


def summary_counts() -> dict[str, int]:
    return {
        "queued": count_rows("summary_status = 'queued'"),
        "processing": count_rows("summary_status = 'processing'"),
        "ready": count_rows("summary_status = 'ready' AND summary_feedback = 'unreviewed'"),
        "failed": count_rows("summary_status = 'failed'"),
    }


def feed_counts() -> dict[str, int]:
    all_pending_rows = fetch_pending_feed_articles()
    predicted_rows, _ = predict_feed_rows(all_pending_rows)
    recommended = sum(1 for row in predicted_rows if row["prediction"]["recommended"])
    return {
        "pending": len(all_pending_rows),
        "recommended": recommended,
        "labeled_skip": count_rows("feed_decision = 'skip'"),
        "labeled_summarize": count_rows("feed_decision = 'summarize'"),
        "archived": count_rows("feed_decision = 'archived'"),
    }


def ollama_health() -> dict[str, Any]:
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        models = response.json().get("models", [])
        available = any(model.get("name") == OLLAMA_MODEL for model in models)
        return {"reachable": True, "model_available": available, "model_count": len(models)}
    except Exception:
        return {"reachable": False, "model_available": False, "model_count": 0}


@APP.after_request
def add_cors_headers(response):  # type: ignore[no-untyped-def]
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@APP.errorhandler(Exception)
def handle_unexpected_exception(error):  # type: ignore[no-untyped-def]
    if isinstance(error, HTTPException):
        return jsonify({"status": "error", "message": error.description}), error.code

    LOGGER.exception("Unhandled API error: %s", error)
    return jsonify({"status": "error", "message": str(error) or "Internal server error"}), 500


@APP.before_request
def handle_options():  # type: ignore[no-untyped-def]
    if request.method == "OPTIONS":
        return ("", 204)
    return None


@APP.get("/api/health")
def api_health():
    return jsonify(
        {
            "status": "ok",
            "db_path": str(DB_PATH),
            "ollama": ollama_health(),
            "version": "local-v1",
        }
    )


@APP.get("/api/status")
def api_status():
    latest_feed_run = latest_model_run("feed_recommendation")
    latest_summary_run = latest_model_run("summary_interest")
    return jsonify(
        {
            "feed": feed_counts(),
            "summaries": summary_counts(),
            "models": {
                "feed_recommendation": {
                    "loaded": get_loaded_model("feed_recommendation") is not None,
                    "latest_run": latest_feed_run,
                },
                "summary_interest": {
                    "loaded": get_loaded_model("summary_interest") is not None,
                    "latest_run": latest_summary_run,
                },
            },
            "ollama": ollama_health(),
            "last_feed_refresh": get_app_state("last_feed_refresh"),
            "config": {
                "config_path": str(CONFIG_PATH),
                "ollama_model": OLLAMA_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
                "feed_refresh_seconds": FEED_REFRESH_SECONDS,
                "feed_count": len(RSS_FEED_URLS),
            },
        }
    )


@APP.post("/api/feeds/refresh")
def api_refresh_feeds():
    return jsonify(refresh_feeds())


@APP.post("/api/feed/reset")
def api_feed_reset():
    return jsonify(archive_pending_feed())


@APP.get("/api/feed")
def api_feed():
    mode = request.args.get("mode", "all")
    rows = fetch_pending_feed_articles()
    predicted_rows, run_id = predict_feed_rows(rows)
    update_cached_feed_predictions(predicted_rows, run_id)

    if mode == "recommended":
        filtered = [row for row in predicted_rows if row["prediction"]["recommended"]]
    else:
        filtered = predicted_rows

    return jsonify(
        {
            "mode": mode,
            "items": [serialize_article_for_feed(item) for item in filtered],
            "counts": {
                "total_pending": len(predicted_rows),
                "recommended_pending": sum(
                    1 for row in predicted_rows if row["prediction"]["recommended"]
                ),
            },
        }
    )


@APP.post("/api/articles/<int:article_id>/skip")
def api_skip_article(article_id: int):
    try:
        return jsonify(set_feed_decision(article_id, "skip"))
    except LookupError:
        return jsonify({"status": "error", "message": "Article not found"}), 404
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 409


@APP.post("/api/articles/<int:article_id>/summarize")
def api_summarize_article(article_id: int):
    try:
        return jsonify(set_feed_decision(article_id, "summarize"))
    except LookupError:
        return jsonify({"status": "error", "message": "Article not found"}), 404


@APP.get("/api/summaries")
def api_summaries():
    items = fetch_ready_summaries()
    return jsonify(
        {
            "items": [serialize_summary(item) for item in items],
            "counts": summary_counts(),
        }
    )


@APP.post("/api/summaries/<int:article_id>/feedback")
def api_summary_feedback(article_id: int):
    body = request.get_json(silent=True) or {}
    feedback = body.get("feedback")
    try:
        return jsonify(set_summary_feedback(article_id, feedback))
    except ValueError:
        return jsonify({"status": "error", "message": "Unsupported feedback"}), 400
    except LookupError:
        return jsonify({"status": "error", "message": "Article not found"}), 404


@APP.get("/api/model-ops")
def api_model_ops():
    payload: dict[str, Any] = {"targets": {}}
    for target in TARGET_CONFIG:
        latest_run = latest_model_run(target)
        counts = latest_labels_count(target)
        payload["targets"][target] = {
            "latest_run": latest_run,
            "label_counts": counts,
            "new_labels_since_training": new_labels_since_training(
                target,
                latest_run["trained_at"] if latest_run else None,
            ),
            "retraining": retraining_recommendation(target, latest_run),
            "quality": model_quality_assessment(target, latest_run),
        }
        if latest_run and latest_run.get("confusion_matrix_json"):
            payload["targets"][target]["latest_run"]["confusion_matrix"] = json.loads(
                latest_run["confusion_matrix_json"]
            )
    return jsonify(payload)


@APP.post("/api/legacy/import")
def api_legacy_import():
    body = request.get_json(silent=True) or {}
    payload = body.get("payload", "")
    auto_train = bool(body.get("auto_train", True))
    result = import_legacy_preferences(payload, auto_train=auto_train)
    if result.get("status") == "error":
        return jsonify(result), 400
    return jsonify(result)


@APP.post("/api/model-ops/train")
def api_model_train():
    body = request.get_json(silent=True) or {}
    target = body.get("target", "all")
    targets = list(TARGET_CONFIG.keys()) if target == "all" else [target]
    try:
        results = train_targets(targets)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "ok", "results": results})


def start_background_threads() -> list[threading.Thread]:
    threads = [
        threading.Thread(target=summary_worker, name="summary-worker", daemon=True),
        threading.Thread(target=feed_refresh_worker, name="feed-refresh-worker", daemon=True),
    ]
    for thread in threads:
        thread.start()
    return threads


def shutdown() -> None:
    STATE.stop_event.set()


def main() -> None:
    init_db()
    load_persisted_models()
    start_background_threads()
    LOGGER.info("Starting local news backend on http://%s:%s", HOST, PORT)
    try:
        APP.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
