#!/usr/bin/env python3
"""Local backend for the News App.

Replaces the old Firebase + Cloud Function stack with:
- SQLite for storage
- RSS polling
- article extraction
- Ollama-based summarization
- simple local classifiers with model metrics

The frontend can be served by this backend or opened directly as a local HTML
file. In both modes it talks to the local API at http://localhost:8765.
"""

from __future__ import annotations

import html as html_lib
import hashlib
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import feedparser
import joblib
import requests
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    fbeta_score,
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
APP_HTML_PATH = BASE_DIR / "local-news-app.html"
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
        "model": "qwen3.6:latest",
        "summary_timeout_seconds": 600,
        "embedding_model": "nomic-embed-text-v2-moe:latest",
        "embedding_timeout_seconds": 120,
    },
    "llm_compare": {
        "enabled": False,
        "export_dir": "compare_exports",
        "request_timeout_seconds": 600,
        "models": [
            "qwen3.5:35b",
            "qwen3.6:latest",
            "gpt-oss:20b",
            "nemotron-3-nano:30b",
        ],
    },
    "timing": {
        "feed_refresh_seconds": 300,
        "summary_poll_seconds": 3,
        "request_timeout_seconds": 20,
        "embedding_poll_seconds": 20,
    },
    "feeds": [
        "https://news.google.com/rss/search?q=%22generative%20ai%22%20OR%20llm%20when%3A1h&hl=en-US&gl=US&ceid=US%3Aen",
        "https://news.google.com/rss/search?q=artificial%20intelligence%20when%3A1h&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=anthropic%20OR%20openai%20OR%20%22google%20gemini%22%20OR%20%22open%20source%20llm%22%20OR%20nvidia%20when%3A1h&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=machine+learning+when:2d&hl=en-US&gl=US&ceid=US:en",
    ],
}

TITLE_SIMILARITY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by", "from",
    "with", "without", "after", "before", "over", "under", "into", "about", "than",
    "is", "are", "was", "were", "be", "being", "been", "as", "it", "its", "their",
    "his", "her", "they", "them", "this", "that", "these", "those", "new", "news",
    "report", "reports", "reportedly", "update", "updates", "says", "say", "said",
    "using", "used", "use", "will", "would", "could", "should", "how", "why", "what",
    "when", "where", "who", "which", "you", "your", "amid", "amidst", "faces",
    "face", "slams", "warns", "warning", "latest", "via", "afterwards", "today",
}
SIMILARITY_LOOKBACK_HOURS = 48
SUMMARY_PROCESSING_STALE_MINUTES = 20


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
COMPARE_EXPORT_DIR = resolve_local_path(SETTINGS["llm_compare"]["export_dir"])
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
MIN_EXTRACTED_ARTICLE_CHARS = 300
EMBEDDING_POLL_SECONDS = float(
    os.environ.get("LOCAL_NEWS_EMBEDDING_POLL_SECONDS", str(SETTINGS["timing"].get("embedding_poll_seconds", 20)))
)
RSS_FEED_URLS = list(SETTINGS["feeds"])
COMPARE_MODELS = list(SETTINGS["llm_compare"]["models"])

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
    summary_is_fallback INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS llm_compare_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enabled_at TEXT NOT NULL,
    disabled_at TEXT,
    export_path TEXT NOT NULL,
    primary_model TEXT NOT NULL,
    models_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS llm_compare_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    article_id INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'ok',
    duration_ms INTEGER NOT NULL DEFAULT 0,
    summary_title TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    error_text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES llm_compare_sessions(id) ON DELETE CASCADE,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE,
    UNIQUE(session_id, article_id, model_name)
);

CREATE TABLE IF NOT EXISTS article_embeddings (
    article_id INTEGER PRIMARY KEY,
    embedding_model TEXT NOT NULL,
    embedding_input_hash TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(article_id) REFERENCES articles(id) ON DELETE CASCADE
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
CREATE INDEX IF NOT EXISTS idx_compare_sessions_status
    ON llm_compare_sessions(status, enabled_at DESC);
CREATE INDEX IF NOT EXISTS idx_compare_results_article
    ON llm_compare_results(session_id, article_id);
CREATE INDEX IF NOT EXISTS idx_article_embeddings_model
    ON article_embeddings(embedding_model, updated_at DESC);
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
        self.training_status_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self.ollama_lock = threading.Lock()
        self.summary_event = threading.Event()
        self.compare_event = threading.Event()
        self.feed_similarity_lock = threading.Lock()
        self.feed_similarity_snapshot: dict[str, Any] = {}
        self.feed_similarity_dirty = True
        self.feed_similarity_building = False
        self.models: dict[str, dict[str, Any]] = {}
        self.training_status: dict[str, Any] = {
            "active": False,
            "target": None,
            "started_at": None,
            "finished_at": None,
            "results": None,
            "error": None,
        }


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
    COMPARE_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(
        DB_PATH,
        timeout=10,
        check_same_thread=False,
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 10000")
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    ensure_dirs()
    with db_connection() as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_SQL)
        migrate_db(conn)


def column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def migrate_db(conn: sqlite3.Connection) -> None:
    if not column_exists(conn, "articles", "summary_is_fallback"):
        conn.execute("ALTER TABLE articles ADD COLUMN summary_is_fallback INTEGER NOT NULL DEFAULT 0")
    if not column_exists(conn, "llm_compare_results", "status"):
        conn.execute("ALTER TABLE llm_compare_results ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'")
    if not column_exists(conn, "llm_compare_results", "duration_ms"):
        conn.execute("ALTER TABLE llm_compare_results ADD COLUMN duration_ms INTEGER NOT NULL DEFAULT 0")
    if not column_exists(conn, "llm_compare_results", "error_text"):
        conn.execute("ALTER TABLE llm_compare_results ADD COLUMN error_text TEXT NOT NULL DEFAULT ''")
    backfill_summary_fallback_flags(conn)


def backfill_summary_fallback_flags(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        UPDATE articles
        SET summary_is_fallback = 1
        WHERE summary_is_fallback = 0
          AND id IN (
              SELECT article_id
              FROM article_events
              WHERE event_type = 'summary_generated'
                AND (
                    event_payload LIKE '%"extraction_fallback": true%'
                    OR event_payload LIKE '%"extraction_fallback":true%'
                )
          )
        """
    )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def fetch_article_by_id(conn: sqlite3.Connection, article_id: int) -> Optional[dict[str, Any]]:
    row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    return row_to_dict(row) if row else None


def fetch_article_events(conn: sqlite3.Connection, article_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM article_events
        WHERE article_id = ?
        ORDER BY id
        """,
        (article_id,),
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_recent_article_events(
    conn: sqlite3.Connection,
    event_types: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not event_types or limit <= 0:
        return []
    placeholders = ",".join("?" for _ in event_types)
    rows = conn.execute(
        f"""
        SELECT *
        FROM article_events
        WHERE event_type IN ({placeholders})
        ORDER BY datetime(created_at) DESC, id DESC
        LIMIT ?
        """,
        (*event_types, limit),
    ).fetchall()
    return [row_to_dict(row) for row in rows]


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


def build_embedding_input_text(article: dict[str, Any]) -> str:
    title = html_lib.unescape(article.get("title", "") or "").strip()
    normalized_for_similarity = normalize_title_for_similarity(title)
    if normalized_for_similarity:
        return normalized_for_similarity

    fallback = unicodedata.normalize("NFKC", title)
    if " - " in fallback:
        head, tail = fallback.rsplit(" - ", 1)
        if len(head.split()) >= 4 and 1 <= len(tail.split()) <= 6:
            fallback = head
    fallback = re.sub(r"https?://\S+", " ", fallback)
    fallback = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in fallback
    )
    return re.sub(r"\s+", " ", fallback).strip()


def build_embedding_input_hash(article: dict[str, Any]) -> str:
    return hashlib.sha256(build_embedding_input_text(article).encode("utf-8")).hexdigest()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    if not vector_a or not vector_b or len(vector_a) != len(vector_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def encode_embedding_vector(vector: list[float]) -> str:
    coerced = coerce_embedding_vector(vector)
    if not coerced:
        raise ValueError("Embedding vector must contain finite numeric values")
    return json.dumps(coerced)


def decode_embedding_vector(value: Any) -> Optional[list[float]]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return None
    return coerce_embedding_vector(parsed)


def load_embeddings_for_article_ids(article_ids: list[int]) -> dict[int, list[float]]:
    if not article_ids:
        return {}
    placeholders = ",".join("?" for _ in article_ids)
    with db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT article_id, embedding_json
            FROM article_embeddings
            WHERE embedding_model = ?
              AND article_id IN ({placeholders})
            """,
            (get_embedding_model(), *article_ids),
        ).fetchall()
    embeddings: dict[int, list[float]] = {}
    for row in rows:
        vector = decode_embedding_vector(row["embedding_json"])
        if vector is None:
            continue
        embeddings[int(row["article_id"])] = vector
    return embeddings


def normalize_title_for_similarity(title: str) -> str:
    text = html_lib.unescape(title or "").strip().lower()
    if " - " in text:
        head, tail = text.rsplit(" - ", 1)
        if len(head.split()) >= 4 and 1 <= len(tail.split()) <= 6:
            text = head
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_title_signature(title: str) -> dict[str, Any]:
    normalized = normalize_title_for_similarity(title)
    raw_tokens = [token for token in normalized.split() if len(token) >= 2]
    filtered_tokens = [
        token
        for token in raw_tokens
        if len(token) >= 3 and token not in TITLE_SIMILARITY_STOPWORDS
    ]
    tokens = filtered_tokens or raw_tokens
    token_set = set(tokens)
    long_tokens = {token for token in token_set if len(token) >= 5 and not token.isdigit()}
    numbers = {token for token in token_set if any(char.isdigit() for char in token)}
    sequence_text = " ".join(tokens) if tokens else normalized
    return {
        "normalized": normalized,
        "tokens": token_set,
        "long_tokens": long_tokens,
        "numbers": numbers,
        "sequence_text": sequence_text,
    }


def title_similarity_metrics(title_a: str, title_b: str) -> dict[str, Any]:
    signature_a = build_title_signature(title_a)
    signature_b = build_title_signature(title_b)

    if not signature_a["normalized"] or not signature_b["normalized"]:
        return {"similar": False, "score": 0.0}

    if signature_a["normalized"] == signature_b["normalized"]:
        return {
            "similar": True,
            "score": 1.0,
            "overlap": 1.0,
            "jaccard": 1.0,
            "sequence": 1.0,
        }

    tokens_a = signature_a["tokens"]
    tokens_b = signature_b["tokens"]
    if not tokens_a or not tokens_b:
        return {"similar": False, "score": 0.0}

    intersection = tokens_a & tokens_b
    overlap = len(intersection) / max(1, min(len(tokens_a), len(tokens_b)))
    jaccard = len(intersection) / max(1, len(tokens_a | tokens_b))
    sequence = SequenceMatcher(None, signature_a["sequence_text"], signature_b["sequence_text"]).ratio()
    long_overlap = len(signature_a["long_tokens"] & signature_b["long_tokens"])
    numbers_conflict = bool(
        signature_a["numbers"] and signature_b["numbers"] and signature_a["numbers"] != signature_b["numbers"]
    )
    score = round((0.5 * overlap) + (0.2 * jaccard) + (0.3 * sequence), 4)

    similar = False
    if overlap >= 0.9 and len(intersection) >= 3:
        similar = True
    elif not numbers_conflict and overlap >= 0.68 and long_overlap >= 2 and (jaccard >= 0.40 or sequence >= 0.72):
        similar = True
    elif not numbers_conflict and overlap >= 0.58 and long_overlap >= 2 and sequence >= 0.80:
        similar = True
    elif not numbers_conflict and overlap >= 0.50 and long_overlap >= 2 and sequence >= 0.88:
        similar = True
    elif not numbers_conflict and overlap >= 0.46 and long_overlap >= 3 and sequence >= 0.72:
        similar = True
    elif not numbers_conflict and len(intersection) >= 4 and long_overlap >= 3 and (jaccard >= 0.30 or sequence >= 0.68):
        similar = True

    return {
        "similar": similar,
        "score": score,
        "overlap": round(overlap, 4),
        "jaccard": round(jaccard, 4),
        "sequence": round(sequence, 4),
    }


def article_similarity_metrics(article_a: dict[str, Any], article_b: dict[str, Any]) -> dict[str, Any]:
    title_metrics = title_similarity_metrics(article_a.get("title", ""), article_b.get("title", ""))
    vector_a = article_a.get("embedding_vector")
    vector_b = article_b.get("embedding_vector")
    embedding_similarity = None
    if vector_a and vector_b:
        embedding_similarity = round(float(cosine_similarity(vector_a, vector_b)), 4)

    similar = bool(title_metrics["similar"])
    score = float(title_metrics["score"])
    if embedding_similarity is not None:
        if embedding_similarity >= 0.93:
            similar = True
            score = max(score, embedding_similarity)
        elif embedding_similarity >= 0.89 and (
            title_metrics["overlap"] >= 0.42 or title_metrics["sequence"] >= 0.72
        ):
            similar = True
            score = max(score, embedding_similarity)
        elif embedding_similarity >= 0.86 and (
            title_metrics["overlap"] >= 0.50 or title_metrics["sequence"] >= 0.78
        ):
            similar = True
            score = max(score, embedding_similarity)

    return {
        **title_metrics,
        "embedding_similarity": embedding_similarity,
        "similar": similar,
        "score": round(score, 4),
    }


def articles_within_similarity_window(article_a: dict[str, Any], article_b: dict[str, Any]) -> bool:
    published_a = parse_datetime(article_a.get("published_at") or article_a.get("created_at"))
    published_b = parse_datetime(article_b.get("published_at") or article_b.get("created_at"))
    return abs((published_a - published_b).total_seconds()) <= (SIMILARITY_LOOKBACK_HOURS * 3600)


def best_similarity_to_cluster(
    item: dict[str, Any],
    cluster_members: list[dict[str, Any]],
) -> tuple[Optional[dict[str, Any]], float]:
    best_metrics: Optional[dict[str, Any]] = None
    best_score = 0.0
    for member in cluster_members:
        if not articles_within_similarity_window(item, member):
            continue
        metrics = article_similarity_metrics(item, member)
        if metrics["similar"] and metrics["score"] > best_score:
            best_metrics = metrics
            best_score = metrics["score"]
    return best_metrics, best_score


def cluster_items_by_similarity(rows: list[sqlite3.Row | dict[str, Any]]) -> list[dict[str, Any]]:
    items = [row if isinstance(row, dict) else row_to_dict(row) for row in rows]
    attach_embeddings_to_items(items)
    sorted_items = sorted(
        items,
        key=lambda item: (
            parse_datetime(item.get("published_at")),
            int(item.get("id", 0)),
        ),
        reverse=True,
    )

    clusters: list[dict[str, Any]] = []
    for item in sorted_items:
        best_cluster: Optional[dict[str, Any]] = None
        best_score = 0.0
        for cluster in clusters:
            _metrics, score = best_similarity_to_cluster(item, cluster["members"])
            if score > best_score:
                best_cluster = cluster
                best_score = score
        if best_cluster:
            best_cluster["members"].append(item)
        else:
            clusters.append({"canonical": item, "members": [item]})
    return clusters


def summarize_pending_similarity(rows: list[sqlite3.Row | dict[str, Any]]) -> dict[str, Any]:
    items = [row if isinstance(row, dict) else row_to_dict(row) for row in rows]
    clusters = cluster_items_by_similarity(items)

    visible_items: list[dict[str, Any]] = []
    for cluster in clusters:
        canonical = dict(cluster["canonical"])
        members = cluster["members"]
        canonical["similar_count"] = max(len(members) - 1, 0)
        canonical["similar_sources"] = sorted(
            {
                member.get("source_label", "")
                for member in members[1:]
                if member.get("source_label")
            }
        )
        visible_items.append(canonical)

    similar_group_count = sum(1 for cluster in clusters if len(cluster["members"]) > 1)
    similar_hidden_count = sum(max(len(cluster["members"]) - 1, 0) for cluster in clusters)
    return {
        "visible_items": visible_items,
        "pending_total": len(items),
        "visible_total": len(visible_items),
        "similar_group_count": similar_group_count,
        "similar_hidden_count": similar_hidden_count,
    }


def build_visible_pending_feed_items(rows: list[sqlite3.Row | dict[str, Any]]) -> list[dict[str, Any]]:
    return summarize_pending_similarity(rows)["visible_items"]


def build_feed_similarity_snapshot(rows: list[sqlite3.Row | dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_pending_similarity(rows)
    visible_items = summary["visible_items"]
    return {
        "pending_total": summary["pending_total"],
        "visible_total": summary["visible_total"],
        "similar_group_count": summary["similar_group_count"],
        "similar_hidden_count": summary["similar_hidden_count"],
        "visible_article_ids": [int(item["id"]) for item in visible_items],
        "similar_count_by_id": {
            str(int(item["id"])): int(item.get("similar_count", 0))
            for item in visible_items
            if item.get("similar_count", 0)
        },
        "updated_at": utc_now_iso(),
    }


def trigger_feed_similarity_refresh() -> None:
    with STATE.feed_similarity_lock:
        STATE.feed_similarity_dirty = True


def current_feed_similarity_snapshot() -> dict[str, Any]:
    with STATE.feed_similarity_lock:
        return dict(STATE.feed_similarity_snapshot)


def update_feed_similarity_snapshot(rows: Optional[list[sqlite3.Row | dict[str, Any]]] = None) -> dict[str, Any]:
    pending_rows = rows if rows is not None else fetch_pending_feed_articles()
    snapshot = build_feed_similarity_snapshot(pending_rows)
    with STATE.feed_similarity_lock:
        STATE.feed_similarity_snapshot = snapshot
        STATE.feed_similarity_dirty = False
        STATE.feed_similarity_building = False
    return snapshot


def _feed_similarity_build_loop() -> None:
    while not STATE.stop_event.is_set():
        try:
            update_feed_similarity_snapshot()
        except Exception as exc:  # pragma: no cover - runtime defensive
            LOGGER.warning("Feed similarity refresh failed: %s", exc)
        with STATE.feed_similarity_lock:
            if not STATE.feed_similarity_dirty:
                STATE.feed_similarity_building = False
                return
            STATE.feed_similarity_dirty = False


def archive_pending_duplicates_for_article(
    conn: sqlite3.Connection,
    article: dict[str, Any],
    triggering_decision: str,
) -> list[int]:
    cutoff = (utc_now() - timedelta(hours=SIMILARITY_LOOKBACK_HOURS)).isoformat()
    candidates = conn.execute(
        """
        SELECT *
        FROM articles
        WHERE feed_decision = 'pending'
          AND id != ?
          AND datetime(published_at) >= datetime(?)
        ORDER BY datetime(published_at) DESC, id DESC
        """,
        (article["id"], cutoff),
    ).fetchall()

    article_items = attach_embeddings_to_items([dict(article)])
    source_article = article_items[0]
    candidate_items = attach_embeddings_to_items([row_to_dict(candidate) for candidate in candidates])

    duplicate_ids: list[int] = []
    for candidate_dict in candidate_items:
        if not articles_within_similarity_window(article, candidate_dict):
            continue
        metrics = article_similarity_metrics(source_article, candidate_dict)
        if metrics["similar"] and metrics["score"] >= 0.76:
            duplicate_ids.append(int(candidate_dict["id"]))

    if not duplicate_ids:
        return []

    now = utc_now_iso()
    placeholders = ",".join("?" for _ in duplicate_ids)
    conn.execute(
        f"""
        UPDATE articles
        SET feed_decision = 'archived',
            feed_decision_at = ?,
            updated_at = ?
        WHERE id IN ({placeholders})
        """,
        (now, now, *duplicate_ids),
    )

    for duplicate_id in duplicate_ids:
        log_event(
            conn,
            duplicate_id,
            "feed_auto_deduplicated",
            {
                "canonical_article_id": article["id"],
                "triggering_decision": triggering_decision,
            },
        )

    return duplicate_ids


def auto_archive_pending_duplicates_of_handled_articles(conn: sqlite3.Connection) -> int:
    cutoff = (utc_now() - timedelta(hours=SIMILARITY_LOOKBACK_HOURS)).isoformat()
    handled_rows = conn.execute(
        """
        SELECT *
        FROM articles
        WHERE feed_decision IN ('skip', 'summarize')
          AND feed_decision_at IS NOT NULL
          AND datetime(feed_decision_at) >= datetime(?)
        ORDER BY datetime(feed_decision_at) DESC, id DESC
        """,
        (cutoff,),
    ).fetchall()
    pending_rows = conn.execute(
        """
        SELECT *
        FROM articles
        WHERE feed_decision = 'pending'
          AND datetime(published_at) >= datetime(?)
        ORDER BY datetime(published_at) DESC, id DESC
        """,
        (cutoff,),
    ).fetchall()

    handled_articles = [row_to_dict(row) for row in handled_rows]
    pending_articles = attach_embeddings_to_items([row_to_dict(row) for row in pending_rows])
    archive_map: dict[int, int] = {}
    handled_clusters = cluster_items_by_similarity(handled_articles)

    for pending_article in pending_articles:
        for cluster in handled_clusters:
            metrics, score = best_similarity_to_cluster(pending_article, cluster["members"])
            if metrics and score >= 0.76:
                archive_map[int(pending_article["id"])] = int(cluster["canonical"]["id"])
                break

    if not archive_map:
        return 0

    now = utc_now_iso()
    archive_ids = list(archive_map.keys())
    placeholders = ",".join("?" for _ in archive_ids)
    conn.execute(
        f"""
        UPDATE articles
        SET feed_decision = 'archived',
            feed_decision_at = ?,
            updated_at = ?
        WHERE id IN ({placeholders})
        """,
        (now, now, *archive_ids),
    )

    for article_id, canonical_article_id in archive_map.items():
        log_event(
            conn,
            article_id,
            "feed_auto_deduplicated_to_handled_story",
            {"canonical_article_id": canonical_article_id},
        )

    return len(archive_ids)


def deduplicate_current_pending_feed() -> dict[str, Any]:
    now = utc_now_iso()
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE feed_decision = 'pending'
            ORDER BY datetime(published_at) DESC, id DESC
            """
        ).fetchall()
        items = [row_to_dict(row) for row in rows]
        clusters = cluster_items_by_similarity(items)

        before_pending = len(items)
        before_visible = len(clusters)
        archive_map: dict[int, int] = {}
        for cluster in clusters:
            canonical_id = int(cluster["canonical"]["id"])
            for member in cluster["members"][1:]:
                archive_map[int(member["id"])] = canonical_id

        if not archive_map:
            trigger_feed_similarity_refresh()
            ensure_feed_similarity_snapshot_async()
            return {
                "status": "ok",
                "archived_count": 0,
                "cluster_count": len(clusters),
                "before_pending": before_pending,
                "after_pending": before_pending,
                "before_visible": before_visible,
                "after_visible": before_visible,
                "processed_at": now,
            }

        archive_ids = list(archive_map.keys())
        placeholders = ",".join("?" for _ in archive_ids)
        conn.execute(
            f"""
            UPDATE articles
            SET feed_decision = 'archived',
                feed_decision_at = ?,
                updated_at = ?
            WHERE id IN ({placeholders})
            """,
            (now, now, *archive_ids),
        )
        for article_id, canonical_article_id in archive_map.items():
            log_event(
                conn,
                article_id,
                "feed_manual_deduplicated",
                {"canonical_article_id": canonical_article_id},
            )

    trigger_feed_similarity_refresh()
    ensure_feed_similarity_snapshot_async()
    return {
        "status": "ok",
        "archived_count": len(archive_map),
        "cluster_count": len(clusters),
        "before_pending": before_pending,
        "after_pending": before_pending - len(archive_map),
        "before_visible": before_visible,
        "after_visible": before_visible,
        "processed_at": now,
    }


def fetch_visible_pending_feed_articles_from_snapshot() -> list[dict[str, Any]]:
    snapshot = current_feed_similarity_snapshot()
    visible_ids = [int(article_id) for article_id in snapshot.get("visible_article_ids", [])]
    similar_count_by_id = snapshot.get("similar_count_by_id", {}) or {}
    if not visible_ids:
        return []

    placeholders = ",".join("?" for _ in visible_ids)
    with db_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM articles
            WHERE feed_decision = 'pending'
              AND id IN ({placeholders})
            ORDER BY datetime(published_at) DESC, id DESC
            """,
            tuple(visible_ids),
        ).fetchall()

    items = [row_to_dict(row) for row in rows]
    for item in items:
        item["similar_count"] = int(similar_count_by_id.get(str(int(item["id"])), 0))
    return items


def ensure_feed_similarity_snapshot_async() -> None:
    with STATE.feed_similarity_lock:
        needs_build = STATE.feed_similarity_dirty or not STATE.feed_similarity_snapshot
        if not needs_build or STATE.feed_similarity_building:
            return
        STATE.feed_similarity_building = True
        STATE.feed_similarity_dirty = False
    threading.Thread(
        target=_feed_similarity_build_loop,
        name="feed-similarity-build",
        daemon=True,
    ).start()


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


def encode_app_state_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def decode_app_state_value(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def decode_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def decode_json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value is None:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def upsert_app_state(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        """
        INSERT INTO app_state(key, value, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value,
            updated_at = excluded.updated_at
        """,
        (key, encode_app_state_value(value), utc_now_iso()),
    )


def fetch_app_state(conn: sqlite3.Connection, key: str, default: Any = None) -> Any:
    row = conn.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    return decode_app_state_value(row["value"], default)


def update_app_state(key: str, value: Any) -> None:
    with db_connection() as conn:
        upsert_app_state(conn, key, value)


def get_app_state(key: str, default: Any = None) -> Any:
    with db_connection() as conn:
        return fetch_app_state(conn, key, default)


def get_training_status() -> dict[str, Any]:
    with STATE.training_status_lock:
        return dict(STATE.training_status)


def set_training_status(**updates: Any) -> None:
    with STATE.training_status_lock:
        STATE.training_status.update(updates)


def wake_summary_worker() -> None:
    STATE.summary_event.set()


def encode_event_payload(payload: Optional[dict[str, Any]] = None) -> str:
    return json.dumps(payload or {}, ensure_ascii=True, sort_keys=True)


def decode_event_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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
            encode_event_payload(payload),
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
        LOGGER.info("Google News URL decoder returned no decoded URL for %s: %s", url, decoded)
    except Exception as exc:  # pragma: no cover - defensive runtime handling
        LOGGER.warning("Could not decode Google News URL %s: %s", url, exc)
    return url


def is_probable_homepage_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    path = (parsed.path or "").strip("/")
    return bool(parsed.scheme and parsed.netloc and not path)


def article_fetch_url_candidates(article: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for key in ("link_to_article", "rss_source_url", "source_url"):
        url = (article.get(key) or "").strip()
        if not url:
            continue
        if key == "source_url" and is_probable_homepage_url(url):
            continue
        decoded_url = decode_google_news_url(url)
        for candidate in (decoded_url, url):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    return candidates


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


def fetch_best_article_text(article: dict[str, Any]) -> tuple[str, str]:
    candidates = article_fetch_url_candidates(article)
    if not candidates:
        raise RuntimeError("No article URL available for extraction")

    best_text = ""
    best_url = candidates[0]
    fetched_any_candidate = False
    errors: list[str] = []
    for url in candidates:
        try:
            text, final_url = fetch_and_extract_article_text(url)
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            continue
        fetched_any_candidate = True
        if len(text.strip()) >= MIN_EXTRACTED_ARTICLE_CHARS:
            return text, final_url
        if len(text.strip()) > len(best_text.strip()):
            best_text = text
            best_url = final_url

    if best_text.strip() or fetched_any_candidate:
        return best_text, best_url
    raise RuntimeError("Article extraction failed for all URL candidates: " + "; ".join(errors))


def build_summary_fallback_text(job: dict[str, Any], extracted_text: str, final_url: str) -> str:
    title = (job.get("title") or "").strip()
    source_label = (job.get("source_label") or "").strip()
    source_url = (job.get("source_url") or "").strip()
    published_at = (job.get("published_at") or "").strip()
    short_excerpt = (extracted_text or "").strip()
    if len(short_excerpt) > 1200:
        short_excerpt = short_excerpt[:1200].rsplit(" ", 1)[0].strip()

    parts = [
        "Der Volltext konnte nicht zuverlässig extrahiert werden.",
        "Fasse deshalb ausschließlich die folgenden Metadaten und vorhandenen Textauszüge zusammen.",
        "Erfinde keine zusätzlichen Details, Zahlen, Zitate oder Hintergründe.",
        f"Titel: {title}" if title else "",
        f"Quelle: {source_label}" if source_label else "",
        f"Quell-URL: {source_url}" if source_url else "",
        f"Finale URL: {final_url}" if final_url else "",
        f"Veröffentlicht: {published_at}" if published_at else "",
        f"Vorhandener Textauszug: {short_excerpt}" if short_excerpt else "",
    ]
    return "\n".join(part for part in parts if part).strip()


def summary_source_text(job: dict[str, Any], extracted_text: str, final_url: str) -> tuple[str, bool]:
    cleaned = (extracted_text or "").strip()
    if len(cleaned) >= MIN_EXTRACTED_ARTICLE_CHARS:
        return cleaned, False
    return build_summary_fallback_text(job, cleaned, final_url), True


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


def get_compare_enabled() -> bool:
    stored = get_app_state("llm_compare_enabled", None)
    if stored is None:
        return bool(SETTINGS["llm_compare"]["enabled"])
    return bool(stored)


def get_compare_models() -> list[str]:
    models: list[str] = []
    for model in [OLLAMA_MODEL, *COMPARE_MODELS]:
        if model not in models:
            models.append(model)
    return models


def get_compare_timeout_seconds() -> int:
    compare_settings = SETTINGS.get("llm_compare", {})
    timeout = compare_settings.get("request_timeout_seconds")
    if timeout is None:
        timeout = SETTINGS.get("ollama", {}).get("summary_timeout_seconds", 300)
    try:
        return max(30, int(timeout))
    except (TypeError, ValueError):
        return 300


def get_summary_timeout_seconds() -> int:
    timeout = SETTINGS.get("ollama", {}).get("summary_timeout_seconds", 300)
    try:
        return max(30, int(timeout))
    except (TypeError, ValueError):
        return 300


def get_embedding_model() -> str:
    return str(SETTINGS.get("ollama", {}).get("embedding_model", "nomic-embed-text-v2-moe:latest")).strip()


def get_embedding_timeout_seconds() -> int:
    timeout = SETTINGS.get("ollama", {}).get("embedding_timeout_seconds", 120)
    try:
        return max(15, int(timeout))
    except (TypeError, ValueError):
        return 120


def get_request_timeout_seconds() -> int:
    timeout = SETTINGS.get("timing", {}).get("request_timeout_seconds", 20)
    try:
        return max(5, int(timeout))
    except (TypeError, ValueError):
        return 20


def attach_embeddings_to_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not items:
        return items
    embedding_map = load_embeddings_for_article_ids(
        [int(item["id"]) for item in items if item.get("id") is not None]
    )
    for item in items:
        item["embedding_vector"] = embedding_map.get(int(item["id"])) if item.get("id") is not None else None
    return items


def coerce_embedding_vector(value: Any) -> Optional[list[float]]:
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in value):
        return [float(item) for item in value]
    if isinstance(value[0], list):
        return coerce_embedding_vector(value[0])
    return None


def extract_ollama_embedding_vector(body: dict[str, Any]) -> Optional[list[float]]:
    vector = coerce_embedding_vector(body.get("embedding"))
    if vector:
        return vector

    vector = coerce_embedding_vector(body.get("embeddings"))
    if vector:
        return vector

    data = body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return coerce_embedding_vector(data[0].get("embedding"))

    return None


def normalize_embedding_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_ollama_embedding_request_specs(text: str, model_name: Optional[str] = None) -> list[dict[str, Any]]:
    normalized_text = normalize_embedding_text(text)
    if not normalized_text:
        raise ValueError("Embedding input is empty after normalization")

    model = (model_name or get_embedding_model()).strip()
    return [
        {
            "path": "/api/embed",
            "json": {
                "model": model,
                "input": normalized_text,
            },
        },
        {
            "path": "/api/embeddings",
            "json": {
                "model": model,
                "prompt": normalized_text,
            },
        },
    ]


def ollama_embed_text(text: str) -> list[float]:
    request_specs = build_ollama_embedding_request_specs(text)
    with STATE.ollama_lock:
        response = None
        for index, spec in enumerate(request_specs):
            response = requests.post(
                f"{OLLAMA_BASE_URL}{spec['path']}",
                timeout=get_embedding_timeout_seconds(),
                json=spec["json"],
            )
            if response.status_code != 404 or index == len(request_specs) - 1:
                break
    if response is None:  # pragma: no cover - request_specs is never empty
        raise RuntimeError("No Ollama embedding request was prepared")
    response.raise_for_status()
    body = response.json()
    vector = extract_ollama_embedding_vector(body)
    if vector:
        return vector
    response_keys = ", ".join(sorted(str(key) for key in body.keys())) or "none"
    raise RuntimeError(f"Ollama embedding response did not contain a usable vector; keys={response_keys}")


def select_embedding_candidate(rows: list[sqlite3.Row | dict[str, Any]], model_name: str) -> Optional[dict[str, Any]]:
    for row in rows:
        item = row_to_dict(row) if isinstance(row, sqlite3.Row) else dict(row)
        if not build_embedding_input_text(item):
            continue
        expected_hash = build_embedding_input_hash(item)
        if item.get("embedding_model") != model_name or item.get("embedding_input_hash") != expected_hash:
            item["expected_embedding_hash"] = expected_hash
            return item
    return None


def select_article_for_embedding() -> Optional[dict[str, Any]]:
    cutoff = (utc_now() - timedelta(hours=72)).isoformat()
    embedding_model = get_embedding_model()
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                a.id,
                a.title,
                a.published_at,
                a.created_at,
                a.updated_at,
                ae.embedding_input_hash,
                ae.embedding_model
            FROM articles a
            LEFT JOIN article_embeddings ae
              ON ae.article_id = a.id
             AND ae.embedding_model = ?
            WHERE a.feed_decision IN ('pending', 'skip', 'summarize')
              AND LENGTH(TRIM(COALESCE(a.title, ''))) > 0
              AND datetime(a.published_at) >= datetime(?)
            ORDER BY
              CASE a.feed_decision
                WHEN 'pending' THEN 0
                WHEN 'summarize' THEN 1
                ELSE 2
              END,
              datetime(a.published_at) DESC,
              a.id DESC
            LIMIT 250
            """,
            (embedding_model, cutoff),
        ).fetchall()

    return select_embedding_candidate(rows, embedding_model)


def store_article_embedding(article: dict[str, Any], vector: list[float]) -> None:
    now = utc_now_iso()
    with db_connection() as conn:
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
            (
                int(article["id"]),
                get_embedding_model(),
                article["expected_embedding_hash"],
                encode_embedding_vector(vector),
                now,
                now,
            ),
        )


def summary_work_pending() -> bool:
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM articles
            WHERE summary_status IN ('queued', 'processing')
            LIMIT 1
            """
        ).fetchone()
    return row is not None


def run_embedding_worker_once() -> str:
    if summary_work_pending():
        STATE.stop_event.wait(max(1, min(5, EMBEDDING_POLL_SECONDS)))
        return "summary_pending"

    job = select_article_for_embedding()
    if not job:
        STATE.stop_event.wait(EMBEDDING_POLL_SECONDS)
        return "idle"

    try:
        vector = ollama_embed_text(build_embedding_input_text(job))
        store_article_embedding(job, vector)
        trigger_feed_similarity_refresh()
        ensure_feed_similarity_snapshot_async()
        return "embedded"
    except Exception as exc:  # pragma: no cover - runtime defensive
        LOGGER.warning("Embedding generation failed for article %s: %s", job.get("id"), exc)
        STATE.stop_event.wait(max(5, EMBEDDING_POLL_SECONDS))
        return "failed"


def embedding_worker() -> None:
    while not STATE.stop_event.is_set():
        run_embedding_worker_once()


def current_compare_session() -> Optional[dict[str, Any]]:
    session_id = get_app_state("llm_compare_session_id", None)
    if not session_id:
        return None
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM llm_compare_sessions
            WHERE id = ? AND status = 'active'
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    return row_to_dict(row) if row else None


def latest_compare_session() -> Optional[dict[str, Any]]:
    session = current_compare_session()
    if session:
        return session
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM llm_compare_sessions
            ORDER BY datetime(enabled_at) DESC, id DESC
            LIMIT 1
            """
        ).fetchone()
    return row_to_dict(row) if row else None


def clear_compare_progress() -> None:
    update_app_state("llm_compare_progress", None)


def wake_compare_worker() -> None:
    STATE.compare_event.set()


def append_compare_session_header(export_path: Path, enabled_at: str, models: list[str]) -> None:
    header = [
        "# LLM Compare Session",
        "",
        f'<compare_session enabled_at="{html_lib.escape(enabled_at)}" primary_model="{html_lib.escape(OLLAMA_MODEL)}">',
        "<models>",
    ]
    for model in models:
        header.append(f"  <model>{html_lib.escape(model)}</model>")
    header.extend(["</models>", ""])
    export_path.write_text("\n".join(header), encoding="utf-8")


def set_compare_enabled(enabled: bool) -> dict[str, Any]:
    existing_session = current_compare_session()
    now = utc_now_iso()

    if enabled:
        if existing_session:
            update_app_state("llm_compare_enabled", True)
            return {
                "enabled": True,
                "session": existing_session,
            }

        models = get_compare_models()
        export_name = f"llm_compare_{utc_now().strftime('%Y%m%d_%H%M%S')}.md"
        export_path = COMPARE_EXPORT_DIR / export_name
        append_compare_session_header(export_path, now, models)

        with db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO llm_compare_sessions(enabled_at, export_path, primary_model, models_json, status)
                VALUES (?, ?, ?, ?, 'active')
                """,
                (
                    now,
                    str(export_path),
                    OLLAMA_MODEL,
                    json.dumps(models),
                ),
            )
            session_id = int(cursor.lastrowid)

        update_app_state("llm_compare_enabled", True)
        update_app_state("llm_compare_session_id", session_id)
        clear_compare_progress()
        wake_compare_worker()
        session = current_compare_session()
        return {"enabled": True, "session": session}

    if existing_session:
        export_path = Path(existing_session["export_path"])
        if export_path.exists():
            with open(export_path, "a", encoding="utf-8") as handle:
                handle.write(f"\n</compare_session>\n")
        with db_connection() as conn:
            conn.execute(
                """
                UPDATE llm_compare_sessions
                SET disabled_at = ?, status = 'closed'
                WHERE id = ?
                """,
                (now, existing_session["id"]),
            )

    update_app_state("llm_compare_enabled", False)
    update_app_state("llm_compare_session_id", None)
    clear_compare_progress()
    wake_compare_worker()
    return {"enabled": False, "session": None}


def bootstrap_compare_mode() -> None:
    session = current_compare_session()
    if get_compare_enabled() and session is None:
        set_compare_enabled(True)
    elif get_compare_enabled() and session is not None:
        wake_compare_worker()


def rewrite_compare_export_article(
    session: dict[str, Any],
    article: dict[str, Any],
    final_url: str,
) -> None:
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT model_name, status, summary_title, summary_text, error_text
            FROM llm_compare_results
            WHERE session_id = ? AND article_id = ?
            """,
            (session["id"], article["id"]),
        ).fetchall()

    summaries_by_model = {row["model_name"]: row_to_dict(row) for row in rows}
    ordered_summaries: list[dict[str, Any]] = []
    for model_name in decode_json_array(session["models_json"]):
        if model_name in summaries_by_model:
            ordered_summaries.append(summaries_by_model[model_name])

    export_path = Path(session["export_path"])
    lines = [
        f'<article_compare article_id="{article["id"]}" created_at="{html_lib.escape(utc_now_iso())}">',
        f"  <article_title>{html_lib.escape(article.get('title', ''))}</article_title>",
        f"  <source>{html_lib.escape(article.get('source_label', ''))}</source>",
        f"  <url>{html_lib.escape(final_url)}</url>",
    ]
    for summary in ordered_summaries:
        status = summary.get("status", "ok")
        lines.append(f'  <model_summary model="{html_lib.escape(summary["model_name"])}" status="{html_lib.escape(status)}">')
        lines.append(f"    <summary_title>{html_lib.escape(summary.get('summary_title', ''))}</summary_title>")
        lines.append(f"    <summary_text>{html_lib.escape(summary.get('summary_text', ''))}</summary_text>")
        if summary.get("error_text"):
            lines.append(f"    <error>{html_lib.escape(summary.get('error_text', ''))}</error>")
        lines.append("  </model_summary>")
    lines.extend(["</article_compare>", ""])
    article_block = "\n".join(lines)
    content = export_path.read_text(encoding="utf-8") if export_path.exists() else ""
    pattern = re.compile(
        rf'<article_compare article_id="{article["id"]}"[^>]*>.*?</article_compare>\n*',
        re.DOTALL,
    )
    content = pattern.sub("", content).rstrip()
    if content and not content.endswith("\n"):
        content += "\n"
    content = f"{content}\n{article_block}\n".lstrip()
    export_path.write_text(content, encoding="utf-8")


def store_compare_result(
    conn: sqlite3.Connection,
    session_id: int,
    article_id: int,
    model_name: str,
    status: str,
    duration_ms: int,
    summary_title: str,
    summary_text: str,
    error_text: str = "",
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO llm_compare_results(
            session_id, article_id, model_name, status, duration_ms, summary_title, summary_text, error_text, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            article_id,
            model_name,
            status,
            duration_ms,
            summary_title,
            summary_text,
            error_text,
            utc_now_iso(),
        ),
    )


def run_compare_summaries(
    article: dict[str, Any],
    article_text: str,
    final_url: str,
    primary_title: str,
    primary_summary: str,
) -> None:
    if not get_compare_enabled():
        return

    session = current_compare_session()
    if not session:
        return

    models = decode_json_array(session["models_json"])
    article_id = int(article["id"])
    with db_connection() as conn:
        existing_rows = conn.execute(
            """
            SELECT model_name, status
            FROM llm_compare_results
            WHERE session_id = ? AND article_id = ?
            """,
            (session["id"], article_id),
        ).fetchall()
        existing_models = {row["model_name"] for row in existing_rows}
        existing_failed_models = sum(1 for row in existing_rows if row["status"] == "failed")

    completed_models = len(existing_models)
    failed_models = existing_failed_models
    total_models = len(models)
    started_at = utc_now_iso()

    update_app_state(
        "llm_compare_progress",
        {
            "session_id": int(session["id"]),
            "article_id": article_id,
            "article_title": article.get("title", ""),
            "completed_models": completed_models,
            "failed_models": failed_models,
            "total_models": total_models,
            "current_model": None,
            "status": "running",
            "started_at": started_at,
            "updated_at": utc_now_iso(),
        },
    )

    for model_name in models:
        if model_name in existing_models:
            continue

        update_app_state(
            "llm_compare_progress",
            {
                "session_id": int(session["id"]),
                "article_id": article_id,
                "article_title": article.get("title", ""),
                "completed_models": completed_models,
                "failed_models": failed_models,
                "total_models": total_models,
                "current_model": model_name,
                "status": "running",
                "started_at": started_at,
                "updated_at": utc_now_iso(),
            },
        )

        if model_name == OLLAMA_MODEL:
            model_started = time.perf_counter()
            compare_title, compare_text = primary_title, primary_summary
            duration_ms = int((time.perf_counter() - model_started) * 1000)
        else:
            model_started = time.perf_counter()
            try:
                compare_title, compare_text = ollama_generate_summary(
                    article["title"],
                    article_text,
                    model_name=model_name,
                    timeout_seconds=get_compare_timeout_seconds(),
                )
                duration_ms = int((time.perf_counter() - model_started) * 1000)
            except Exception as exc:
                duration_ms = int((time.perf_counter() - model_started) * 1000)
                failed_models += 1
                with db_connection() as conn:
                    store_compare_result(
                        conn,
                        int(session["id"]),
                        article_id,
                        model_name,
                        "failed",
                        duration_ms,
                        "",
                        "",
                        str(exc),
                    )
                    log_event(
                        conn,
                        article_id,
                        "llm_compare_model_failed",
                        {
                            "session_id": session["id"],
                            "model_name": model_name,
                            "error": str(exc),
                        },
                    )
                completed_models += 1
                update_app_state(
                    "llm_compare_progress",
                    {
                        "session_id": int(session["id"]),
                        "article_id": article_id,
                        "article_title": article.get("title", ""),
                        "completed_models": completed_models,
                        "failed_models": failed_models,
                        "total_models": total_models,
                        "current_model": None,
                        "status": "running" if completed_models < total_models else "completed",
                        "started_at": started_at,
                        "updated_at": utc_now_iso(),
                    },
                )
                continue

        with db_connection() as conn:
            store_compare_result(
                conn,
                int(session["id"]),
                article_id,
                model_name,
                "ok",
                duration_ms,
                compare_title,
                compare_text,
            )
            log_event(
                conn,
                article_id,
                "llm_compare_summary_generated",
                {
                    "session_id": session["id"],
                    "model_name": model_name,
                },
            )

        completed_models += 1
        update_app_state(
            "llm_compare_progress",
            {
                "session_id": int(session["id"]),
                "article_id": article_id,
                "article_title": article.get("title", ""),
                "completed_models": completed_models,
                "failed_models": failed_models,
                "total_models": total_models,
                "current_model": None,
                "status": "running" if completed_models < total_models else "completed",
                "started_at": started_at,
                "updated_at": utc_now_iso(),
            },
        )

    if completed_models:
        rewrite_compare_export_article(session, article, final_url)

    clear_compare_progress()


def build_summary_prompt(article_title: str, article_text: str) -> str:
    return f"""
Du bist ein präziser Nachrichtenassistent.
Erzeuge einen deutschen Titel und eine deutsche Zusammenfassung.

Regeln:
- sachlich, knapp, klar
- keine Einleitung wie "Der Artikel beschreibt"
- kein Verweis auf Zeitung, Portal oder Quelle
- der Titel soll wie eine kurze News-Headline klingen
- die Zusammenfassung soll 3 bis 5 Sätze lang sein
- wenn der Artikeltext nur Metadaten oder kurze Auszüge enthält: nur diese Fakten zusammenfassen und nichts ergänzen
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


def build_ollama_summary_payload(
    article_title: str,
    article_text: str,
    model_name: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "model": model_name or OLLAMA_MODEL,
        "prompt": build_summary_prompt(article_title, article_text),
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }


def ollama_generate_summary(
    article_title: str,
    article_text: str,
    model_name: Optional[str] = None,
    timeout_seconds: Optional[int] = None,
) -> tuple[str, str]:
    payload = build_ollama_summary_payload(article_title, article_text, model_name=model_name)

    # Only one Ollama generation may run at a time. Large local models will
    # otherwise compete for RAM/VRAM and trigger slowdowns or timeouts.
    with STATE.ollama_lock:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            timeout=timeout_seconds or get_summary_timeout_seconds(),
            json=payload,
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


def latest_model_run(target: str, include_rejected: bool = True) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        where_sql = "WHERE target = ?" if include_rejected else "WHERE target = ? AND status != 'rejected'"
        row = conn.execute(
            f"""
            SELECT *
            FROM model_runs
            {where_sql}
            ORDER BY datetime(trained_at) DESC, id DESC
            LIMIT 1
            """,
            (target,),
        ).fetchone()
    return row_to_dict(row) if row else None


def previous_model_run(target: str, include_rejected: bool = True) -> Optional[dict[str, Any]]:
    with db_connection() as conn:
        where_sql = "WHERE target = ?" if include_rejected else "WHERE target = ? AND status != 'rejected'"
        row = conn.execute(
            f"""
            SELECT *
            FROM model_runs
            {where_sql}
            ORDER BY datetime(trained_at) DESC, id DESC
            LIMIT 1 OFFSET 1
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


def build_unavailable_feed_prediction() -> dict[str, Any]:
    return {
        "available": False,
        "recommended": None,
        "maybe": None,
        "tier": None,
        "probability": None,
        "run_id": None,
    }


def build_feed_prediction(recommended: bool, maybe: bool, probability: float, run_id: int) -> dict[str, Any]:
    return {
        "available": True,
        "recommended": bool(recommended),
        "maybe": bool(maybe),
        "tier": "recommended" if recommended else ("maybe" if maybe else "no"),
        "probability": round(float(probability), 4),
        "run_id": run_id,
    }


def build_threshold_feed_prediction(
    probability: float,
    threshold: float,
    maybe_threshold: float,
    run_id: int,
) -> dict[str, Any]:
    return build_feed_prediction(
        probability >= threshold,
        probability >= maybe_threshold,
        probability,
        run_id,
    )


def build_cached_feed_prediction(item: dict[str, Any], run_id: int, maybe_threshold: Optional[float] = None) -> Optional[dict[str, Any]]:
    probability = item.get("predicted_probability")
    recommendation = item.get("predicted_recommendation")
    if item.get("prediction_model_run_id") != run_id:
        return None
    if probability is None or recommendation is None:
        return None
    maybe = False
    if maybe_threshold is not None:
        try:
            maybe = float(probability) >= float(maybe_threshold)
        except (TypeError, ValueError):
            maybe = bool(recommendation)
    return build_feed_prediction(bool(recommendation), bool(maybe), float(probability), run_id)


def predict_feed_rows(rows: list[sqlite3.Row | dict[str, Any]]) -> tuple[list[dict[str, Any]], Optional[int]]:
    if not rows:
        return [], None

    artifact = get_loaded_model("feed_recommendation")
    if not artifact:
        items = []
        for row in rows:
            item = row if isinstance(row, dict) else row_to_dict(row)
            item["prediction"] = build_unavailable_feed_prediction()
            items.append(item)
        return items, None

    pipeline: Pipeline = artifact["pipeline"]
    items = [row if isinstance(row, dict) else row_to_dict(row) for row in rows]
    stale_items: list[dict[str, Any]] = []
    for item in items:
        cached_prediction = build_cached_feed_prediction(item, artifact["run_id"], artifact.get("maybe_threshold"))
        if cached_prediction is not None:
            item["prediction"] = cached_prediction
        else:
            stale_items.append(item)

    if stale_items:
        features = [build_feature_text("feed_recommendation", row) for row in stale_items]
        probabilities = pipeline.predict_proba(features)[:, 1]
        for item, probability in zip(stale_items, probabilities):
            item["prediction"] = build_threshold_feed_prediction(
                float(probability),
                float(artifact["threshold"]),
                float(artifact.get("maybe_threshold", artifact["threshold"])),
                int(artifact["run_id"]),
            )

    return items, artifact["run_id"]


def compute_feed_prediction_snapshot(row: sqlite3.Row | dict[str, Any]) -> Optional[dict[str, Any]]:
    artifact = get_loaded_model("feed_recommendation")
    if not artifact:
        return None

    item = row if isinstance(row, dict) else row_to_dict(row)
    cached_prediction = build_cached_feed_prediction(item, artifact["run_id"], artifact.get("maybe_threshold"))
    if cached_prediction is None:
        pipeline: Pipeline = artifact["pipeline"]
        feature = build_feature_text("feed_recommendation", item)
        probability = float(pipeline.predict_proba([feature])[0][1])
        cached_prediction = build_threshold_feed_prediction(
            probability,
            float(artifact["threshold"]),
            float(artifact.get("maybe_threshold", artifact["threshold"])),
            int(artifact["run_id"]),
        )

    return {
        "prediction_model_run_id": cached_prediction["run_id"],
        "predicted_probability": cached_prediction["probability"],
        "predicted_recommendation": bool(cached_prediction["recommended"]),
        "predicted_maybe": bool(cached_prediction.get("maybe")),
        "predicted_tier": cached_prediction.get("tier"),
        "threshold": round(float(artifact["threshold"]), 4),
        "maybe_threshold": round(float(artifact.get("maybe_threshold", artifact["threshold"])), 4),
    }


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


def feed_row_is_recommended(row: dict[str, Any]) -> bool:
    return bool(row.get("prediction", {}).get("recommended"))


def feed_row_is_maybe(row: dict[str, Any]) -> bool:
    return row.get("prediction", {}).get("tier") == "maybe"


def filter_predicted_feed_rows(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode == "recommended":
        return [row for row in rows if feed_row_is_recommended(row)]
    if mode == "maybe":
        return [row for row in rows if feed_row_is_maybe(row)]
    if mode == "maybe_plus":
        return [row for row in rows if feed_row_is_recommended(row) or feed_row_is_maybe(row)]
    return list(rows)


def build_feed_response_counts(rows: list[dict[str, Any]], similarity: dict[str, int]) -> dict[str, int]:
    return {
        "total_pending": len(rows),
        "recommended_pending": sum(1 for row in rows if feed_row_is_recommended(row)),
        "maybe_pending": sum(1 for row in rows if feed_row_is_maybe(row)),
        "similar_group_count": similarity["similar_group_count"],
        "similar_hidden_count": similarity["similar_hidden_count"],
    }


def build_feed_similarity_counts(snapshot: dict[str, Any], visible_count: int) -> dict[str, int]:
    return {
        "pending_total": int(snapshot.get("pending_total", 0)),
        "visible_total": int(snapshot.get("visible_total", visible_count)),
        "similar_group_count": int(snapshot.get("similar_group_count", 0)),
        "similar_hidden_count": int(snapshot.get("similar_hidden_count", 0)),
    }


def load_feed_rows_for_api() -> tuple[list[dict[str, Any]], dict[str, int]]:
    snapshot = current_feed_similarity_snapshot()
    rows = fetch_visible_pending_feed_articles_from_snapshot()
    similarity = build_feed_similarity_counts(snapshot, len(rows))

    if rows:
        return rows, similarity

    pending_rows = fetch_pending_feed_articles()
    fallback_rows = [row_to_dict(row) for row in pending_rows]
    ensure_feed_similarity_snapshot_async()
    return fallback_rows, {
        "pending_total": len(fallback_rows),
        "visible_total": len(fallback_rows),
        "similar_group_count": 0,
        "similar_hidden_count": 0,
    }


def build_feed_api_payload(mode: str) -> dict[str, Any]:
    rows, similarity = load_feed_rows_for_api()
    predicted_rows, run_id = predict_feed_rows(rows)
    update_cached_feed_predictions(predicted_rows, run_id)
    filtered = filter_predicted_feed_rows(predicted_rows, mode)
    return {
        "mode": mode,
        "items": [serialize_article_for_feed(item) for item in filtered],
        "counts": build_feed_response_counts(predicted_rows, similarity),
    }


def refresh_feeds() -> dict[str, Any]:
    if not STATE.refresh_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Feed refresh already running"}

    inserted = 0
    updated = 0
    feed_errors: list[str] = []
    started_at = utc_now_iso()
    try:
        LOGGER.info("Starting RSS refresh")
        parsed_feeds: list[tuple[str, Any]] = []
        for feed_url in RSS_FEED_URLS:
            try:
                feed = feedparser.parse(feed_url)
                parsed_feeds.append((feed_url, feed))
            except Exception as exc:
                feed_errors.append(f"{feed_url}: {exc}")
                LOGGER.warning("Could not parse feed %s: %s", feed_url, exc)
                continue

        with db_connection() as conn:
            for feed_url, feed in parsed_feeds:
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

            deduplicated = auto_archive_pending_duplicates_of_handled_articles(conn)

        result = {
            "status": "ok",
            "inserted": inserted,
            "updated": updated,
            "deduplicated": deduplicated,
            "errors": feed_errors,
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        }
        update_app_state("last_feed_refresh", result)
        trigger_feed_similarity_refresh()
        ensure_feed_similarity_snapshot_async()
        LOGGER.info("RSS refresh finished: %s inserted, %s updated", inserted, updated)
        return result
    finally:
        STATE.refresh_lock.release()


def set_feed_decision(article_id: int, decision: str) -> dict[str, Any]:
    if decision not in {"skip", "summarize"}:
        raise ValueError("Unsupported feed decision")
    with db_connection() as conn:
        row_dict = fetch_article_by_id(conn, article_id)
        if not row_dict:
            raise LookupError("Article not found")

        prediction_snapshot = compute_feed_prediction_snapshot(row_dict)
        now = utc_now_iso()
        transition = build_feed_decision_transition(row_dict, decision, now)

        apply_feed_decision_transition(conn, article_id, transition)
        log_event(
            conn,
            article_id,
            transition["event_name"],
            {
                "previous_feed_decision": row_dict["feed_decision"],
                "new_feed_decision": decision,
                "prediction_snapshot": prediction_snapshot,
            },
        )
        deduplicated_ids = archive_pending_duplicates_for_article(conn, row_dict, decision)
    if decision == "summarize":
        wake_summary_worker()
    trigger_feed_similarity_refresh()
    ensure_feed_similarity_snapshot_async()
    return {
        "status": "ok",
        "article_id": article_id,
        "decision": decision,
        "deduplicated_count": len(deduplicated_ids),
    }


def build_feed_decision_transition(article: dict[str, Any], decision: str, decided_at: str) -> dict[str, Any]:
    if decision not in {"skip", "summarize"}:
        raise ValueError("Unsupported feed decision")

    summary_status = article["summary_status"]
    summary_requested_at = article["summary_requested_at"]

    if decision == "skip":
        if summary_status in {"queued", "processing", "ready"}:
            raise RuntimeError("Article is already in summary flow")
        return {
            "decision": decision,
            "decided_at": decided_at,
            "summary_status": summary_status,
            "summary_requested_at": summary_requested_at,
            "event_name": "article_skipped",
        }

    if summary_status in {"not_requested", "failed"}:
        summary_status = "queued"
        summary_requested_at = decided_at
    return {
        "decision": decision,
        "decided_at": decided_at,
        "summary_status": summary_status,
        "summary_requested_at": summary_requested_at,
        "event_name": "summary_requested",
    }


def apply_feed_decision_transition(
    conn: sqlite3.Connection,
    article_id: int,
    transition: dict[str, Any],
) -> None:
    decision = transition["decision"]
    conn.execute(
        """
        UPDATE articles
        SET feed_decision = ?,
            feed_decision_at = ?,
            summary_status = ?,
            summary_requested_at = ?,
            updated_at = ?,
            last_error = CASE WHEN ? = 'summarize' THEN '' ELSE last_error END,
            summary_is_fallback = CASE WHEN ? = 'summarize' THEN 0 ELSE summary_is_fallback END
        WHERE id = ?
        """,
        (
            decision,
            transition["decided_at"],
            transition["summary_status"],
            transition["summary_requested_at"],
            transition["decided_at"],
            decision,
            decision,
            article_id,
        ),
    )


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
            trigger_feed_similarity_refresh()
            ensure_feed_similarity_snapshot_async()
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
    trigger_feed_similarity_refresh()
    ensure_feed_similarity_snapshot_async()
    return result


def set_summary_feedback(article_id: int, feedback: str) -> dict[str, Any]:
    validate_summary_feedback(feedback)
    with db_connection() as conn:
        if not fetch_article_by_id(conn, article_id):
            raise LookupError("Article not found")
        now = utc_now_iso()
        apply_summary_feedback(conn, article_id, feedback, now)
    return {"status": "ok", "article_id": article_id, "feedback": feedback}


def validate_summary_feedback(feedback: str) -> None:
    if feedback not in {"interesting", "not_interesting", "not_available"}:
        raise ValueError("Unsupported summary feedback")


def apply_summary_feedback(
    conn: sqlite3.Connection,
    article_id: int,
    feedback: str,
    feedback_at: str,
) -> None:
    validate_summary_feedback(feedback)
    conn.execute(
        """
        UPDATE articles
        SET summary_feedback = ?,
            summary_feedback_at = ?,
            updated_at = ?
        WHERE id = ?
        """,
        (feedback, feedback_at, feedback_at, article_id),
    )
    log_event(conn, article_id, "summary_feedback", {"feedback": feedback})


def recover_stale_processing_jobs() -> int:
    cutoff = (utc_now() - timedelta(minutes=SUMMARY_PROCESSING_STALE_MINUTES)).isoformat()
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id
            FROM articles
            WHERE summary_status = 'processing'
              AND datetime(updated_at) < datetime(?)
            ORDER BY id
            """,
            (cutoff,),
        ).fetchall()
        if not rows:
            return 0

        stale_ids = [int(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in stale_ids)
        now = utc_now_iso()
        conn.execute(
            f"""
            UPDATE articles
            SET summary_status = 'failed',
                updated_at = ?,
                last_error = CASE
                    WHEN last_error = '' THEN 'Summary processing exceeded the recovery timeout.'
                    ELSE last_error
                END
            WHERE id IN ({placeholders})
            """,
            (now, *stale_ids),
        )
        for article_id in stale_ids:
            log_event(
                conn,
                article_id,
                "summary_processing_recovered",
                {"recovery": "stale_to_failed", "timeout_minutes": SUMMARY_PROCESSING_STALE_MINUTES},
            )
    return len(stale_ids)


def mark_summary_processing(conn: sqlite3.Connection, article_id: int) -> None:
    conn.execute(
        """
        UPDATE articles
        SET summary_status = 'processing',
            updated_at = ?
        WHERE id = ?
        """,
        (utc_now_iso(), article_id),
    )
    log_event(conn, article_id, "summary_processing_started", {})


def select_next_queued_summary_job(conn: sqlite3.Connection) -> Optional[dict[str, Any]]:
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
    return row_to_dict(row)


def claim_next_summary_job() -> Optional[dict[str, Any]]:
    recover_stale_processing_jobs()
    with db_connection() as conn:
        row_dict = select_next_queued_summary_job(conn)
        if not row_dict:
            return None
        mark_summary_processing(conn, row_dict["id"])
    return row_dict


def get_next_summary_job() -> Optional[dict[str, Any]]:
    return claim_next_summary_job()


def mark_summary_ready(
    conn: sqlite3.Connection,
    article_id: int,
    article_text: str,
    final_url: str,
    summary_title: str,
    summary_text: str,
    extraction_fallback: bool = False,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE articles
        SET link_to_article = ?,
            article_text = ?,
            article_text_extracted_at = ?,
            summary_title = ?,
            summary_text = ?,
            summary_is_fallback = ?,
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
            1 if extraction_fallback else 0,
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
        {"model": OLLAMA_MODEL, "final_url": final_url, "extraction_fallback": bool(extraction_fallback)},
    )


def mark_summary_failed(conn: sqlite3.Connection, article_id: int, error_text: str) -> None:
    conn.execute(
        """
        UPDATE articles
        SET summary_status = 'failed',
            updated_at = ?,
            summary_is_fallback = 0,
            last_error = ?
        WHERE id = ?
        """,
        (utc_now_iso(), error_text, article_id),
    )
    log_event(conn, article_id, "summary_failed", {"error": error_text})


def process_summary_job(job: dict[str, Any]) -> None:
    article_id = int(job["id"])
    try:
        article_text, final_url = fetch_best_article_text(job)
        source_text, extraction_fallback = summary_source_text(job, article_text, final_url)

        summary_title, summary_text = ollama_generate_summary(job["title"], source_text)
        with db_connection() as conn:
            mark_summary_ready(
                conn,
                article_id,
                source_text,
                final_url,
                summary_title,
                summary_text,
                extraction_fallback=extraction_fallback,
            )

        if get_compare_enabled():
            wake_compare_worker()
    except Exception as exc:  # pragma: no cover - runtime defensive
        LOGGER.warning("Summary job failed for article %s: %s", article_id, exc)
        with db_connection() as conn:
            mark_summary_failed(conn, article_id, str(exc))


def summary_worker() -> None:
    LOGGER.info("Summary worker started")
    while not STATE.stop_event.is_set():
        STATE.summary_event.clear()
        job = claim_next_summary_job()
        if job is None:
            STATE.summary_event.wait(SUMMARY_POLL_SECONDS)
            continue
        process_summary_job(job)


def fetch_next_compare_job() -> Optional[dict[str, Any]]:
    if not get_compare_enabled():
        return None

    session = current_compare_session()
    if not session:
        return None

    models = decode_json_array(session["models_json"])
    model_count = len(models)
    with db_connection() as conn:
        row = conn.execute(
            """
            SELECT a.*
            FROM articles a
            WHERE a.summary_status = 'ready'
              AND datetime(a.summarized_at) >= datetime(?)
              AND (
                SELECT COUNT(DISTINCT model_name)
                FROM llm_compare_results
                WHERE session_id = ? AND article_id = a.id
              ) < ?
            ORDER BY datetime(a.summarized_at) ASC, a.id ASC
            LIMIT 1
            """,
            (session["enabled_at"], session["id"], model_count),
        ).fetchone()
    return row_to_dict(row) if row else None


def compare_worker() -> None:
    LOGGER.info("LLM compare worker started")
    while not STATE.stop_event.is_set():
        if not get_compare_enabled():
            if STATE.compare_event.wait(SUMMARY_POLL_SECONDS):
                STATE.compare_event.clear()
            if STATE.stop_event.is_set():
                break
            continue

        job = fetch_next_compare_job()
        if job is None:
            if STATE.compare_event.wait(SUMMARY_POLL_SECONDS):
                STATE.compare_event.clear()
            if STATE.stop_event.is_set():
                break
            continue

        try:
            article_text = job.get("article_text") or ""
            final_url = job.get("link_to_article") or job.get("rss_source_url") or ""
            if not article_text and final_url:
                article_text, final_url = fetch_and_extract_article_text(final_url)
            if not article_text:
                LOGGER.warning("LLM compare skipped article %s because article text is empty", job["id"])
                STATE.compare_event.wait(SUMMARY_POLL_SECONDS)
                STATE.compare_event.clear()
                continue
            run_compare_summaries(
                job,
                article_text,
                final_url,
                job.get("summary_title") or job.get("title") or "",
                job.get("summary_text") or "",
            )
        except Exception as exc:  # pragma: no cover - runtime defensive
            LOGGER.warning("LLM compare worker failed for article %s: %s", job.get("id"), exc)
            with db_connection() as conn:
                log_event(
                    conn,
                    int(job["id"]),
                    "llm_compare_failed",
                    {"error": str(exc)},
                )
            clear_compare_progress()


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


def fetch_visible_pending_feed_articles() -> list[dict[str, Any]]:
    return build_visible_pending_feed_items(fetch_pending_feed_articles())


def is_reviewable_summary_status(summary_status: str, summary_feedback: str) -> bool:
    return summary_feedback == "unreviewed" and summary_status in {"ready", "failed"}


def fetch_review_summary_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT *
        FROM articles
        WHERE summary_feedback = 'unreviewed'
          AND summary_status IN ('ready', 'failed')
        ORDER BY datetime(COALESCE(summarized_at, summary_requested_at, updated_at)) DESC, id DESC
        """
    ).fetchall()
    return [row_to_dict(row) for row in rows]


def fetch_review_summaries() -> list[dict[str, Any]]:
    with db_connection() as conn:
        return fetch_review_summary_rows(conn)


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
    trained_at = latest_run["trained_at"]
    return {
        "recommended": delta >= threshold,
        "reason": "enough_new_labels" if delta >= threshold else "delta_too_small",
        "headline": (
            "Neues Training lohnt sich"
            if delta >= threshold
            else "Mit Retraining noch warten"
        ),
        "detail": (
            f"Letzter Lauf: {trained_at}. Seitdem kamen {delta} neue Labels dazu. Ab etwa {threshold} lohnt sich der nächste Lauf."
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
                "detail": "Es gibt noch zu wenig aktuelle positive Beispiele fuer ein wirklich treffsicheres strenges Filtering.",
            }
        if precision < 0.28:
            return {
                "level": "weak",
                "headline": "Noch zu viele falsch Positive",
                "detail": "Fuer deinen Use Case ist das Modell noch nicht selektiv genug. Im Feed besser weiter alle Eintraege sehen und nur hohe Signale ernst nehmen.",
            }
        if precision < 0.4 or f1 < 0.35:
            return {
                "level": "usable",
                "headline": "Vorsichtig selektiv nutzbar",
                "detail": "Das Modell wird brauchbarer, sollte aber weiter eher wenige starke Empfehlungen liefern statt breit zu filtern.",
            }
        return {
            "level": "strong",
            "headline": "Fuer strengeres Filtering brauchbar",
            "detail": "Die Trefferqualitaet ist inzwischen hoch genug, um nur noch die staerkeren Empfehlungen ernsthafter zu beachten.",
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


def attach_model_run_json_fields(run: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not run:
        return run
    if run.get("confusion_matrix_json"):
        run["confusion_matrix"] = decode_json_array(run["confusion_matrix_json"])
    if run.get("notes"):
        run["notes_json"] = decode_json_object(run["notes"])
    return run


def feed_prediction_outcome_stats(limit: int = 1500) -> dict[str, Any]:
    buckets: dict[str, dict[str, int]] = {
        "recommended": {"skip": 0, "summarize": 0},
        "maybe": {"skip": 0, "summarize": 0},
        "no": {"skip": 0, "summarize": 0},
        "unknown": {"skip": 0, "summarize": 0},
    }

    with db_connection() as conn:
        rows = fetch_recent_article_events(conn, ["article_skipped", "summary_requested"], limit)

    considered = 0
    for row in rows:
        payload = decode_event_payload(row["event_payload"])
        snapshot = payload.get("prediction_snapshot") or {}
        tier = snapshot.get("predicted_tier")
        if tier not in buckets:
            tier = "recommended" if snapshot.get("predicted_recommendation") else "unknown"
        action = "summarize" if row["event_type"] == "summary_requested" else "skip"
        buckets[tier][action] += 1
        considered += 1

    return {
        "considered_events": considered,
        "recommended_skip": buckets["recommended"]["skip"],
        "recommended_summarize": buckets["recommended"]["summarize"],
        "maybe_skip": buckets["maybe"]["skip"],
        "maybe_summarize": buckets["maybe"]["summarize"],
        "no_skip": buckets["no"]["skip"],
        "no_summarize": buckets["no"]["summarize"],
        "unknown_skip": buckets["unknown"]["skip"],
        "unknown_summarize": buckets["unknown"]["summarize"],
    }


def collapse_feed_training_rows(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    clusters = cluster_items_by_similarity(list(rows))
    collapsed_rows: list[dict[str, Any]] = []

    for cluster in clusters:
        members = [row if isinstance(row, dict) else row_to_dict(row) for row in cluster["members"]]
        summarize_members = [member for member in members if member.get("feed_decision") == "summarize"]
        skip_members = [member for member in members if member.get("feed_decision") == "skip"]

        if summarize_members:
            chosen = max(
                summarize_members,
                key=lambda item: (
                    parse_datetime(item.get("feed_decision_at") or item.get("published_at")),
                    int(item.get("id", 0)),
                ),
            )
            chosen = dict(chosen)
            chosen["feed_decision"] = "summarize"
            chosen["story_variant_count"] = len(members)
            chosen["story_label_strategy"] = "summarize_dominates_cluster"
            collapsed_rows.append(chosen)
            continue

        if skip_members:
            chosen = max(
                skip_members,
                key=lambda item: (
                    parse_datetime(item.get("feed_decision_at") or item.get("published_at")),
                    int(item.get("id", 0)),
                ),
            )
            chosen = dict(chosen)
            chosen["feed_decision"] = "skip"
            chosen["story_variant_count"] = len(members)
            chosen["story_label_strategy"] = "all_cluster_members_skipped"
            collapsed_rows.append(chosen)

    return collapsed_rows


def precision_at_k(probabilities: list[float], labels: list[int], k: int) -> dict[str, Any]:
    if not probabilities or not labels:
        return {"value": None, "evaluated_count": 0}
    ranked = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    top = ranked[: min(k, len(ranked))]
    if not top:
        return {"value": None, "evaluated_count": 0}
    positives = sum(label for _, label in top)
    return {
        "value": round(positives / len(top), 4),
        "evaluated_count": len(top),
    }


def select_feed_threshold(labels: list[int], probabilities: list[float]) -> tuple[float, float, dict[str, Any]]:
    if not probabilities:
        return 0.5, 0.35, {
            "strategy": "fixed_default",
            "objective": "precision_first_two_stage",
            "candidate_count": 1,
            "beta": 0.5,
        }

    candidate_thresholds = sorted({round(float(probability), 6) for probability in probabilities} | {0.5})
    baseline_predicted = [1 if probability >= 0.5 else 0 for probability in probabilities]
    baseline_precision = float(precision_score(labels, baseline_predicted, zero_division=0))
    baseline_recall = float(recall_score(labels, baseline_predicted, zero_division=0))
    beta = 0.5
    precision_floor = max(0.22, baseline_precision * 1.05)
    recall_floor = max(0.10, baseline_recall * 0.55)

    eligible_snapshots: list[tuple[float, float, float, float, float]] = []
    fallback_snapshots: list[tuple[float, float, float, float, float]] = []

    for threshold in candidate_thresholds:
        predicted = [1 if probability >= threshold else 0 for probability in probabilities]
        precision = float(precision_score(labels, predicted, zero_division=0))
        recall = float(recall_score(labels, predicted, zero_division=0))
        fbeta = float(fbeta_score(labels, predicted, beta=beta, zero_division=0))
        snapshot = (fbeta, precision, recall, -abs(threshold - 0.6), threshold)
        fallback_snapshots.append(snapshot)
        if precision >= precision_floor and recall >= recall_floor:
            eligible_snapshots.append(snapshot)

    chosen_snapshot = max(eligible_snapshots or fallback_snapshots)
    best_fbeta, best_precision, best_recall, _distance_bias, best_threshold = chosen_snapshot

    maybe_beta = 1.0
    maybe_precision_floor = max(0.16, best_precision * 0.72)
    maybe_recall_floor = max(best_recall, baseline_recall * 0.95)
    maybe_candidates: list[tuple[float, float, float, float, float]] = []
    for threshold in candidate_thresholds:
        if threshold > best_threshold:
            continue
        predicted = [1 if probability >= threshold else 0 for probability in probabilities]
        precision = float(precision_score(labels, predicted, zero_division=0))
        recall = float(recall_score(labels, predicted, zero_division=0))
        fbeta = float(fbeta_score(labels, predicted, beta=maybe_beta, zero_division=0))
        if precision >= maybe_precision_floor and recall >= maybe_recall_floor:
            maybe_candidates.append((fbeta, recall, precision, -abs(threshold - (best_threshold - 0.08)), threshold))
    if maybe_candidates:
        _maybe_fbeta, maybe_recall, maybe_precision, _bias, maybe_threshold = max(maybe_candidates)
    else:
        maybe_threshold = max(0.0, round(best_threshold - 0.10, 4))
        maybe_predicted = [1 if probability >= maybe_threshold else 0 for probability in probabilities]
        maybe_precision = float(precision_score(labels, maybe_predicted, zero_division=0))
        maybe_recall = float(recall_score(labels, maybe_predicted, zero_division=0))

    maybe_threshold = round(float(min(maybe_threshold, best_threshold)), 4)

    return round(float(best_threshold), 4), maybe_threshold, {
        "strategy": "auto_tuned",
        "objective": "precision_first_two_stage",
        "candidate_count": len(candidate_thresholds),
        "beta": beta,
        "baseline_precision": round(baseline_precision, 4),
        "baseline_recall": round(baseline_recall, 4),
        "precision_floor": round(precision_floor, 4),
        "recall_floor": round(recall_floor, 4),
        "best_fbeta": round(best_fbeta, 4),
        "best_precision": round(best_precision, 4),
        "best_recall": round(best_recall, 4),
        "used_precision_floor": bool(eligible_snapshots),
        "maybe_threshold": maybe_threshold,
        "maybe_precision": round(maybe_precision, 4),
        "maybe_recall": round(maybe_recall, 4),
        "maybe_precision_floor": round(maybe_precision_floor, 4),
    }


def should_promote_model(
    target: str,
    candidate_metrics: dict[str, float],
    candidate_notes: Optional[dict[str, Any]],
    current_active_run: Optional[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    if current_active_run is None:
        return True, {
            "decision": "promote",
            "reason": "no_existing_active_model",
        }

    if target != "feed_recommendation":
        return True, {
            "decision": "promote",
            "reason": "non_feed_target",
        }

    active_notes = decode_json_object(current_active_run.get("notes"))

    active_f1 = float(current_active_run.get("f1", 0.0))
    active_precision = float(current_active_run.get("precision", 0.0))
    active_recall = float(current_active_run.get("recall", 0.0))

    candidate_f1 = float(candidate_metrics.get("f1", 0.0))
    candidate_precision = float(candidate_metrics.get("precision", 0.0))
    candidate_recall = float(candidate_metrics.get("recall", 0.0))
    candidate_p20 = float((candidate_notes or {}).get("precision_at_k", {}).get("20", {}).get("value") or 0.0)
    active_p20 = float((active_notes or {}).get("precision_at_k", {}).get("20", {}).get("value") or 0.0)

    if candidate_precision >= active_precision + 0.02 and candidate_f1 >= active_f1 - 0.01:
        return True, {
            "decision": "promote",
            "reason": "precision_improved_with_stable_f1",
            "champion_run_id": current_active_run["id"],
        }

    if candidate_p20 >= active_p20 + 0.05 and candidate_precision >= active_precision - 0.01:
        return True, {
            "decision": "promote",
            "reason": "precision_at_20_improved",
            "champion_run_id": current_active_run["id"],
        }

    if candidate_f1 >= active_f1 + 0.01:
        return True, {
            "decision": "promote",
            "reason": "f1_improved_meaningfully",
            "champion_run_id": current_active_run["id"],
        }

    if (
        candidate_f1 >= active_f1
        and candidate_precision >= active_precision
        and candidate_recall >= active_recall - 0.02
    ):
        return True, {
            "decision": "promote",
            "reason": "no_regression_with_precision_hold",
            "champion_run_id": current_active_run["id"],
        }

    return False, {
        "decision": "reject",
        "reason": "candidate_worse_than_active_model",
        "champion_run_id": current_active_run["id"],
        "active_f1": round(active_f1, 4),
        "active_precision": round(active_precision, 4),
        "active_recall": round(active_recall, 4),
        "candidate_f1": round(candidate_f1, 4),
        "candidate_precision": round(candidate_precision, 4),
        "candidate_recall": round(candidate_recall, 4),
        "active_p20": round(active_p20, 4),
        "candidate_p20": round(candidate_p20, 4),
    }


def train_model(target: str) -> dict[str, Any]:
    if target not in TARGET_CONFIG:
        raise ValueError(f"Unsupported target {target}")

    training_started = time.perf_counter()
    config = TARGET_CONFIG[target]
    current_active_run = latest_model_run(target, include_rejected=False)
    with db_connection() as conn:
        rows = conn.execute(config["query"]).fetchall()

    raw_label_count = len(rows)
    collapsed_cluster_count = None
    if target == "feed_recommendation":
        collapsed_rows = collapse_feed_training_rows(rows)
        rows = collapsed_rows
        collapsed_cluster_count = len(rows)

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
    threshold_value = 0.5
    maybe_threshold_value = 0.35
    evaluation_notes: dict[str, Any] = {}
    if target == "feed_recommendation":
        probabilities = [float(value) for value in pipeline.predict_proba(test_x)[:, 1]]
        threshold_value, maybe_threshold_value, threshold_notes = select_feed_threshold(test_y, probabilities)
        predicted = [1 if probability >= threshold_value else 0 for probability in probabilities]
        evaluation_notes = {
            "threshold_strategy": threshold_notes,
            "maybe_threshold_value": maybe_threshold_value,
            "precision_at_k": {
                "10": precision_at_k(probabilities, test_y, 10),
                "20": precision_at_k(probabilities, test_y, 20),
                "50": precision_at_k(probabilities, test_y, 50),
            },
        }
    else:
        predicted = pipeline.predict(test_x)
    matrix = confusion_matrix(test_y, predicted, labels=[0, 1]).tolist()
    accuracy = float(accuracy_score(test_y, predicted))
    precision = float(precision_score(test_y, predicted, zero_division=0))
    recall = float(recall_score(test_y, predicted, zero_division=0))
    f1 = float(f1_score(test_y, predicted, zero_division=0))
    training_duration_ms = int((time.perf_counter() - training_started) * 1000)

    trained_at = utc_now_iso()
    model_path = MODEL_DIR / config["artifact_name"]
    promotion_allowed, promotion_notes = should_promote_model(
        target,
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        evaluation_notes,
        current_active_run,
    )
    effective_model_path = model_path
    if not promotion_allowed:
        effective_model_path = model_path.with_name(
            f"{model_path.stem}-candidate-{trained_at.replace(':', '').replace('-', '')}.joblib"
        )
    notes_payload = {
        "training_duration_ms": training_duration_ms,
        **evaluation_notes,
        "promotion_decision": promotion_notes,
        "raw_label_count": raw_label_count,
        "collapsed_cluster_count": collapsed_cluster_count,
    }
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
                str(effective_model_path),
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
                threshold_value,
                json.dumps(matrix),
                json.dumps(notes_payload),
                "active" if promotion_allowed else "rejected",
            ),
        )
        run_id = int(cursor.lastrowid)

    artifact = {
        "target": target,
        "run_id": run_id,
        "trained_at": trained_at,
        "threshold": threshold_value,
        "maybe_threshold": maybe_threshold_value,
        "pipeline": pipeline,
    }
    joblib.dump(artifact, effective_model_path)

    if promotion_allowed:
        with STATE.model_lock:
            STATE.models[target] = artifact

        if target == "feed_recommendation":
            rows_for_prediction = fetch_visible_pending_feed_articles()
            predicted_rows, used_run_id = predict_feed_rows(rows_for_prediction)
            update_cached_feed_predictions(predicted_rows, used_run_id)

    return {
        "target": target,
        "status": "trained" if promotion_allowed else "rejected",
        "run_id": run_id,
        "trained_at": trained_at,
        "labels_used": len(labels),
        "positive_labels": positives,
        "negative_labels": negatives,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold_value": threshold_value,
        "maybe_threshold_value": maybe_threshold_value,
        "training_duration_ms": training_duration_ms,
        "notes": notes_payload,
        "promoted": promotion_allowed,
        "confusion_matrix": matrix,
    }


def train_targets(targets: list[str]) -> list[dict[str, Any]]:
    results = []
    with STATE.training_lock:
        started_at = utc_now_iso()
        with STATE.training_status_lock:
            STATE.training_status = {
                "active": True,
                "target": "all" if len(targets) > 1 else targets[0],
                "started_at": started_at,
                "finished_at": None,
                "results": None,
                "error": None,
            }
        try:
            for target in targets:
                results.append(train_model(target))
            with STATE.training_status_lock:
                STATE.training_status = {
                    "active": False,
                    "target": "all" if len(targets) > 1 else targets[0],
                    "started_at": started_at,
                    "finished_at": utc_now_iso(),
                    "results": results,
                    "error": None,
                }
        except Exception as exc:
            with STATE.training_status_lock:
                STATE.training_status = {
                    "active": False,
                    "target": "all" if len(targets) > 1 else targets[0],
                    "started_at": started_at,
                    "finished_at": utc_now_iso(),
                    "results": None,
                    "error": str(exc),
                }
            raise
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
        "similar_count": item.get("similar_count", 0),
        "prediction": item["prediction"],
    }


def serialize_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "title": item["title"],
        "summary_title": item["summary_title"],
        "summary_text": item["summary_text"],
        "summary_is_fallback": bool(item.get("summary_is_fallback")),
        "summary_status": item["summary_status"],
        "source_label": item["source_label"],
        "source_url": item["source_url"],
        "published_at": item["published_at"],
        "link_to_article": item["link_to_article"],
        "summary_model": item["summary_model"],
        "summary_feedback": item["summary_feedback"],
    }


def build_summaries_api_payload() -> dict[str, Any]:
    return {
        "items": [serialize_summary(item) for item in fetch_review_summaries()],
        "counts": summary_counts(),
    }


def fetch_summary_counts(conn: sqlite3.Connection) -> dict[str, int]:
    row = conn.execute(
        """
        SELECT
            SUM(CASE WHEN summary_status = 'queued' THEN 1 ELSE 0 END) AS queued,
            SUM(CASE WHEN summary_status = 'processing' THEN 1 ELSE 0 END) AS processing,
            SUM(CASE WHEN summary_status = 'ready' AND summary_feedback = 'unreviewed' THEN 1 ELSE 0 END) AS ready,
            SUM(CASE WHEN summary_status = 'failed' AND summary_feedback = 'unreviewed' THEN 1 ELSE 0 END) AS failed
        FROM articles
        """
    ).fetchone()
    ready = int(row["ready"] or 0)
    failed = int(row["failed"] or 0)
    return {
        "queued": int(row["queued"] or 0),
        "processing": int(row["processing"] or 0),
        "ready": ready,
        "failed": failed,
        "review_total": ready + failed,
    }


def summary_counts() -> dict[str, int]:
    with db_connection() as conn:
        return fetch_summary_counts(conn)


def classify_primary_summary_failure(error_text: str) -> str:
    lower = (error_text or "").lower()
    if "127.0.0.1" in lower or ":11434" in lower or "ollama" in lower:
        return "ollama"
    return "source"


def llm_compare_status() -> dict[str, Any]:
    session = current_compare_session()
    diagnostics_session = latest_compare_session()
    progress = get_app_state("llm_compare_progress")
    session_stats: Optional[dict[str, Any]] = None
    diagnostics: Optional[dict[str, Any]] = None

    if session:
        models = decode_json_array(session["models_json"])
        total_models = max(len(models), 1)
        with db_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    COUNT(*) AS result_count,
                    COUNT(DISTINCT article_id) AS article_count,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count
                FROM llm_compare_results
                WHERE session_id = ?
                """,
                (session["id"],),
            ).fetchone()
            completed_articles_row = conn.execute(
                """
                SELECT COUNT(*) AS completed_articles
                FROM (
                    SELECT article_id
                    FROM llm_compare_results
                    WHERE session_id = ?
                    GROUP BY article_id
                    HAVING COUNT(DISTINCT model_name) >= ?
                )
                """,
                (session["id"], total_models),
            ).fetchone()
            pending_articles_row = conn.execute(
                """
                SELECT COUNT(*) AS pending_articles
                FROM articles a
                WHERE a.summary_status = 'ready'
                  AND datetime(a.summarized_at) >= datetime(?)
                  AND (
                    SELECT COUNT(DISTINCT model_name)
                    FROM llm_compare_results
                    WHERE session_id = ? AND article_id = a.id
                  ) < ?
                """,
                (session["enabled_at"], session["id"], total_models),
            ).fetchone()
            primary_rows = conn.execute(
                """
                SELECT
                    COUNT(*) AS requested_total,
                    SUM(CASE WHEN summary_status = 'queued' THEN 1 ELSE 0 END) AS queued_count,
                    SUM(CASE WHEN summary_status = 'processing' THEN 1 ELSE 0 END) AS processing_count,
                    SUM(CASE WHEN summary_status = 'ready' THEN 1 ELSE 0 END) AS ready_count,
                    SUM(CASE WHEN summary_status = 'failed' THEN 1 ELSE 0 END) AS failed_count
                FROM articles
                WHERE feed_decision = 'summarize'
                  AND summary_requested_at IS NOT NULL
                  AND datetime(summary_requested_at) >= datetime(?)
                """,
                (session["enabled_at"],),
            ).fetchone()
            failed_primary_rows = conn.execute(
                """
                SELECT last_error
                FROM articles
                WHERE feed_decision = 'summarize'
                  AND summary_status = 'failed'
                  AND summary_requested_at IS NOT NULL
                  AND datetime(summary_requested_at) >= datetime(?)
                """,
                (session["enabled_at"],),
            ).fetchall()

        result_count = int(rows["result_count"] if rows else 0)
        article_count = int(rows["article_count"] if rows else 0)
        compare_ok_count = int(rows["ok_count"] if rows and rows["ok_count"] is not None else 0)
        compare_failed_count = int(rows["failed_count"] if rows and rows["failed_count"] is not None else 0)
        completed_articles = int(completed_articles_row["completed_articles"] if completed_articles_row else 0)
        pending_articles = int(pending_articles_row["pending_articles"] if pending_articles_row else 0)
        requested_total = int(primary_rows["requested_total"] if primary_rows and primary_rows["requested_total"] is not None else 0)
        queued_count = int(primary_rows["queued_count"] if primary_rows and primary_rows["queued_count"] is not None else 0)
        processing_count = int(primary_rows["processing_count"] if primary_rows and primary_rows["processing_count"] is not None else 0)
        ready_count = int(primary_rows["ready_count"] if primary_rows and primary_rows["ready_count"] is not None else 0)
        failed_count = int(primary_rows["failed_count"] if primary_rows and primary_rows["failed_count"] is not None else 0)
        primary_failed_source = 0
        primary_failed_ollama = 0
        for row in failed_primary_rows:
            failure_type = classify_primary_summary_failure(row["last_error"] or "")
            if failure_type == "ollama":
                primary_failed_ollama += 1
            else:
                primary_failed_source += 1
        session_stats = {
            "requested_total": requested_total,
            "primary_queued": queued_count,
            "primary_processing": processing_count,
            "primary_ready": ready_count,
            "primary_failed": failed_count,
            "primary_failed_source": primary_failed_source,
            "primary_failed_ollama": primary_failed_ollama,
            "compare_article_count": article_count,
            "compare_candidate_total": ready_count,
            "completed_articles": completed_articles,
            "pending_articles": pending_articles,
            "articles_in_progress": 1 if progress and progress.get("session_id") == session["id"] else 0,
            "result_count": result_count,
            "compare_ok_results": compare_ok_count,
            "compare_failed_results": compare_failed_count,
            "total_models": total_models,
        }

    if diagnostics_session:
        models = decode_json_array(diagnostics_session["models_json"])
        with db_connection() as conn:
            aggregate_rows = conn.execute(
                """
                SELECT
                    model_name,
                    SUM(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) AS ok_count,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_count
                FROM llm_compare_results
                WHERE session_id = ?
                GROUP BY model_name
                """,
                (diagnostics_session["id"],),
            ).fetchall()
            latest_rows = conn.execute(
                """
                SELECT
                    r.model_name,
                    r.status,
                    r.duration_ms,
                    r.error_text,
                    r.created_at,
                    r.article_id,
                    a.title AS article_title
                FROM llm_compare_results r
                JOIN articles a ON a.id = r.article_id
                WHERE r.session_id = ?
                ORDER BY r.model_name ASC, datetime(r.created_at) DESC, r.id DESC
                """,
                (diagnostics_session["id"],),
            ).fetchall()

        aggregates = {row["model_name"]: row_to_dict(row) for row in aggregate_rows}
        latest_by_model: dict[str, dict[str, Any]] = {}
        for row in latest_rows:
            if row["model_name"] not in latest_by_model:
                latest_by_model[row["model_name"]] = row_to_dict(row)

        diagnostics_models: list[dict[str, Any]] = []
        for model_name in models:
            latest = latest_by_model.get(model_name, {})
            aggregate = aggregates.get(model_name, {})
            diagnostics_models.append(
                {
                    "model_name": model_name,
                    "last_status": latest.get("status"),
                    "last_duration_ms": latest.get("duration_ms"),
                    "last_error": latest.get("error_text") or "",
                    "last_completed_at": latest.get("created_at"),
                    "last_article_id": latest.get("article_id"),
                    "last_article_title": latest.get("article_title") or "",
                    "ok_count": int(aggregate.get("ok_count", 0) or 0),
                    "failed_count": int(aggregate.get("failed_count", 0) or 0),
                }
            )

        diagnostics = {
            "session_id": diagnostics_session["id"],
            "session_status": diagnostics_session["status"],
            "enabled_at": diagnostics_session["enabled_at"],
            "disabled_at": diagnostics_session.get("disabled_at"),
            "export_path": diagnostics_session["export_path"],
            "models": diagnostics_models,
        }

    return {
        "enabled": get_compare_enabled(),
        "models": get_compare_models(),
        "session": session,
        "last_session": diagnostics_session,
        "progress": progress,
        "session_stats": session_stats,
        "diagnostics": diagnostics,
    }


def feed_counts() -> dict[str, int]:
    snapshot = current_feed_similarity_snapshot()
    pending = int(snapshot.get("visible_total", count_rows("feed_decision = 'pending'")))
    similar_groups = int(snapshot.get("similar_group_count", 0))
    similar_hidden = int(snapshot.get("similar_hidden_count", 0))
    with db_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM articles
            WHERE feed_decision = 'pending'
            ORDER BY datetime(published_at) DESC, id DESC
            """
        ).fetchall()
    predicted_rows, _run_id = predict_feed_rows(list(rows))
    recommended = sum(1 for row in predicted_rows if row["prediction"]["recommended"])
    maybe = sum(1 for row in predicted_rows if row["prediction"].get("tier") == "maybe")
    return {
        "pending": pending,
        "recommended": recommended,
        "maybe": maybe,
        "similar_groups": similar_groups,
        "similar_hidden": similar_hidden,
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
        return api_error_response(error.description, error.code or 500)

    LOGGER.exception("Unhandled API error: %s", error)
    return api_error_response(str(error) or "Internal server error", 500)


def api_error_response(message: str, status_code: int):
    return jsonify({"status": "error", "message": message}), status_code


def run_feed_action_api(article_id: int, decision: str):
    try:
        return jsonify(set_feed_decision(article_id, decision))
    except LookupError:
        return api_error_response("Article not found", 404)
    except RuntimeError as exc:
        return api_error_response(str(exc), 409)


def run_summary_feedback_api(article_id: int, feedback: Any):
    try:
        return jsonify(set_summary_feedback(article_id, feedback))
    except ValueError:
        return api_error_response("Unsupported feedback", 400)
    except LookupError:
        return api_error_response("Article not found", 404)


def build_model_ops_target_payload(target: str) -> dict[str, Any]:
    latest_run = attach_model_run_json_fields(latest_model_run(target))
    previous_run = attach_model_run_json_fields(previous_model_run(target))
    active_run = attach_model_run_json_fields(latest_model_run(target, include_rejected=False))
    payload = {
        "latest_run": latest_run,
        "previous_run": previous_run,
        "active_run": active_run,
        "label_counts": latest_labels_count(target),
        "new_labels_since_training": new_labels_since_training(
            target,
            active_run["trained_at"] if active_run else None,
        ),
        "retraining": retraining_recommendation(target, active_run),
        "quality": model_quality_assessment(target, active_run),
    }
    if target == "feed_recommendation":
        payload["prediction_outcomes"] = feed_prediction_outcome_stats()
    return payload


def build_model_ops_payload() -> dict[str, Any]:
    return {
        "targets": {target: build_model_ops_target_payload(target) for target in TARGET_CONFIG},
        "training": get_training_status(),
    }


def build_status_config_payload() -> dict[str, Any]:
    return {
        "config_path": str(CONFIG_PATH),
        "ollama_model": OLLAMA_MODEL,
        "ollama_embedding_model": get_embedding_model(),
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_summary_timeout_seconds": get_summary_timeout_seconds(),
        "ollama_embedding_timeout_seconds": get_embedding_timeout_seconds(),
        "feed_refresh_seconds": FEED_REFRESH_SECONDS,
        "embedding_poll_seconds": EMBEDDING_POLL_SECONDS,
        "feed_count": len(RSS_FEED_URLS),
        "llm_compare_models": get_compare_models(),
        "llm_compare_export_dir": str(COMPARE_EXPORT_DIR),
        "llm_compare_timeout_seconds": get_compare_timeout_seconds(),
    }


def build_status_models_payload() -> dict[str, Any]:
    return {
        "feed_recommendation": {
            "loaded": get_loaded_model("feed_recommendation") is not None,
            "latest_run": latest_model_run("feed_recommendation", include_rejected=False),
        },
        "summary_interest": {
            "loaded": get_loaded_model("summary_interest") is not None,
            "latest_run": latest_model_run("summary_interest", include_rejected=False),
        },
    }


def build_status_payload() -> dict[str, Any]:
    return {
        "feed": feed_counts(),
        "summaries": summary_counts(),
        "models": build_status_models_payload(),
        "llm_compare": llm_compare_status(),
        "ollama": ollama_health(),
        "last_feed_refresh": get_app_state("last_feed_refresh"),
        "config": build_status_config_payload(),
    }


@APP.before_request
def handle_options():  # type: ignore[no-untyped-def]
    if request.method == "OPTIONS":
        return ("", 204)
    return None


@APP.get("/")
@APP.get("/app")
def app_index():
    return send_file(APP_HTML_PATH)


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
    return jsonify(build_status_payload())


@APP.post("/api/feeds/refresh")
def api_refresh_feeds():
    return jsonify(refresh_feeds())


@APP.post("/api/llm-compare")
def api_llm_compare_toggle():
    body = request.get_json(silent=True) or {}
    enabled = bool(body.get("enabled", False))
    return jsonify(set_compare_enabled(enabled))


@APP.post("/api/feed/reset")
def api_feed_reset():
    return jsonify(archive_pending_feed())


@APP.post("/api/feed/deduplicate")
def api_feed_deduplicate():
    return jsonify(deduplicate_current_pending_feed())


@APP.get("/api/feed")
def api_feed():
    return jsonify(build_feed_api_payload(request.args.get("mode", "all")))


@APP.post("/api/articles/<int:article_id>/skip")
def api_skip_article(article_id: int):
    return run_feed_action_api(article_id, "skip")


@APP.post("/api/articles/<int:article_id>/summarize")
def api_summarize_article(article_id: int):
    return run_feed_action_api(article_id, "summarize")


@APP.get("/api/summaries")
def api_summaries():
    return jsonify(build_summaries_api_payload())


@APP.post("/api/summaries/<int:article_id>/feedback")
def api_summary_feedback(article_id: int):
    body = request.get_json(silent=True) or {}
    return run_summary_feedback_api(article_id, body.get("feedback"))


@APP.get("/api/model-ops")
def api_model_ops():
    return jsonify(build_model_ops_payload())


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
        threading.Thread(target=compare_worker, name="llm-compare-worker", daemon=True),
        threading.Thread(target=feed_refresh_worker, name="feed-refresh-worker", daemon=True),
        threading.Thread(target=embedding_worker, name="embedding-worker", daemon=True),
    ]
    for thread in threads:
        thread.start()
    return threads


def shutdown() -> None:
    STATE.stop_event.set()


def main() -> None:
    init_db()
    load_persisted_models()
    bootstrap_compare_mode()
    ensure_feed_similarity_snapshot_async()
    start_background_threads()
    LOGGER.info("Starting local news backend on http://%s:%s", HOST, PORT)
    try:
        APP.run(host=HOST, port=PORT, debug=False, use_reloader=False)
    finally:
        shutdown()


if __name__ == "__main__":
    main()
