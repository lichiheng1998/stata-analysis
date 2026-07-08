import argparse
import csv
import json
import logging
import re
import sqlite3
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pysbd
from FlagEmbedding import FlagModel

try:
    import jieba
except ImportError:
    jieba = None


ROOT = Path(__file__).resolve().parents[1]
TEXT_ROOT = ROOT / "txt 2001-2025" / "管理层讨论与分析"
THEME_PATH = ROOT / "数字化主题向量列表.txt"
DEFAULT_KEYWORD_PATH = ROOT / "digital_keywords.txt"
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_LOG_DIR = ROOT / "logs"
MODEL_NAME = "BAAI/bge-base-zh-v1.5"
ANNUAL_RE = re.compile(r"^(?P<stock_id>\d{6})_(?P<report_date>(?P<year>\d{4})-12-31)\.txt$")
MIN_SENTENCE_CHARS = 8
WEAK_KEYWORDS = {
    "\u4fe1\u606f",  # 信息
    "\u6570\u636e",  # 数据
    "\u667a\u80fd",  # 智能
    "\u8054\u7f51",  # 联网
    "\u901a\u4fe1",  # 通信
    "\u673a\u5668",  # 机器
    "\u4ea7\u4e1a\u94fe",  # 产业链
    "\u4ea7\u5b66\u7814",  # 产学研
    "\u5173\u952e\u6280\u672f",  # 关键技术
    "\u6838\u5fc3\u6280\u672f",  # 核心技术
    "\u6280\u672f\u5f00\u53d1",  # 技术开发
    "\u6280\u672f\u6539\u9020",  # 技术改造
    "\u7535\u52a8",  # 电动
    "\u8054\u901a",  # 联通
}

REPORT_FIELDS = [
    "stock_id",
    "year",
    "report_date",
    "file_path",
    "threshold",
    "total_chars",
    "total_sentences",
    "digital_chars",
    "digital_sentences",
    "digital_char_ratio",
    "digital_sent_ratio",
    "max_score",
    "avg_digital_score",
    "top_theme_id",
    "top_theme_name",
]

MATCH_FIELDS = [
    "stock_id",
    "year",
    "sentence_id",
    "sentence",
    "char_count",
    "max_score",
    "theme_id",
    "theme_name",
    "is_digital",
    "matched_keywords",
    "match_method",
]


@dataclass(frozen=True)
class Task:
    stock_id: str
    year: int
    report_date: str
    path: Path


@dataclass
class PreparedReport:
    task: Task
    total_chars: int
    total_sentences: int
    sentences: list[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annual MD&A digitalization sentence embedding pipeline."
    )
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--start-year", type=int, default=2001)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--gpu-report-batch-size",
        type=int,
        default=16,
        help="Number of prepared reports to combine into one GPU encode call.",
    )
    parser.add_argument(
        "--max-gpu-sentences",
        type=int,
        default=8192,
        help="Flush the GPU report batch when accumulated sentences reach this count.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--exclude-theme-ids",
        default="",
        help="Comma-separated theme ids to exclude from embedding matching, e.g. D14.",
    )
    parser.add_argument("--keyword-path", type=Path, default=DEFAULT_KEYWORD_PATH)
    parser.add_argument(
        "--disable-keyword-match",
        action="store_true",
        help="Only use embedding threshold; ignore keyword lexicon matches.",
    )
    parser.add_argument(
        "--keyword-match-mode",
        choices=["strict", "any"],
        default="strict",
        help="strict ignores generic standalone terms such as 数据/信息; any matches every keyword.",
    )
    parser.add_argument(
        "--keyword-tokenizer",
        choices=["jieba", "substring"],
        default="jieba",
        help="Use jieba token matching for keyword hits, or raw substring matching.",
    )
    parser.add_argument(
        "--match-rule",
        choices=[
            "keyword-or-embedding",
            "keyword-only",
            "embedding-only",
            "keyword-and-embedding",
        ],
        default="keyword-or-embedding",
        help="Control whether keyword hits, embedding hits, or both define a matched sentence.",
    )
    parser.add_argument(
        "--keyword-score-threshold",
        type=float,
        default=0.5,
        help="Minimum embedding score required for keyword-hit sentences.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--resume", action="store_true", help="Skip completed reports.")
    parser.add_argument("--retry-failed", action="store_true", help="Only run failed reports.")
    parser.add_argument("--force", action="store_true", help="Clear pipeline state and rerun.")
    parser.add_argument("--limit", type=int, default=None, help="Limit reports for testing.")
    parser.add_argument(
        "--save-matches",
        choices=["all", "sample", "none"],
        default="sample",
        help="Save sentence-level audit rows.",
    )
    parser.add_argument("--sample-per-report", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export CSV files from the SQLite state database.",
    )
    return parser.parse_args()


def setup_logging(log_dir: Path, verbose: bool):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"digital_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Log file: %s", log_path)
    logging.debug("Verbose logging enabled.")


def connect_state(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            stock_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            report_date TEXT NOT NULL,
            file_path TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS report_results (
            stock_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            report_date TEXT NOT NULL,
            file_path TEXT NOT NULL PRIMARY KEY,
            threshold REAL NOT NULL,
            total_chars INTEGER NOT NULL,
            total_sentences INTEGER NOT NULL,
            digital_chars INTEGER NOT NULL,
            digital_sentences INTEGER NOT NULL,
            digital_char_ratio REAL NOT NULL,
            digital_sent_ratio REAL NOT NULL,
            max_score REAL NOT NULL,
            avg_digital_score REAL NOT NULL,
            top_theme_id TEXT,
            top_theme_name TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentence_matches (
            file_path TEXT NOT NULL,
            stock_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            sentence_id INTEGER NOT NULL,
            sentence TEXT NOT NULL,
            char_count INTEGER NOT NULL,
            max_score REAL NOT NULL,
            theme_id TEXT NOT NULL,
            theme_name TEXT NOT NULL,
            is_digital INTEGER NOT NULL,
            matched_keywords TEXT NOT NULL DEFAULT '',
            match_method TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (file_path, sentence_id)
        )
        """
    )
    ensure_column(conn, "sentence_matches", "matched_keywords", "TEXT NOT NULL DEFAULT ''")
    ensure_column(conn, "sentence_matches", "match_method", "TEXT NOT NULL DEFAULT ''")
    conn.commit()
    return conn


def ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str):
    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def utcnow():
    return datetime.utcnow().isoformat(timespec="seconds")


def parse_theme_ids(value: str):
    return {part.strip() for part in value.split(",") if part.strip()}


def load_themes(exclude_theme_ids: set[str] | None = None):
    exclude_theme_ids = exclude_theme_ids or set()
    themes = []
    for line in THEME_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            theme = json.loads(line)
            if theme.get("id") not in exclude_theme_ids:
                themes.append(theme)
    if not themes:
        raise RuntimeError(f"No themes loaded from {THEME_PATH}")
    if exclude_theme_ids:
        logging.info("Excluded theme ids from embedding matching: %s", sorted(exclude_theme_ids))
    logging.debug("Loaded %s themes from %s", len(themes), THEME_PATH)
    return themes


def load_keywords(path: Path, disabled: bool):
    if disabled:
        logging.info("Keyword lexicon matching disabled.")
        return []
    if not path.exists():
        logging.warning("Keyword lexicon not found: %s; embedding-only matching will be used.", path)
        return []
    keywords = []
    seen = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        keyword = line.strip()
        if keyword and keyword not in seen:
            seen.add(keyword)
            keywords.append(keyword)
    keywords.sort(key=len, reverse=True)
    logging.info("Loaded keyword lexicon: %s terms from %s", len(keywords), path)
    logging.debug("Keyword lexicon terms: %s", keywords)
    return keywords


def register_jieba_keywords(keywords: list[str], tokenizer: str):
    if tokenizer != "jieba" or not keywords:
        return
    if jieba is None:
        raise RuntimeError("jieba is required when --keyword-tokenizer jieba is used.")
    for keyword in keywords:
        jieba.add_word(keyword)
    logging.info("Registered %s keyword terms in jieba dictionary.", len(keywords))


def tokenize_sentence(sentence: str):
    if jieba is None:
        raise RuntimeError("jieba is required when --keyword-tokenizer jieba is used.")
    return {token.strip() for token in jieba.lcut(sentence) if token.strip()}


def keyword_hits(sentence: str, keywords: list[str], match_mode: str, tokenizer: str):
    if not keywords:
        return []

    if tokenizer == "jieba":
        tokens = tokenize_sentence(sentence)
        hits = [
            keyword
            for keyword in keywords
            if (keyword in tokens if len(keyword) <= 3 else keyword in sentence)
        ]
    else:
        hits = [keyword for keyword in keywords if keyword in sentence]

    if match_mode == "any":
        return hits
    strong_hits = [keyword for keyword in hits if keyword not in WEAK_KEYWORDS]
    if strong_hits:
        return hits
    return []


def classify_match(
    score: float,
    threshold: float,
    keyword_score_threshold: float,
    hits: list[str],
    match_rule: str,
):
    embedding_hit = score >= threshold
    keyword_hit = bool(hits)
    keyword_score_hit = keyword_hit and score >= keyword_score_threshold

    if match_rule == "keyword-only":
        matched = keyword_score_hit
    elif match_rule == "embedding-only":
        matched = embedding_hit
    elif match_rule == "keyword-and-embedding":
        matched = keyword_hit and embedding_hit
    else:
        matched = keyword_score_hit or embedding_hit

    if not matched:
        return False, ""
    if embedding_hit and keyword_hit:
        return True, "both"
    if keyword_score_hit:
        return True, "keyword"
    return True, "embedding"


def discover_tasks(start_year: int, end_year: int, limit: int | None):
    tasks = []
    for year in range(start_year, end_year + 1):
        text_dir = TEXT_ROOT / str(year) / "文本"
        if not text_dir.exists():
            logging.warning("Missing text directory: %s", text_dir)
            continue
        for path in sorted(text_dir.glob(f"*_{year}-12-31.txt")):
            match = ANNUAL_RE.match(path.name)
            if not match:
                continue
            tasks.append(
                Task(
                    stock_id=match.group("stock_id"),
                    year=int(match.group("year")),
                    report_date=match.group("report_date"),
                    path=path,
                )
            )
            if limit is not None and len(tasks) >= limit:
                logging.debug("Discovery stopped at limit=%s", limit)
                return tasks
    logging.debug(
        "Discovered %s annual report tasks from %s to %s",
        len(tasks),
        start_year,
        end_year,
    )
    return tasks


def initialize_tasks(conn: sqlite3.Connection, tasks: list[Task], force: bool):
    if force:
        logging.info("Force mode: clearing previous state tables.")
        conn.execute("DELETE FROM sentence_matches")
        conn.execute("DELETE FROM report_results")
        conn.execute("DELETE FROM tasks")
        conn.commit()

    now = utcnow()
    rows = [
        (task.stock_id, task.year, task.report_date, str(task.path), "pending", now)
        for task in tasks
    ]
    conn.executemany(
        """
        INSERT OR IGNORE INTO tasks
            (stock_id, year, report_date, file_path, status, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.execute("UPDATE tasks SET status='pending', updated_at=? WHERE status='running'", (now,))
    conn.commit()
    logging.debug("Initialized/kept %s tasks in SQLite state.", len(tasks))


def select_tasks(
    conn: sqlite3.Connection,
    tasks: list[Task],
    resume: bool,
    retry_failed: bool,
):
    task_by_path = {str(task.path): task for task in tasks}
    if retry_failed:
        rows = conn.execute(
            "SELECT file_path FROM tasks WHERE status='failed' ORDER BY year, stock_id"
        ).fetchall()
    elif resume:
        rows = conn.execute(
            "SELECT file_path FROM tasks WHERE status!='done' ORDER BY year, stock_id"
        ).fetchall()
    else:
        rows = [(str(task.path),) for task in tasks]
        conn.executemany(
            "UPDATE tasks SET status='pending', error=NULL, updated_at=? WHERE file_path=? AND status!='done'",
            [(utcnow(), str(task.path)) for task in tasks],
        )
        conn.commit()
    return [task_by_path[row[0]] for row in rows if row[0] in task_by_path]


def mark_status(conn: sqlite3.Connection, task: Task, status: str, error: str | None = None):
    conn.execute(
        """
        UPDATE tasks
        SET status=?, attempts=attempts + CASE WHEN ?='running' THEN 1 ELSE 0 END,
            error=?, updated_at=?
        WHERE file_path=?
        """,
        (status, status, error, utcnow(), str(task.path)),
    )
    conn.commit()


def clean_text(text: str):
    return re.sub(r"\s+", "", text)


def split_sentences(text: str):
    segmenter = pysbd.Segmenter(language="zh", clean=False)
    primary = segmenter.segment(text)
    sentences = []
    for sent in primary:
        for part in re.findall(r"[^；;]+[；;]?", sent):
            part = clean_text(part).strip()
            if len(part) >= MIN_SENTENCE_CHARS:
                sentences.append(part)
    return sentences


def prepare_report(task: Task):
    start = time.perf_counter()
    text = task.path.read_text(encoding="utf-8-sig", errors="ignore")
    total_chars = len(clean_text(text))
    sentences = split_sentences(text)
    logging.debug(
        "Prepared %s %s chars=%s sentences=%s elapsed=%.3fs",
        task.stock_id,
        task.year,
        total_chars,
        len(sentences),
        time.perf_counter() - start,
    )
    return PreparedReport(
        task=task,
        total_chars=total_chars,
        total_sentences=len(sentences),
        sentences=sentences,
    )


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def build_model(model_name: str, device: str):
    use_fp16 = device.startswith("cuda")
    logging.info("Loading model=%s device=%s fp16=%s", model_name, device, use_fp16)
    return FlagModel(model_name, use_fp16=use_fp16, devices=[device])


def process_prepared(
    prepared: PreparedReport,
    model,
    theme_emb: np.ndarray,
    themes: list[dict],
    threshold: float,
    keyword_score_threshold: float,
    keywords: list[str],
    keyword_match_mode: str,
    keyword_tokenizer: str,
    match_rule: str,
    batch_size: int,
    save_matches: str,
    sample_per_report: int,
):
    task = prepared.task
    start = time.perf_counter()

    if not prepared.sentences:
        report_row = {
            "stock_id": task.stock_id,
            "year": task.year,
            "report_date": task.report_date,
            "file_path": str(task.path),
            "threshold": threshold,
            "total_chars": prepared.total_chars,
            "total_sentences": 0,
            "digital_chars": 0,
            "digital_sentences": 0,
            "digital_char_ratio": 0.0,
            "digital_sent_ratio": 0.0,
            "max_score": 0.0,
            "avg_digital_score": 0.0,
            "top_theme_id": "",
            "top_theme_name": "",
        }
        return report_row, [], time.perf_counter() - start

    sent_emb = normalize(model.encode(prepared.sentences, batch_size=batch_size))
    sims = sent_emb @ theme_emb.T
    best_theme_idx = sims.argmax(axis=1)
    best_scores = sims.max(axis=1)
    sentence_keyword_hits = [
        keyword_hits(sentence, keywords, keyword_match_mode, keyword_tokenizer)
        for sentence in prepared.sentences
    ]
    classifications = [
        classify_match(float(score), threshold, keyword_score_threshold, hits, match_rule)
        for score, hits in zip(best_scores, sentence_keyword_hits)
    ]
    is_digital = np.array([matched for matched, _ in classifications], dtype=bool)
    match_methods = [method for _, method in classifications]

    sentence_chars = np.array([len(sentence) for sentence in prepared.sentences], dtype=np.int64)
    digital_chars = int(sentence_chars[is_digital].sum())
    digital_sentences = int(is_digital.sum())
    top_idx = int(best_scores.argmax())
    top_theme = themes[int(best_theme_idx[top_idx])]
    digital_scores = best_scores[is_digital]

    report_row = {
        "stock_id": task.stock_id,
        "year": task.year,
        "report_date": task.report_date,
        "file_path": str(task.path),
        "threshold": threshold,
        "total_chars": prepared.total_chars,
        "total_sentences": prepared.total_sentences,
        "digital_chars": digital_chars,
        "digital_sentences": digital_sentences,
        "digital_char_ratio": digital_chars / prepared.total_chars if prepared.total_chars else 0.0,
        "digital_sent_ratio": digital_sentences / prepared.total_sentences
        if prepared.total_sentences
        else 0.0,
        "max_score": float(best_scores[top_idx]),
        "avg_digital_score": float(digital_scores.mean()) if digital_sentences else 0.0,
        "top_theme_id": top_theme["id"],
        "top_theme_name": top_theme["name"],
    }

    match_rows = []
    if save_matches != "none":
        indices = list(range(len(prepared.sentences)))
        if save_matches == "sample":
            digital_indices = [idx for idx in indices if is_digital[idx]]
            digital_indices.sort(key=lambda idx: (-float(best_scores[idx]), idx))
            indices = digital_indices[:sample_per_report]

        for idx in indices:
            theme = themes[int(best_theme_idx[idx])]
            match_rows.append(
                {
                    "file_path": str(task.path),
                    "stock_id": task.stock_id,
                    "year": task.year,
                    "sentence_id": idx + 1,
                    "sentence": prepared.sentences[idx],
                    "char_count": int(sentence_chars[idx]),
                    "max_score": float(best_scores[idx]),
                    "theme_id": theme["id"],
                    "theme_name": theme["name"],
                    "is_digital": int(bool(is_digital[idx])),
                    "matched_keywords": "、".join(sentence_keyword_hits[idx]),
                    "match_method": match_methods[idx],
                }
            )

    return report_row, match_rows, time.perf_counter() - start


def build_report_outputs(
    prepared: PreparedReport,
    sent_emb: np.ndarray | None,
    theme_emb: np.ndarray,
    themes: list[dict],
    threshold: float,
    keyword_score_threshold: float,
    keywords: list[str],
    keyword_match_mode: str,
    keyword_tokenizer: str,
    match_rule: str,
    save_matches: str,
    sample_per_report: int,
):
    task = prepared.task

    if not prepared.sentences:
        report_row = {
            "stock_id": task.stock_id,
            "year": task.year,
            "report_date": task.report_date,
            "file_path": str(task.path),
            "threshold": threshold,
            "total_chars": prepared.total_chars,
            "total_sentences": 0,
            "digital_chars": 0,
            "digital_sentences": 0,
            "digital_char_ratio": 0.0,
            "digital_sent_ratio": 0.0,
            "max_score": 0.0,
            "avg_digital_score": 0.0,
            "top_theme_id": "",
            "top_theme_name": "",
        }
        return report_row, []

    if sent_emb is None:
        best_theme_idx = np.full(len(prepared.sentences), -1, dtype=np.int64)
        best_scores = np.zeros(len(prepared.sentences), dtype=np.float32)
    else:
        sims = sent_emb @ theme_emb.T
        best_theme_idx = sims.argmax(axis=1)
        best_scores = sims.max(axis=1)
    sentence_keyword_hits = [
        keyword_hits(sentence, keywords, keyword_match_mode, keyword_tokenizer)
        for sentence in prepared.sentences
    ]
    classifications = [
        classify_match(float(score), threshold, keyword_score_threshold, hits, match_rule)
        for score, hits in zip(best_scores, sentence_keyword_hits)
    ]
    is_digital = np.array([matched for matched, _ in classifications], dtype=bool)
    match_methods = [method for _, method in classifications]

    sentence_chars = np.array([len(sentence) for sentence in prepared.sentences], dtype=np.int64)
    digital_chars = int(sentence_chars[is_digital].sum())
    digital_sentences = int(is_digital.sum())
    top_idx = int(best_scores.argmax())
    top_theme = (
        themes[int(best_theme_idx[top_idx])]
        if int(best_theme_idx[top_idx]) >= 0
        else {"id": "", "name": ""}
    )
    digital_scores = best_scores[is_digital]

    report_row = {
        "stock_id": task.stock_id,
        "year": task.year,
        "report_date": task.report_date,
        "file_path": str(task.path),
        "threshold": threshold,
        "total_chars": prepared.total_chars,
        "total_sentences": prepared.total_sentences,
        "digital_chars": digital_chars,
        "digital_sentences": digital_sentences,
        "digital_char_ratio": digital_chars / prepared.total_chars if prepared.total_chars else 0.0,
        "digital_sent_ratio": digital_sentences / prepared.total_sentences
        if prepared.total_sentences
        else 0.0,
        "max_score": float(best_scores[top_idx]),
        "avg_digital_score": float(digital_scores.mean()) if digital_sentences else 0.0,
        "top_theme_id": top_theme["id"],
        "top_theme_name": top_theme["name"],
    }

    match_rows = []
    if save_matches != "none":
        indices = list(range(len(prepared.sentences)))
        if save_matches == "sample":
            digital_indices = [idx for idx in indices if is_digital[idx]]
            digital_indices.sort(key=lambda idx: (-float(best_scores[idx]), idx))
            indices = digital_indices[:sample_per_report]

        for idx in indices:
            theme = (
                themes[int(best_theme_idx[idx])]
                if int(best_theme_idx[idx]) >= 0
                else {"id": "", "name": ""}
            )
            match_rows.append(
                {
                    "file_path": str(task.path),
                    "stock_id": task.stock_id,
                    "year": task.year,
                    "sentence_id": idx + 1,
                    "sentence": prepared.sentences[idx],
                    "char_count": int(sentence_chars[idx]),
                    "max_score": float(best_scores[idx]),
                    "theme_id": theme["id"],
                    "theme_name": theme["name"],
                    "is_digital": int(bool(is_digital[idx])),
                    "matched_keywords": "、".join(sentence_keyword_hits[idx]),
                    "match_method": match_methods[idx],
                }
            )

    return report_row, match_rows


def process_prepared_batch(
    prepared_reports: list[PreparedReport],
    model,
    theme_emb: np.ndarray,
    themes: list[dict],
    threshold: float,
    keyword_score_threshold: float,
    keywords: list[str],
    keyword_match_mode: str,
    keyword_tokenizer: str,
    match_rule: str,
    batch_size: int,
    save_matches: str,
    sample_per_report: int,
):
    start = time.perf_counter()
    nonempty = [prepared for prepared in prepared_reports if prepared.sentences]
    embeddings_by_path = {}

    if nonempty and model is not None:
        all_sentences = []
        spans = []
        offset = 0
        for prepared in nonempty:
            count = len(prepared.sentences)
            all_sentences.extend(prepared.sentences)
            spans.append((prepared, offset, offset + count))
            offset += count

        all_emb = normalize(model.encode(all_sentences, batch_size=batch_size))
        logging.debug(
            "Encoded GPU batch reports=%s sentences=%s batch_size=%s",
            len(nonempty),
            len(all_sentences),
            batch_size,
        )
        for prepared, start_idx, end_idx in spans:
            embeddings_by_path[str(prepared.task.path)] = all_emb[start_idx:end_idx]

    outputs = []
    for prepared in prepared_reports:
        sent_emb = embeddings_by_path.get(str(prepared.task.path))
        report_row, match_rows = build_report_outputs(
            prepared=prepared,
            sent_emb=sent_emb,
            theme_emb=theme_emb,
            themes=themes,
            threshold=threshold,
            keyword_score_threshold=keyword_score_threshold,
            keywords=keywords,
            keyword_match_mode=keyword_match_mode,
            keyword_tokenizer=keyword_tokenizer,
            match_rule=match_rule,
            save_matches=save_matches,
            sample_per_report=sample_per_report,
        )
        outputs.append((prepared, report_row, match_rows))

    return outputs, time.perf_counter() - start


def save_result(conn: sqlite3.Connection, report_row: dict, match_rows: list[dict]):
    conn.execute(
        """
        INSERT OR REPLACE INTO report_results (
            stock_id, year, report_date, file_path, threshold, total_chars,
            total_sentences, digital_chars, digital_sentences,
            digital_char_ratio, digital_sent_ratio, max_score,
            avg_digital_score, top_theme_id, top_theme_name
        )
        VALUES (
            :stock_id, :year, :report_date, :file_path, :threshold, :total_chars,
            :total_sentences, :digital_chars, :digital_sentences,
            :digital_char_ratio, :digital_sent_ratio, :max_score,
            :avg_digital_score, :top_theme_id, :top_theme_name
        )
        """,
        report_row,
    )
    conn.execute("DELETE FROM sentence_matches WHERE file_path=?", (report_row["file_path"],))
    if match_rows:
        conn.executemany(
            """
            INSERT OR REPLACE INTO sentence_matches (
                file_path, stock_id, year, sentence_id, sentence, char_count,
                max_score, theme_id, theme_name, is_digital,
                matched_keywords, match_method
            )
            VALUES (
                :file_path, :stock_id, :year, :sentence_id, :sentence, :char_count,
                :max_score, :theme_id, :theme_name, :is_digital,
                :matched_keywords, :match_method
            )
            """,
            match_rows,
        )
    conn.commit()


def export_csv(conn: sqlite3.Connection, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "digital_report_level_annual.csv"
    match_path = output_dir / "digital_sentence_matches_annual.csv"

    write_query_csv(
        conn,
        report_path,
        REPORT_FIELDS,
        f"SELECT {', '.join(REPORT_FIELDS)} FROM report_results ORDER BY year, stock_id, file_path",
    )
    write_query_csv(
        conn,
        match_path,
        MATCH_FIELDS,
        """
        SELECT stock_id, year, sentence_id, sentence, char_count, max_score,
               theme_id, theme_name, is_digital, matched_keywords, match_method
        FROM sentence_matches
        ORDER BY year, stock_id, file_path, sentence_id
        """,
    )
    logging.info("Exported %s", report_path)
    logging.info("Exported %s", match_path)


def write_query_csv(conn: sqlite3.Connection, path: Path, fields: list[str], query: str):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    rows = conn.execute(query)
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        writer.writerows(rows)
    tmp_path.replace(path)


def count_status(conn: sqlite3.Connection):
    rows = conn.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
    return {status: count for status, count in rows}


def run_pipeline(args):
    if args.keyword_tokenizer == "jieba" and jieba is None and not args.disable_keyword_match:
        raise RuntimeError("Install jieba or use --keyword-tokenizer substring.")

    output_dir = args.output_dir
    db_path = output_dir / "pipeline_state.sqlite"
    conn = connect_state(db_path)

    if args.export_only:
        export_csv(conn, output_dir)
        return

    uses_embedding = args.match_rule != "keyword-only" or args.keyword_score_threshold > 0
    exclude_theme_ids = parse_theme_ids(args.exclude_theme_ids)
    themes = load_themes(exclude_theme_ids) if uses_embedding else []
    keywords = load_keywords(args.keyword_path, args.disable_keyword_match)
    register_jieba_keywords(keywords, args.keyword_tokenizer)
    tasks = discover_tasks(args.start_year, args.end_year, args.limit)
    logging.info("Discovered annual reports: %s", len(tasks))
    initialize_tasks(conn, tasks, args.force)
    run_tasks = select_tasks(conn, tasks, args.resume, args.retry_failed)
    logging.info("Tasks selected for this run: %s", len(run_tasks))
    logging.debug("Initial SQLite status: %s", count_status(conn))
    if not run_tasks:
        export_csv(conn, output_dir)
        return

    if uses_embedding:
        model = build_model(args.model_name, args.device)
        theme_emb = normalize(
            model.encode([theme["vector_text"] for theme in themes], batch_size=len(themes))
        )
    else:
        logging.info(
            "Keyword-only mode with keyword_score_threshold<=0: skipping model load and embedding inference."
        )
        model = None
        theme_emb = None

    completed = 0
    failed = 0
    started = time.perf_counter()
    next_task_idx = 0
    futures = {}
    prepared_buffer = []
    prepared_sentence_count = 0

    def flush_prepared_buffer(force: bool = False):
        nonlocal completed, failed, prepared_buffer, prepared_sentence_count
        if not prepared_buffer:
            return
        if (
            not force
            and len(prepared_buffer) < max(1, args.gpu_report_batch_size)
            and prepared_sentence_count < max(1, args.max_gpu_sentences)
        ):
            logging.debug(
                "Holding GPU buffer reports=%s/%s sentences=%s/%s",
                len(prepared_buffer),
                args.gpu_report_batch_size,
                prepared_sentence_count,
                args.max_gpu_sentences,
            )
            return

        batch = prepared_buffer
        batch_sentence_count = prepared_sentence_count
        prepared_buffer = []
        prepared_sentence_count = 0

        try:
            logging.debug(
                "Flushing GPU buffer force=%s reports=%s sentences=%s",
                force,
                len(batch),
                batch_sentence_count,
            )
            outputs, batch_elapsed = process_prepared_batch(
                prepared_reports=batch,
                model=model,
                theme_emb=theme_emb,
                themes=themes,
                threshold=args.threshold,
                keyword_score_threshold=args.keyword_score_threshold,
                keywords=keywords,
                keyword_match_mode=args.keyword_match_mode,
                keyword_tokenizer=args.keyword_tokenizer,
                match_rule=args.match_rule,
                batch_size=args.batch_size,
                save_matches=args.save_matches,
                sample_per_report=args.sample_per_report,
            )
            logging.debug(
                "GPU_BATCH reports=%s sentences=%s elapsed=%.2fs avg_report=%.2fs",
                len(batch),
                batch_sentence_count,
                batch_elapsed,
                batch_elapsed / len(batch) if batch else 0.0,
            )
            for prepared, report_row, match_rows in outputs:
                task = prepared.task
                try:
                    save_result(conn, report_row, match_rows)
                    mark_status(conn, task, "done")
                    completed += 1
                    logging.debug(
                        "DONE %s %s sentences=%s digital=%s max=%.4f",
                        task.stock_id,
                        task.year,
                        report_row["total_sentences"],
                        report_row["digital_sentences"],
                        report_row["max_score"],
                    )
                except Exception:
                    failed += 1
                    err = traceback.format_exc()
                    mark_status(conn, task, "failed", err)
                    logging.error(
                        "FAILED_SAVE %s %s %s\n%s",
                        task.stock_id,
                        task.year,
                        task.path,
                        err,
                    )
        except Exception:
            err = traceback.format_exc()
            for prepared in batch:
                task = prepared.task
                failed += 1
                mark_status(conn, task, "failed", err)
                logging.error(
                    "FAILED_GPU_BATCH %s %s %s\n%s",
                    task.stock_id,
                    task.year,
                    task.path,
                    err,
                )

        processed = completed + failed
        if processed and processed % max(1, args.progress_every) == 0:
            elapsed_total = time.perf_counter() - started
            avg = elapsed_total / processed if processed else 0.0
            remaining = len(run_tasks) - processed
            logging.info(
                "PROGRESS processed=%s/%s completed=%s failed=%s avg=%.2fs eta=%.1fmin status=%s",
                processed,
                len(run_tasks),
                completed,
                failed,
                avg,
                (avg * remaining) / 60 if avg else 0.0,
                count_status(conn),
            )

    def submit_next(executor):
        nonlocal next_task_idx
        if next_task_idx >= len(run_tasks):
            return
        task = run_tasks[next_task_idx]
        next_task_idx += 1
        mark_status(conn, task, "running")
        logging.debug("START %s %s %s", task.stock_id, task.year, task.path)
        future = executor.submit(prepare_report, task)
        futures[future] = task

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        for _ in range(min(max(1, args.num_workers), len(run_tasks))):
            submit_next(executor)

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                task = futures.pop(future)
                submit_next(executor)
                try:
                    prepared = future.result()
                    logging.debug(
                        "PREPARED %s %s sentences=%s buffer_reports=%s buffer_sentences=%s",
                        task.stock_id,
                        task.year,
                        prepared.total_sentences,
                        len(prepared_buffer) + 1,
                        prepared_sentence_count + prepared.total_sentences,
                    )
                    prepared_buffer.append(prepared)
                    prepared_sentence_count += prepared.total_sentences
                    flush_prepared_buffer(force=False)
                except Exception:
                    failed += 1
                    err = traceback.format_exc()
                    mark_status(conn, task, "failed", err)
                    logging.error("FAILED %s %s %s\n%s", task.stock_id, task.year, task.path, err)

        flush_prepared_buffer(force=True)

    export_csv(conn, output_dir)
    logging.info("Final status: %s", count_status(conn))


def main():
    args = parse_args()
    setup_logging(args.log_dir, args.verbose)
    logging.info("Arguments: %s", vars(args))
    run_pipeline(args)


if __name__ == "__main__":
    main()
