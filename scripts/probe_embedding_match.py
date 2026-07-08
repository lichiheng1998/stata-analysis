import json
import re
from pathlib import Path

import numpy as np
import pysbd
from FlagEmbedding import FlagModel


ROOT = Path(__file__).resolve().parents[1]
THEME_PATH = ROOT / "数字化主题向量列表.txt"
TEXT_ROOT = ROOT / "txt 2001-2025" / "管理层讨论与分析"
OUT_PATH = ROOT / "sample_embedding_matches.txt"

KEYWORDS = [
    "数字化转型",
    "数字化",
    "智能化",
    "人工智能",
    "大数据",
    "云计算",
    "工业互联网",
    "数据平台",
    "数据中心",
    "区块链",
    "物联网",
    "智能制造",
    "网络安全",
]


def load_themes():
    themes = []
    for line in THEME_PATH.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            themes.append(json.loads(line))
    return themes


def keyword_score(text):
    return sum(text.count(keyword) for keyword in KEYWORDS)


def pick_reports(limit=5):
    candidates = []
    for year in range(2020, 2026):
        text_dir = TEXT_ROOT / str(year) / "文本"
        if not text_dir.exists():
            continue
        for path in text_dir.glob("*.txt"):
            text = path.read_text(encoding="utf-8-sig", errors="ignore")
            score = keyword_score(text)
            if score:
                candidates.append((score, path))
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    return [path for _, path in candidates[:limit]]


def split_sentences(text):
    segmenter = pysbd.Segmenter(language="zh", clean=False)
    primary = segmenter.segment(text)
    sentences = []
    for sent in primary:
        for part in re.findall(r"[^；;]+[；;]?", sent):
            part = re.sub(r"\s+", "", part).strip()
            if len(part) >= 8:
                sentences.append(part)
    return sentences


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def main():
    themes = load_themes()
    reports = pick_reports(limit=5)

    model = FlagModel(
        "BAAI/bge-base-zh-v1.5",
        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
        use_fp16=True,
        devices=["cuda:0"],
    )

    theme_texts = [theme["vector_text"] for theme in themes]
    theme_emb = normalize(model.encode(theme_texts, batch_size=18))

    lines = []
    lines.append("BGE embedding match probe")
    lines.append(f"reports={len(reports)}")
    lines.append("")

    for report in reports:
        text = report.read_text(encoding="utf-8-sig", errors="ignore")
        sentences = split_sentences(text)
        if not sentences:
            continue

        sent_emb = normalize(model.encode(sentences, batch_size=64))
        sims = sent_emb @ theme_emb.T
        best_theme_idx = sims.argmax(axis=1)
        best_scores = sims.max(axis=1)

        rows = []
        for idx, sent in enumerate(sentences):
            hits = [keyword for keyword in KEYWORDS if keyword in sent]
            if hits or best_scores[idx] >= 0.62:
                theme = themes[int(best_theme_idx[idx])]
                rows.append(
                    {
                        "sentence_id": idx + 1,
                        "score": float(best_scores[idx]),
                        "theme": f'{theme["id"]} {theme["name"]}',
                        "hits": "、".join(hits) if hits else "-",
                        "sentence": sent,
                    }
                )

        rows.sort(key=lambda row: (-row["score"], row["sentence_id"]))
        lines.append("=" * 88)
        lines.append(str(report.relative_to(ROOT)))
        lines.append(f"total_sentences={len(sentences)} keyword_hits={keyword_score(text)} shown={min(12, len(rows))}")
        for row in rows[:12]:
            lines.append(
                f'[{row["sentence_id"]:04d}] score={row["score"]:.4f} theme={row["theme"]} keywords={row["hits"]}'
            )
            lines.append(row["sentence"])
            lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8-sig")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
