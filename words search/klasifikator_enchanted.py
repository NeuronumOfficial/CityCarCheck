import argparse
import os
import re
import unicodedata
from typing import List, Dict, Tuple

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except Exception:
    HAS_RAPIDFUZZ = False

WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text

def tokenize(text: str) -> List[str]:
    n = normalize_text(text)
    return WORD_RE.findall(n)

def find_exact_word_positions(tokens: List[str], keyword: str) -> List[int]:
    return [i for i, t in enumerate(tokens) if t == keyword]

def fuzzy_token_hits(tokens: List[str], keyword: str, threshold: int) -> List[Tuple[int, str, float]]:
    if not HAS_RAPIDFUZZ:
        return []
    hits = []
    for i, tok in enumerate(tokens):
        s = fuzz.ratio(tok, keyword)
        if s >= threshold:
            hits.append((i, tok, float(s)))
    return hits

def score_matches_for_file(raw_text: str, keywords: List[str],
                           fuzzy_threshold: int = 80, use_fuzzy: bool = False) -> Tuple[int, Dict]:
    ntext = normalize_text(raw_text)
    tokens = tokenize(raw_text)
    details: Dict[str, Dict] = {"words": {}, "fuzzy": {}}
    score = 0

    for kw in keywords:
        nkw = normalize_text(kw)
        pos = find_exact_word_positions(tokens, nkw)
        if pos:
            details["words"][kw] = {"count": len(pos), "positions": pos}
            score += len(pos)

    if use_fuzzy and HAS_RAPIDFUZZ:
        for kw in keywords:
            nkw = normalize_text(kw)
            hits = fuzzy_token_hits(tokens, nkw, fuzzy_threshold)
            if hits:
                details["fuzzy"][kw] = {
                    "count": len(hits),
                    "token_hits": [{"pos": i, "token": t, "score": s} for (i, t, s) in hits]
                }
                score += len(hits)

    return score, details

def mark_file(path: str, approved: bool):
    folder, name = os.path.split(path)
    base, ext = os.path.splitext(name)
    if approved:
        new_name = f"{base}_Approved{ext}"
    else:
        new_name = f"{base}_Canceled{ext}"

    new_path = os.path.join(folder, new_name)
    if not os.path.exists(new_path):
        os.rename(path, new_path)
        print(f"File renamed -> {new_name}")
    else:
        print(f"File {new_name} already exists, skipping rename.")
    return new_path


def _iter_paths(dir_or_file: str, extensions: List[str]) -> List[str]:
    paths: List[str] = []
    if os.path.isfile(dir_or_file):
        if not extensions or any(dir_or_file.lower().endswith(ext) for ext in extensions):
            paths.append(dir_or_file)
        return paths

    for root, _, files in os.walk(dir_or_file):
        for fname in files:
            if extensions and not any(fname.lower().endswith(ext) for ext in extensions):
                continue
            paths.append(os.path.join(root, fname))
    return paths


def search_and_label(dirpath: str,
                     keywords: List[str],
                     min_score: int = 2,
                     extensions: List[str] = None,
                     use_fuzzy: bool = False,
                     fuzzy_threshold: int = 80):
    results = []
    if extensions is None:
        extensions = [".txt", ".md", ".text", ".csv", ".log"]

    for fp in _iter_paths(dirpath, extensions):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                raw = f.read()

        score, details = score_matches_for_file(
            raw,
            keywords=keywords,
            fuzzy_threshold=fuzzy_threshold,
            use_fuzzy=use_fuzzy,
        )

        approved = score >= min_score
        mark_file(fp, approved)

        results.append({
            "path": fp,
            "score": score,
            "approved": approved,
            "details": details
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def parse_comma_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]


def main():

    ap = argparse.ArgumentParser(description="Keyword-based document classifier with auto-labeling.")
    ap.add_argument("--dir", required=True, help="Directory or single file")
    ap.add_argument("--keywords", default="", help="Comma-separated keywords")
    ap.add_argument("--use-fuzzy", action="store_true", help="Enable fuzzy matching")
    ap.add_argument("--fuzzy-threshold", type=int, default=80)
    ap.add_argument("--min-score", type=int, default=2, help="Min hits for approval (default=2)")
    ap.add_argument("--extensions", default=".txt,.md,.log")
    args = ap.parse_args()

    keywords = parse_comma_list(args.keywords)
    exts = [e if e.startswith(".") else "." + e for e in parse_comma_list(args.extensions)]

    results = search_and_label(
        args.dir,
        keywords=keywords,
        min_score=args.min_score,
        extensions=exts,
        use_fuzzy=args.use_fuzzy,
        fuzzy_threshold=args.fuzzy_threshold,
    )

    print("\nSummary:")
    for r in results:
        label = "APPROVED" if r["approved"] else "CANCELED"
        print(f"{r['path']}  →  {label}  (score={r['score']})")


if __name__ == "__main__":
    main()
