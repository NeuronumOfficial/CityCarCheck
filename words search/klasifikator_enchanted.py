import argparse
import os
import re
import unicodedata
import csv
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
                           fuzzy_threshold: int = 80, use_fuzzy: bool = False) -> int:
    tokens = tokenize(raw_text)
    score = 0

    for kw in keywords:
        nkw = normalize_text(kw)
        score += sum(1 for t in tokens if t == nkw)

        if use_fuzzy and HAS_RAPIDFUZZ:
            score += sum(1 for t in tokens if fuzz.ratio(t, nkw) >= fuzzy_threshold)

    return score

def mark_file(path: str, approved: bool):
    folder, name = os.path.split(path)
    base, ext = os.path.splitext(name)
    new_name = f"{base}_Approved{ext}" if approved else f"{base}_Canceled{ext}"
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
        extensions = [".txt", ".md", ".log"]

    for fp in _iter_paths(dirpath, extensions):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                raw = f.read()
        except UnicodeDecodeError:
            with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                raw = f.read()

        score = score_matches_for_file(
            raw,
            keywords=keywords,
            fuzzy_threshold=fuzzy_threshold,
            use_fuzzy=use_fuzzy,
        )

        approved = score >= min_score
        mark_file(fp, approved)

        results.append("ano" if approved else "nie")

    return results

def parse_comma_list(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[;,]", s)
    return [p.strip() for p in parts if p.strip()]

def save_results_csv(results: List[str], out_csv: str):
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["approved_sk"])
        for val in results:
            writer.writerow([val])
    print(f"\nCSV saved → {out_csv} (rows={len(results)})")

def print_console_table(results: List[str]):
    print("\nTabuľka výsledkov:")
    print("-" * 20)
    for i, val in enumerate(results, start=1):
        print(f"{i:>3}. {val}")
    print("-" * 20)

def main():
    ap = argparse.ArgumentParser(description="Jednoduchý klasifikátor s tabuľkou áno/nie.")
    ap.add_argument("--dir", required=True, help="Directory or single file")
    ap.add_argument("--keywords", default="", help="Comma-separated keywords")
    ap.add_argument("--use-fuzzy", action="store_true", help="Enable fuzzy matching")
    ap.add_argument("--fuzzy-threshold", type=int, default=80)
    ap.add_argument("--min-score", type=int, default=2, help="Min hits for approval (default=2)")
    ap.add_argument("--extensions", default=".txt,.md,.log")
    ap.add_argument("--out", default="results.csv", help="Path to CSV with results")
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

    print_console_table(results)
    save_results_csv(results, args.out)

if __name__ == "__main__":
    main()
