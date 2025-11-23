#!/usr/bin/env python3
"""
generate_c3d.py

Usage examples:
  # full run, output dir annotated_large, 0.2 change prob, 0.5 contradiction threshold
  python generate_c3d.py --out annotated_large --change-prob 0.2 --contradiction-threshold 0.5 --num-files 0

  # run only 3 files per type
  python generate_c3d.py --out annotated_test --num-files 3 --change-prob 0.1

Notes:
 - Default log file: run_<out_dir>.txt (auto)
 - If --fresh is used, output dir and log are deleted before starting
 - Skips PDF if out_dir/<type>/<pdf_stem>/modified.txt and changes.json both exist
 - At end, parses the entire log file to compute totals across all runs in that log
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

from PyPDF2 import PdfReader
from shutil import copy2
from tqdm import tqdm

# transformers & torch for ContractNLI
try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

# -------------------------
# Helpers: Ollama caller
# -------------------------
def ollama_modify_clause(clause: str, model: str = "mistral", attempts: int = 3, timeout: int = 90) -> str:
    prompt = (
        "You are an expert contract drafter.\n"
        "Rewrite the following legal clause so that its meaning changes slightly — "
        "for example, weaken an obligation, invert a condition, or shift responsibility — "
        "but keep it fluent, realistic, and professional.\n\n"
        f"Clause:\n{clause}\n\nRewritten clause:"
    )
    for attempt in range(attempts):
        try:
            proc = subprocess.run(
                ["ollama", "run", model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,   # silence Gin / CUDA / Ollama spam
                check=True,
                timeout=timeout
            )
            out = proc.stdout.decode("utf-8").strip()
            if not out:
                raise ValueError("Empty output")
            if "Rewritten clause:" in out:
                out = out.split("Rewritten clause:", 1)[1].strip()
            return out
        except subprocess.TimeoutExpired:
            tqdm.write(f"[WARN] Ollama timeout (attempt {attempt+1})")
        except subprocess.CalledProcessError:
            tqdm.write(f"[WARN] Ollama process error (attempt {attempt+1})")
        except Exception as e:
            tqdm.write(f"[WARN] Ollama unexpected error (attempt {attempt+1}): {e}")
        time.sleep(1 + attempt)
    return clause + " (ollama failed)"

# -------------------------
# PDF clause extraction
# -------------------------
def normalize_text(t: str) -> str:
    return str(t).replace("\u00a0", " ").strip()

def extract_clauses_with_offsets(pdf_path: Path) -> List[Tuple[str,int,int]]:
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        tqdm.write(f"[WARN] Could not open PDF {pdf_path}: {e}")
        return []

    full_text = ""
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            full_text += str(txt) + "\n"
        except Exception as e:
            tqdm.write(f"[WARN] Failed to read a page from {pdf_path}: {e}")
    full_text = normalize_text(full_text)

    raw_lines = [normalize_text(line) for line in full_text.split("\n") if len(line.strip()) > 5]
    clauses, buf = [], []
    for line in raw_lines:
        buf.append(line)
        if line.endswith((".", ";", ":")):
            clauses.append(" ".join(buf).strip())
            buf = []
    if buf:
        clauses.append(" ".join(buf).strip())

    offsets = []
    cursor = 0
    for c in clauses:
        start = full_text.find(c, cursor)
        end = start + len(c)
        if start == -1:
            start = cursor
            end = cursor + len(c)
        offsets.append((c, start, end))
        cursor = end
    return offsets

# -------------------------
# ContractNLI verifier wrapper
# -------------------------
class ContractNLI:
    def __init__(self, model_name: str = "roberta-large-mnli", device: Optional[str] = None):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("transformers/torch not available in this environment.")
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tqdm.write(f"[INFO] Loading ContractNLI model '{model_name}' on device {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        inputs = self.tokenizer(premise, hypothesis, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        # try to find 'CONTRADICTION' label index
        label_map = {v.upper(): k for k, v in self.id2label.items()}
        for key in ("CONTRADICTION","CONTRADICT","CONTRADICTS"):
            if key in label_map:
                idx = int(label_map[key])
                return float(probs[idx])
        for name, idx in label_map.items():
            if "CONTRA" in name:
                return float(probs[int(idx)])
        # fallback to trying to find the label whose name contains 'contra' in id2label
        for idx, label in self.id2label.items():
            if "contra" in label.lower():
                return float(probs[int(idx)])
        return float(max(probs))

# -------------------------
# Log parsing (final totals from whole log)
# -------------------------
PER_FILE_RE = re.compile(
    r"^\[✓\]\s+(?P<path>.+?)\s+—\s+(?P<mod_kept>\d+)\s+modified_kept,\s+(?P<unchanged>\d+)\s+unchanged,\s+(?P<ollama_failed>\d+)\s+ollama_failed,\s+(?P<discarded>\d+)\s+discarded_low_contradiction",
    re.IGNORECASE
)

def parse_log_totals(log_path: Path):
    totals = {
        "total_files": 0,
        "total_modified_kept": 0,
        "total_modified_discarded": 0,
        "total_ollama_failed": 0,
        "total_unchanged": 0,
    }
    if not log_path.exists():
        return totals
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("[✓]"):
                continue
            m = PER_FILE_RE.match(line)
            if not m:
                continue
            totals["total_files"] += 1
            totals["total_modified_kept"] += int(m.group("mod_kept"))
            totals["total_unchanged"] += int(m.group("unchanged"))
            totals["total_ollama_failed"] += int(m.group("ollama_failed"))
            totals["total_modified_discarded"] += int(m.group("discarded"))
    return totals

# -------------------------
# Main generator
# -------------------------
def generate_c3d(base_cwd: Path,
                 out_dir: str,
                 num_files: int,
                 change_prob: float,
                 contradiction_threshold: float,
                 ollama_model: str,
                 nli_model: str,
                 seed: int,
                 fresh: bool,
                 log_path: Optional[Path]):
    random.seed(seed)
    # torch seed only if available
    if TRANSFORMERS_AVAILABLE:
        import torch
        torch.manual_seed(seed)

    base = base_cwd
    output_root = base / out_dir
    log_file = log_path or Path(f"run_{out_dir}.txt")

    # handle fresh
    if fresh:
        if output_root.exists():
            shutil.rmtree(output_root)
        if log_file.exists():
            log_file.unlink()
    output_root.mkdir(parents=True, exist_ok=True)

    # write RUN START block
    header_lines = [
        "===========================",
        f"RUN START: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "ARGS:",
        f"  change_prob = {change_prob}",
        f"  contradiction_threshold = {contradiction_threshold}",
        f"  seed = {seed}",
        f"  nli_model = {nli_model}",
        f"  ollama_model = {ollama_model}",
        f"  out_dir = {out_dir}",
        f"  fresh = {fresh}",
        "===========================",
        ""
    ]
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write("\n".join(header_lines) + "\n")

    # prepare ContractNLI if available
    nli = None
    if TRANSFORMERS_AVAILABLE:
        try:
            nli = ContractNLI(model_name=nli_model)
        except Exception as e:
            tqdm.write(f"[WARN] Failed to load NLI model '{nli_model}': {e}")
            nli = None

    # build unique contract type list across Part_* (first occurrence only)
    # seen_types = set()
    # part_folders = sorted(base.glob("Part*"))
    # if not part_folders:
    #     tqdm.write("[WARN] No Part_* folders found.")
    #     return
    # type_folders = []
    # for part_folder in part_folders:
    #     for tf in sorted(part_folder.iterdir()):
    #         if tf.is_dir() and tf.name not in seen_types:
    #             seen_types.add(tf.name)
    #             type_folders.append(tf)

    part_folders = sorted(base.glob("Part*"))
    type_folders = {}

    for part_folder in part_folders:
        for tf in sorted(part_folder.iterdir()):
            if tf.is_dir():
                type_folders.setdefault(tf.name, []).append(tf)


    # local counters for THIS run only (these will be appended to log lines per-file)
    # # but final totals will be computed by parsing the entire log (as requested)
    # for type_folder in tqdm(type_folders, desc="Contract Types", unit="type"):
    #     pdfs_all = sorted(type_folder.glob("*.pdf"))
    for type_name, folders in tqdm(type_folders.items(), desc="Contract Types", unit="type"):
        pdfs_all = []
        for folder in folders:
            pdfs_all.extend(sorted(folder.glob("*.pdf")))
            pdfs_all.extend(sorted(folder.glob("*.PDF")))

        if num_files and num_files > 0:
            pdfs = pdfs_all[:num_files]
        else:
            pdfs = pdfs_all

        if not pdfs:
            tqdm.write(f"[WARN] No PDFs for type {type_name}")
            continue

        for pdf_file in pdfs:
            # determine out folder per pdf: out_dir/type/pdf_stem/
            out_dir_for_pdf = output_root / type_name / pdf_file.stem
            # completion condition: both modified.txt and changes.json exist
            done_marker = (out_dir_for_pdf / "modified.txt").exists() and (out_dir_for_pdf / "changes.json").exists()
            if done_marker:
                # skip completed pdf
                continue

            # ensure folder exists (incomplete folders will be overwritten)
            if out_dir_for_pdf.exists():
                # wipe folder so we redo cleanly
                shutil.rmtree(out_dir_for_pdf)
            out_dir_for_pdf.mkdir(parents=True, exist_ok=True)

            clauses_with_offsets = extract_clauses_with_offsets(pdf_file)
            if not clauses_with_offsets:
                tqdm.write(f"[WARN] No clauses extracted from {pdf_file.name}")
                continue

            modified_clauses = []
            changelog = []

            success_count = 0
            fail_count = 0
            unchanged_count = 0
            discarded_count = 0

            inner_desc = f"{type_name}/{pdf_file.name}"
            for idx, (clause, start, end) in enumerate(
                tqdm(clauses_with_offsets, desc=inner_desc, leave=False, unit="clause")
            ):
                if random.random() < change_prob:
                    modified = ollama_modify_clause(clause, ollama_model)
                    if "(ollama failed)" in modified:
                        fail_count += 1
                        changelog.append({
                            "clause_index": idx,
                            "start_char": start,
                            "end_char": end,
                            "original": clause,
                            "modified": modified,
                            "contradiction_score": None,
                            "kept": False,
                            "reason": "ollama_failed"
                        })
                        modified_clauses.append(clause)
                    else:
                        # verify using NLI if available, else conservative: discard (score 0)
                        score = 0.0
                        if nli:
                            try:
                                score = nli.contradiction_score(clause, modified)
                            except Exception as e:
                                tqdm.write(f"[WARN] NLI error for clause idx {idx} in {pdf_file.name}: {e}")
                                score = 0.0
                        else:
                            # if no NLI model, we keep modified always (but user likely has NLI)
                            # however, earlier behavior was to discard when verification not present;
                            # but to avoid surprising behavior, we'll treat no-nli as score=1.0 (keep).
                            score = 1.0

                        if score >= contradiction_threshold:
                            success_count += 1
                            changelog.append({
                                "clause_index": idx,
                                "start_char": start,
                                "end_char": end,
                                "original": clause,
                                "modified": modified,
                                "contradiction_score": float(score),
                                "kept": True
                            })
                            modified_clauses.append(modified)
                        else:
                            discarded_count += 1
                            changelog.append({
                                "clause_index": idx,
                                "start_char": start,
                                "end_char": end,
                                "original": clause,
                                "ollama_generated": modified,
                                "contradiction_score": float(score),
                                "kept": False,
                                "reason": "low_contradiction"
                            })
                            modified_clauses.append(clause)
                else:
                    unchanged_count += 1
                    modified_clauses.append(clause)

            # copy original pdf
            try:
                copy2(pdf_file, out_dir_for_pdf / pdf_file.name)
            except Exception as e:
                tqdm.write(f"[WARN] Failed to copy PDF {pdf_file}: {e}")

            # write outputs
            (out_dir_for_pdf / "modified.txt").write_text("\n\n".join(modified_clauses), encoding="utf-8")
            (out_dir_for_pdf / "changes.json").write_text(json.dumps(changelog, indent=2), encoding="utf-8")

            # single-line per-file summary (append to log)
            line = (f"[✓] {type_name}/{pdf_file.name} — "
                    f"{success_count} modified_kept, {unchanged_count} unchanged, "
                    f"{fail_count} ollama_failed, {discarded_count} discarded_low_contradiction")
            with log_file.open("a", encoding="utf-8") as lf:
                lf.write(line + "\n")
            tqdm.write(line)

    # end of run: parse entire log to compute totals across all runs inside log file
    totals = parse_log_totals(log_file)
    final_lines = []
    final_lines.append("")
    final_lines.append("=== FINAL TOTALS (parsed from log) ===")
    final_lines.append(f"  total_files = {totals['total_files']}")
    final_lines.append(f"  total_modified_kept = {totals['total_modified_kept']}")
    final_lines.append(f"  total_modified_discarded = {totals['total_modified_discarded']}")
    final_lines.append(f"  total_ollama_failed = {totals['total_ollama_failed']}")
    final_lines.append(f"  total_unchanged = {totals['total_unchanged']}")
    final_lines.append("======================================")
    final_text = "\n".join(final_lines)

    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(final_text + "\n")

    print(final_text)
    tqdm.write(f"[INFO] Run log appended to {log_file}")

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="annotated", help="Output directory (root) for annotated files.")
    p.add_argument("--num-files", type=int, default=0, help="Number of files to pick per contract type (0 = all).")
    p.add_argument("--change-prob", type=float, default=0.1, help="Probability to call Ollama and modify each clause.")
    p.add_argument("--contradiction-threshold", type=float, default=0.6, help="Threshold (0-1) to accept a modified clause.")
    p.add_argument("--ollama-model", type=str, default="mistral", help="Ollama model name to use for clause rewriting.")
    p.add_argument("--nli-model", type=str, default="roberta-large-mnli", help="Hugging Face model name for contradiction detection.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--fresh", action="store_true", help="If set, nukes output dir and its run log before starting.")
    p.add_argument("--log", type=str, default="", help="Optional override path for run log. Default = run_<out>.txt")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    base = Path.cwd()
    log_path = Path(args.log) if args.log else None
    part_folders = sorted(base.glob("Part*"))
    total_pdfs = 0

    for part_folder in part_folders:
        for tf in sorted(part_folder.iterdir()):
            if tf.is_dir():
                total_pdfs += len(list(tf.glob("*.PDF")))
                total_pdfs += len(list(tf.glob("*.pdf")))

    print(f"Total PDFs found across Part folders = {total_pdfs}")

    try:
        generate_c3d(
            base_cwd=base,
            out_dir=args.out,
            num_files=args.num_files,
            change_prob=args.change_prob,
            contradiction_threshold=args.contradiction_threshold,
            ollama_model=args.ollama_model,
            nli_model=args.nli_model,
            seed=args.seed,
            fresh=args.fresh,
            log_path=log_path
        )
    except KeyboardInterrupt:
        tqdm.write("\n[WARN] Interrupted by user (KeyboardInterrupt). Partial results (if any) were saved; log was appended.")
        sys.exit(130)

