
from __future__ import annotations

"""
DCG Stage 2 post-processing for detailed outputs.

This script performs Deliberative Complexity Gating (DCG) using only metadata
already present in Stage 1 detailed JSON. It makes zero API calls.
"""

import argparse
import csv
import json
import math
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
except (ModuleNotFoundError, ImportError) as exc:
    accuracy_score = None  # type: ignore[assignment]
    f1_score = None  # type: ignore[assignment]
    precision_recall_fscore_support = None  # type: ignore[assignment]
    _SKLEARN_IMPORT_ERROR = exc
else:
    _SKLEARN_IMPORT_ERROR = None


EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "General": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

CLARITY_LABELS = ["Clear Reply", "Ambivalent", "Clear Non-Reply"]
DEFAULT_GEMINI_WEIGHT = 4

EVAL_EVASION_MAP = {
    "Claims Ignorance": "Claims ignorance",
    "Partial": "Partial/half-answer",
    "Diffusion": "Deflection",
}


def normalize_evasion_label(x: Any) -> str:
    raw = str(x or "").strip()
    if not raw:
        return ""
    mapped = EVAL_EVASION_MAP.get(raw, raw)
    if mapped in EVASION_TO_CLARITY:
        return mapped

    low = raw.lower()
    for lab in EVASION_TO_CLARITY:
        if low == lab.lower():
            return lab
    for k, v in EVAL_EVASION_MAP.items():
        if low == k.lower() or low == str(v).lower():
            return v
    return mapped


def normalize_clarity(x: Any) -> str:
    low = str(x or "").strip().lower()
    if low.startswith("clear reply"):
        return "Clear Reply"
    if low.startswith("ambivalent"):
        return "Ambivalent"
    if low.startswith("direct reply"):
        return "Clear Reply"
    if low.startswith("direct non-reply") or low.startswith("direct non reply"):
        return "Clear Non-Reply"
    if low.startswith("clear non-reply") or low.startswith("clear non reply"):
        return "Clear Non-Reply"
    return str(x or "").strip()


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return default


def percentile_linear(values: List[float], percentile: float) -> float:
    """Linear interpolation percentile (same idea as numpy percentile default)."""
    if not values:
        raise ValueError("Cannot compute percentile on empty list.")
    if percentile <= 0:
        return min(values)
    if percentile >= 100:
        return max(values)
    vals = sorted(values)
    n = len(vals)
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


def majority_label(votes: Dict[str, int]) -> str:
    if not votes:
        return ""
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def parse_vote_counts(raw_votes: Any) -> Dict[str, int]:
    if not isinstance(raw_votes, dict):
        return {}
    out: Dict[str, int] = {}
    for label, count in raw_votes.items():
        c = safe_int(count, 0)
        if c > 0:
            out[str(label)] = c
    return out


def extract_gold_clarity(sample: Dict[str, Any]) -> str:
    gold = sample.get("gold", {}) or {}
    cl = normalize_clarity(gold.get("clarity_label", ""))
    return cl if cl in CLARITY_LABELS else ""


def extract_pre_predictions(sample: Dict[str, Any]) -> Tuple[str, str]:
    ens = sample.get("ensemble", {}) or {}
    ev = str(ens.get("final_evasion", "")).strip()
    if not ev:
        ev = str((sample.get("grok_round0", {}) or {}).get("majority_label", "")).strip() or "General"

    cl = normalize_clarity(ens.get("final_clarity", ""))
    if cl not in CLARITY_LABELS:
        cl = EVASION_TO_CLARITY.get(ev, "Ambivalent")
    return ev, cl


def mean_gemini_response_length(sample: Dict[str, Any]) -> Optional[float]:
    gem = sample.get("gemini_round0", {}) or {}
    responses = gem.get("responses", [])
    if not isinstance(responses, list) or not responses:
        return None
    lengths: List[int] = []
    for r in responses:
        if isinstance(r, dict):
            lengths.append(len(str(r.get("raw_response", "") or "")))
        else:
            lengths.append(len(str(r or "")))
    if not lengths:
        return None
    return float(sum(lengths) / len(lengths))


def validate_detailed_schema(samples: List[Dict[str, Any]]) -> None:
    if not isinstance(samples, list) or not samples:
        raise ValueError("Input JSON must be a non-empty list.")
    sample = samples[0]
    needed = {"grok_round0", "gemini_round0", "ensemble"}
    if not needed.issubset(set(sample.keys())):
        raise ValueError(
            "DCG Stage 2 supports only detailed schema (requires grok_round0/gemini_round0/ensemble)."
        )


def merge_input_csv(samples: List[Dict[str, Any]], csv_path: str) -> None:
    print(f"Merging CSV fields from: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != len(samples):
        raise ValueError(f"CSV rows ({len(rows)}) do not match JSON rows ({len(samples)}).")

    for s, r in zip(samples, rows):
        tf = s.get("text_features", {}) or {}
        if not str(tf.get("question", "")).strip():
            tf["question"] = str(r.get("question", "")).strip()
        if not str(tf.get("answer", "")).strip():
            tf["answer"] = str(r.get("interview_answer", "")).strip()
        s["text_features"] = tf

    with_text = sum(1 for s in samples if str((s.get("text_features", {}) or {}).get("question", "")).strip())
    print(f"Merged text fields for {with_text}/{len(samples)} samples.")


def _read_label_lines(path: str) -> List[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines:
        lines[0] = lines[0].lstrip("\ufeff")
    return lines


def merge_eval_gold_txt(
    samples: List[Dict[str, Any]],
    task1_labels_txt: Optional[str],
    task2_labels_txt: Optional[str],
) -> None:
    if not task1_labels_txt and not task2_labels_txt:
        return

    n = len(samples)
    task1_lines: Optional[List[str]] = None
    task2_lines: Optional[List[str]] = None
    if task1_labels_txt:
        task1_lines = _read_label_lines(task1_labels_txt)
        if len(task1_lines) != n:
            raise ValueError(
                f"task1 labels lines ({len(task1_lines)}) do not match JSON samples ({n})."
            )
    if task2_labels_txt:
        task2_lines = _read_label_lines(task2_labels_txt)
        if len(task2_lines) != n:
            raise ValueError(
                f"task2 labels lines ({len(task2_lines)}) do not match JSON samples ({n})."
            )

    for i, s in enumerate(samples):
        gold = s.get("gold", {}) or {}
        if task1_lines is not None:
            cl = normalize_clarity(task1_lines[i])
            gold["clarity_label"] = cl if cl in CLARITY_LABELS else ""
        if task2_lines is not None:
            ev_labels: List[str] = []
            for tok in task2_lines[i].split(","):
                lab = normalize_evasion_label(tok)
                if lab:
                    ev_labels.append(lab)
            gold["evasion_labels"] = ev_labels
        s["gold"] = gold

    if task1_lines is not None:
        with_cl = sum(1 for s in samples if extract_gold_clarity(s))
        print(f"Merged task1 clarity labels from TXT: {with_cl}/{n}")
    if task2_lines is not None:
        with_ev = sum(
            1
            for s in samples
            if isinstance((s.get("gold", {}) or {}).get("evasion_labels"), list)
            and len((s.get("gold", {}) or {}).get("evasion_labels")) > 0
        )
        print(f"Merged task2 evasion labels from TXT: {with_ev}/{n}")


def apply_dcg_sample(
    sample: Dict[str, Any],
    theta1: float,
    grok_threshold: float,
) -> Dict[str, Any]:
    pre_evasion, pre_clarity = extract_pre_predictions(sample)
    grok_round0 = sample.get("grok_round0", {}) or {}
    gemini_round0 = sample.get("gemini_round0", {}) or {}
    ensemble = sample.get("ensemble", {}) or {}

    avg_len = mean_gemini_response_length(sample)
    if avg_len is None:
        return {
            "pre_evasion": pre_evasion,
            "pre_clarity": pre_clarity,
            "final_evasion": pre_evasion,
            "final_clarity": pre_clarity,
            "decision_reason": str(ensemble.get("decision_reason", "SKIP_NO_GEMINI_RESPONSES")),
            "should_override": False,
            "avg_gemini_response_length": None,
            "grok_consistency": safe_float(grok_round0.get("consistency", 1.0), 1.0),
            "weighted_votes": {},
            "vote_margin": 0.0,
            "gemini_weight": safe_int(ensemble.get("gemini_weight", DEFAULT_GEMINI_WEIGHT), DEFAULT_GEMINI_WEIGHT),
            "status": "SKIP_NO_GEMINI_RESPONSES",
            "grok_majority_evasion": "",
            "gemini_majority_evasion": str(gemini_round0.get("majority_label", "")).strip(),
            "gemini_clarity_used": "",
        }

    vote_counts = parse_vote_counts(grok_round0.get("vote_counts", {}))
    if not vote_counts:
        return {
            "pre_evasion": pre_evasion,
            "pre_clarity": pre_clarity,
            "final_evasion": pre_evasion,
            "final_clarity": pre_clarity,
            "decision_reason": str(ensemble.get("decision_reason", "SKIP_NO_GROK_VOTES")),
            "should_override": False,
            "avg_gemini_response_length": avg_len,
            "grok_consistency": safe_float(grok_round0.get("consistency", 1.0), 1.0),
            "weighted_votes": {},
            "vote_margin": 0.0,
            "gemini_weight": safe_int(ensemble.get("gemini_weight", DEFAULT_GEMINI_WEIGHT), DEFAULT_GEMINI_WEIGHT),
            "status": "SKIP_NO_GROK_VOTES",
            "grok_majority_evasion": "",
            "gemini_majority_evasion": str(gemini_round0.get("majority_label", "")).strip(),
            "gemini_clarity_used": "",
        }

    grok_consistency = safe_float(grok_round0.get("consistency", 1.0), 1.0)
    grok_majority_evasion = majority_label(vote_counts)
    grok_clarity = EVASION_TO_CLARITY.get(grok_majority_evasion, "Ambivalent")

    gemini_majority_evasion = str(gemini_round0.get("majority_label", "")).strip()
    gemini_weight = safe_int(ensemble.get("gemini_weight", DEFAULT_GEMINI_WEIGHT), DEFAULT_GEMINI_WEIGHT)
    if gemini_weight <= 0:
        gemini_weight = DEFAULT_GEMINI_WEIGHT

    should_override = (avg_len > theta1) and (grok_consistency < grok_threshold)
    if should_override:
        gemini_clarity = "Ambivalent"
    else:
        gemini_clarity = EVASION_TO_CLARITY.get(gemini_majority_evasion, "Ambivalent")

    weighted_votes: Counter = Counter()
    for evasion_label, count in vote_counts.items():
        clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
        weighted_votes[clarity] += count
    weighted_votes[gemini_clarity] += gemini_weight

    if grok_clarity == gemini_clarity:
        final_clarity = grok_clarity
        decision_reason = "AGREE_DCG" if should_override else "AGREE"
    else:
        final_clarity = majority_label(dict(weighted_votes))
        if final_clarity == grok_clarity:
            decision_reason = "WEIGHTED_GROK_DCG" if should_override else "WEIGHTED_GROK"
        else:
            decision_reason = "WEIGHTED_DCG" if should_override else "WEIGHTED_GEMINI"

    if final_clarity == grok_clarity:
        final_evasion = grok_majority_evasion
    elif final_clarity == gemini_clarity and not should_override and gemini_majority_evasion:
        final_evasion = gemini_majority_evasion
    else:
        ambivalent_evasions = {
            k: v for k, v in vote_counts.items()
            if EVASION_TO_CLARITY.get(k, "Ambivalent") == "Ambivalent"
        }
        final_evasion = majority_label(ambivalent_evasions) if ambivalent_evasions else "General"

    sorted_votes = sorted(weighted_votes.items(), key=lambda kv: (-kv[1], kv[0]))
    top_count = sorted_votes[0][1] if sorted_votes else 0
    second_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    vote_margin = float(top_count - second_count)

    return {
        "pre_evasion": pre_evasion,
        "pre_clarity": pre_clarity,
        "final_evasion": final_evasion,
        "final_clarity": final_clarity,
        "decision_reason": decision_reason,
        "should_override": should_override,
        "avg_gemini_response_length": avg_len,
        "grok_consistency": grok_consistency,
        "weighted_votes": dict(weighted_votes),
        "vote_margin": vote_margin,
        "gemini_weight": gemini_weight,
        "status": "OK",
        "grok_majority_evasion": grok_majority_evasion,
        "gemini_majority_evasion": gemini_majority_evasion,
        "gemini_clarity_used": gemini_clarity,
    }


def compute_clarity_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if accuracy_score is None or f1_score is None or precision_recall_fscore_support is None:
        raise ModuleNotFoundError(
            "Missing dependency 'scikit-learn'. Install with `pip install scikit-learn`."
        ) from _SKLEARN_IMPORT_ERROR
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLARITY_LABELS, average="macro", zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLARITY_LABELS,
        zero_division=0,
    )
    per_class = {}
    for i, lab in enumerate(CLARITY_LABELS):
        per_class[lab] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class}


def print_summary(
    theta1: float,
    percentile: float,
    n_samples: int,
    override_samples: List[Dict[str, Any]],
    changed_count: int,
    has_gold: bool,
    metrics_error: Optional[str],
    metrics_pre: Optional[Dict[str, Any]],
    metrics_post: Optional[Dict[str, Any]],
    override_outcomes: Optional[Dict[str, int]],
) -> None:
    print("\n" + "=" * 72)
    print("DCG STAGE 2 SUMMARY")
    print("=" * 72)
    print(f"theta1 (Q{percentile:.1f}) = {theta1:.4f}")
    print(f"samples={n_samples} | overrides={len(override_samples)} | changed_predictions={changed_count}")

    if metrics_pre and metrics_post:
        print("\nPre vs Post (clarity)")
        print(f"  accuracy: {metrics_pre['accuracy']:.4f} -> {metrics_post['accuracy']:.4f}")
        print(f"  macro_f1: {metrics_pre['macro_f1']:.4f} -> {metrics_post['macro_f1']:.4f}")
        print("\nPer-class metrics (POST)")
        print("  class                 precision   recall      f1   support")
        for lab in CLARITY_LABELS:
            row = metrics_post["per_class"][lab]
            print(
                f"  {lab:<20} {row['precision']:>9.4f} {row['recall']:>8.4f} "
                f"{row['f1']:>8.4f} {row['support']:>8d}"
            )
        if override_outcomes is not None:
            print("\nOverride outcomes (gold-aware)")
            for k in ["fixed", "worsened", "still_correct", "still_wrong", "post_correct", "post_incorrect"]:
                print(f"  {k}: {override_outcomes.get(k, 0)}")
    elif has_gold and metrics_error:
        print(f"\nMetrics skipped: {metrics_error}")
    else:
        print("\nMetrics skipped: no gold clarity labels found.")

    print("\nExample overrides (first 3 where gate fired):")
    if not override_samples:
        print("  none")
    else:
        for ex in override_samples[:3]:
            print(
                "  "
                f"idx={ex['index']} pos={ex['position']} avg_len={ex['avg_gemini_response_length']:.2f} "
                f"grok_cons={ex['grok_consistency']:.3f} "
                f"{ex['pre_clarity']} -> {ex['post_clarity']} "
                f"changed={ex['changed']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="DCG Stage 2 post-processing for detailed outputs.")
    parser.add_argument("--input", required=True, help="Input Stage 1 detailed JSON.")
    parser.add_argument("--output", required=True, help="Output Stage 2 JSON.")
    parser.add_argument("--percentile", type=float, default=25.0, help="Gemini response-length percentile for theta1 (default: 25).")
    parser.add_argument("--grok-threshold", type=float, default=1.0, help="Grok consistency gate threshold (default: 1.0).")
    parser.add_argument("--input_csv", type=str, default=None, help="Optional CSV for filling text fields (question/answer).")
    parser.add_argument("--task1_labels_txt", type=str, default=None, help="Path to task1 eval labels TXT (clarity labels, one per line).")
    parser.add_argument("--task2_labels_txt", type=str, default=None, help="Path to task2 eval labels TXT (comma-separated evasion labels per line).")
    parser.add_argument("--metrics_output", type=str, default=None, help="Optional path for metrics JSON.")
    args = parser.parse_args()

    if args.percentile < 0 or args.percentile > 100:
        raise ValueError("--percentile must be in [0, 100].")

    in_path = Path(args.input)
    out_path = Path(args.output)
    raw_samples = json.loads(in_path.read_text(encoding="utf-8"))
    validate_detailed_schema(raw_samples)
    samples = deepcopy(raw_samples)

    if args.input_csv:
        merge_input_csv(samples, args.input_csv)
    if args.task1_labels_txt or args.task2_labels_txt:
        merge_eval_gold_txt(
            samples,
            task1_labels_txt=args.task1_labels_txt,
            task2_labels_txt=args.task2_labels_txt,
        )

    lengths: List[float] = []
    for s in samples:
        m = mean_gemini_response_length(s)
        if m is not None:
            lengths.append(m)
    if not lengths:
        print("WARNING: No valid gemini_round0.responses lengths found; DCG gate disabled (no overrides).")
        theta1 = float("inf")
    else:
        theta1 = percentile_linear(lengths, args.percentile)

    pre_clarity_all: List[str] = []
    post_clarity_all: List[str] = []
    gold_clarity_all: List[str] = []

    override_samples: List[Dict[str, Any]] = []
    changed_count = 0
    skipped_no_responses = 0
    skipped_no_grok_votes = 0

    for pos, sample in enumerate(samples):
        info = apply_dcg_sample(sample, theta1=theta1, grok_threshold=args.grok_threshold)

        pre_clarity_all.append(info["pre_clarity"])
        post_clarity_all.append(info["final_clarity"])
        gold_clarity_all.append(extract_gold_clarity(sample))

        if info["status"] == "SKIP_NO_GEMINI_RESPONSES":
            skipped_no_responses += 1
        elif info["status"] == "SKIP_NO_GROK_VOTES":
            skipped_no_grok_votes += 1

        if info["should_override"]:
            override_samples.append(
                {
                    "index": sample.get("index", pos),
                    "position": pos,
                    "avg_gemini_response_length": info["avg_gemini_response_length"],
                    "grok_consistency": info["grok_consistency"],
                    "pre_clarity": info["pre_clarity"],
                    "post_clarity": info["final_clarity"],
                    "pre_evasion": info["pre_evasion"],
                    "post_evasion": info["final_evasion"],
                    "changed": (info["pre_clarity"] != info["final_clarity"]) or (info["pre_evasion"] != info["final_evasion"]),
                }
            )

        if (info["pre_clarity"] != info["final_clarity"]) or (info["pre_evasion"] != info["final_evasion"]):
            changed_count += 1

        ensemble = sample.get("ensemble", {}) or {}
        ensemble["final_clarity"] = info["final_clarity"]
        ensemble["final_evasion"] = info["final_evasion"]
        ensemble["decision_reason"] = info["decision_reason"]
        ensemble["final_clarity_votes_weighted"] = info["weighted_votes"]
        ensemble["final_vote_margin"] = info["vote_margin"]
        ensemble["gemini_weight"] = info["gemini_weight"]
        sample["ensemble"] = ensemble

    valid_positions = [i for i, g in enumerate(gold_clarity_all) if g in CLARITY_LABELS]
    metrics_pre: Optional[Dict[str, Any]] = None
    metrics_post: Optional[Dict[str, Any]] = None
    override_outcomes: Optional[Dict[str, int]] = None
    metrics_error: Optional[str] = None

    if valid_positions:
        if _SKLEARN_IMPORT_ERROR is not None:
            metrics_error = "scikit-learn is not installed; metrics skipped."
            print(f"WARNING: {metrics_error}")
        else:
            y_true = [gold_clarity_all[i] for i in valid_positions]
            y_pre = [pre_clarity_all[i] for i in valid_positions]
            y_post = [post_clarity_all[i] for i in valid_positions]

            metrics_pre = compute_clarity_metrics(y_true, y_pre)
            metrics_post = compute_clarity_metrics(y_true, y_post)

            valid_set = set(valid_positions)
            override_outcomes = Counter()
            for ov in override_samples:
                pos = ov["position"]
                if pos not in valid_set:
                    continue
                gold = gold_clarity_all[pos]
                pre_ok = (ov["pre_clarity"] == gold)
                post_ok = (ov["post_clarity"] == gold)

                if (not pre_ok) and post_ok:
                    override_outcomes["fixed"] += 1
                elif pre_ok and (not post_ok):
                    override_outcomes["worsened"] += 1
                elif pre_ok and post_ok:
                    override_outcomes["still_correct"] += 1
                else:
                    override_outcomes["still_wrong"] += 1

                if post_ok:
                    override_outcomes["post_correct"] += 1
                else:
                    override_outcomes["post_incorrect"] += 1

    report = {
        "input": str(in_path),
        "output": str(out_path),
        "input_csv": args.input_csv,
        "task1_labels_txt": args.task1_labels_txt,
        "task2_labels_txt": args.task2_labels_txt,
        "gold_from_eval_txt": bool(args.task1_labels_txt or args.task2_labels_txt),
        "percentile": args.percentile,
        "grok_threshold": args.grok_threshold,
        "theta1": theta1,
        "counts": {
            "samples": len(samples),
            "overrides": len(override_samples),
            "changed_predictions": changed_count,
            "skipped_no_gemini_responses": skipped_no_responses,
            "skipped_no_grok_votes": skipped_no_grok_votes,
        },
        "metrics": {
            "has_gold": bool(valid_positions),
            "error": metrics_error,
            "pre": metrics_pre,
            "post": metrics_post,
            "override_outcomes": dict(override_outcomes) if override_outcomes is not None else None,
        },
        "override_examples_first3": override_samples[:3],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.metrics_output:
        metrics_path = Path(args.metrics_output)
    else:
        suffix = "".join(out_path.suffixes) if out_path.suffix else ".json"
        if suffix == ".json":
            metrics_path = out_path.with_name(out_path.stem + "_metrics.json")
        else:
            metrics_path = out_path.with_suffix("").with_name(out_path.stem + "_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(
        theta1=theta1,
        percentile=args.percentile,
        n_samples=len(samples),
        override_samples=override_samples,
        changed_count=changed_count,
        has_gold=bool(valid_positions),
        metrics_error=metrics_error,
        metrics_pre=metrics_pre,
        metrics_post=metrics_post,
        override_outcomes=dict(override_outcomes) if override_outcomes is not None else None,
    )
    print(f"\nSaved stage2 JSON: {out_path}")
    print(f"Saved metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
