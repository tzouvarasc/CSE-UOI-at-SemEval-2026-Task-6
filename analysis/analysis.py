
"""
Sections A-B: Main Results Tables & Component Ablation

Reads pre-existing detailed JSONs (no API calls).
Eval set gold labels: from task1_eval_labels.txt ONLY.
Test set gold labels: from JSON gold field.
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Constants

CLARITY_LABELS = ["Ambivalent", "Clear Non-Reply", "Clear Reply"]

EVASION_TO_CLARITY = {
    "Explicit": "Clear Reply",
    "Implicit": "Ambivalent",
    "Dodging": "Ambivalent",
    "Deflection": "Ambivalent",
    "Partial/half-answer": "Ambivalent",
    "General": "Ambivalent",
    "Declining to answer": "Clear Non-Reply",
    "Claims ignorance": "Clear Non-Reply",
    "Clarification": "Clear Non-Reply",
}

BASE_DIR = Path(__file__).resolve().parent.parent  


# Helper: Metrics

def compute_prf(y_true: List[str], y_pred: List[str],
                labels: List[str] = CLARITY_LABELS) -> Dict[str, Any]:
    """Compute per-class P/R/F1 and macro F1 for given labels."""
    per_class = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1,
                          "tp": tp, "fp": fp, "fn": fn,
                          "support": sum(1 for g in y_true if g == lab)}
        f1s.append(f1)
    acc = sum(1 for g, p in zip(y_true, y_pred) if g == p) / len(y_true)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {"accuracy": acc, "macro_f1": macro_f1, "per_class": per_class,
            "n": len(y_true)}


def compute_evasion_macro_f1(y_true_multi: List[List[str]],
                              y_pred: List[str]) -> Dict[str, Any]:
    """Compute evasion macro F1 with multi-annotator gold (match any)."""
    all_labels = sorted(set(l for row in y_true_multi for l in row))
    f1s = []
    per_class = {}
    for lab in all_labels:
        tp = sum(1 for g, p in zip(y_true_multi, y_pred)
                 if p == lab and lab in g)
        fp = sum(1 for g, p in zip(y_true_multi, y_pred)
                 if p == lab and lab not in g)
        fn = sum(1 for g, p in zip(y_true_multi, y_pred)
                 if lab in g and p not in g)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1}
        f1s.append(f1)
    acc = sum(1 for g, p in zip(y_true_multi, y_pred) if p in g) / len(y_pred)
    return {"accuracy": acc, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "per_class": per_class}


# Helper: Data Loading

def load_eval_gold_from_txt() -> Tuple[List[str], List[List[str]]]:
    """Load eval gold labels from TXT files (authoritative source)."""
    # Clarity labels
    t1_path = BASE_DIR / "task1_eval_labels.txt"
    clarity = t1_path.read_text(encoding="utf-8").strip().split("\n")
    clarity[0] = clarity[0].lstrip("\ufeff")
    clarity = [c.strip() for c in clarity if c.strip()]

    # Evasion labels (multi-annotator, comma-separated)
    t2_path = BASE_DIR / "task2_eval_labels.txt"
    evasion_multi = []
    for line in t2_path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.lstrip("\ufeff").strip()
        if not line:
            continue
        labels = [l.strip() for l in line.split(",") if l.strip()]
        evasion_multi.append(labels)

    return clarity, evasion_multi


def load_detailed_json(filename: str) -> List[Dict[str, Any]]:
    """Load a detailed JSON file from paper_code directory."""
    path = BASE_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gold_clarity(samples: List[Dict], set_name: str) -> List[str]:
    """Get gold clarity labels for a set.
    Eval: from TXT files. Test: from JSON gold field."""
    if set_name == "eval":
        clarity, _ = load_eval_gold_from_txt()
        assert len(clarity) == len(samples), \
            f"TXT labels ({len(clarity)}) != samples ({len(samples)})"
        return clarity
    else:
        return [s.get("gold", {}).get("clarity_label", "") for s in samples]


def get_gold_evasion_multi(samples: List[Dict], set_name: str) -> List[List[str]]:
    """Get multi-annotator evasion gold labels."""
    if set_name == "eval":
        _, evasion = load_eval_gold_from_txt()
        assert len(evasion) == len(samples)
        return evasion
    else:
        return [s.get("gold", {}).get("evasion_labels", []) for s in samples]


# Helper: Tie-break (MUST match dcg_stage2.py exactly)

def majority_label(votes: Dict[str, int]) -> str:
    """Pick the label with highest count, breaking ties ALPHABETICALLY.
    EXACT match of dcg_stage2.py: sorted(items, key=(-count, label))[0][0]"""
    if not votes:
        return ""
    return sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


# Helper: Prediction Extraction

def get_ensemble_clarity(sample: Dict) -> str:
    """Get the ensemble's final clarity prediction (post-DCG if applied)."""
    return sample.get("ensemble", {}).get("final_clarity", "")


def get_ensemble_evasion(sample: Dict) -> str:
    """Get the ensemble's final evasion prediction."""
    return sample.get("ensemble", {}).get("final_evasion", "")


def get_pre_dcg_clarity(sample: Dict) -> str:
    """Reconstruct pre-DCG clarity by re-voting WITHOUT DCG override."""
    grok_r0 = sample.get("grok_round0", {})
    gemini_r0 = sample.get("gemini_round0", {})
    ens = sample.get("ensemble", {})

    # Grok votes: each evasion vote → clarity
    vote_counts = grok_r0.get("vote_counts", {})
    weighted_votes = Counter()
    for ev_label, count in vote_counts.items():
        cl = EVASION_TO_CLARITY.get(ev_label, "Ambivalent")
        weighted_votes[cl] += count

    # Gemini: use actual evasion majority → clarity (no DCG override)
    gemini_ev = gemini_r0.get("majority_label", "")
    gemini_cl = EVASION_TO_CLARITY.get(gemini_ev, "Ambivalent")
    gemini_weight = ens.get("gemini_weight", 4)
    weighted_votes[gemini_cl] += gemini_weight

    # Determine winner
    grok_cl = EVASION_TO_CLARITY.get(grok_r0.get("majority_label", ""), "Ambivalent")
    if grok_cl == gemini_cl:
        return grok_cl
    return majority_label(dict(weighted_votes))


def get_model_clarity(sample: Dict, model: str) -> str:
    """Get single-model clarity (Grok or Gemini)."""
    r0 = sample.get(f"{model}_round0", {})
    ev = r0.get("majority_label", "")
    return EVASION_TO_CLARITY.get(ev, "Ambivalent")


def get_model_evasion(sample: Dict, model: str) -> str:
    """Get single-model evasion majority label."""
    return sample.get(f"{model}_round0", {}).get("majority_label", "")


def subsample_k_clarity(sample: Dict, model: str, k: int) -> str:
    """Subsample first k responses from a model, get majority → clarity."""
    r0 = sample.get(f"{model}_round0", {})
    responses = r0.get("responses", [])
    if not responses or k <= 0:
        return get_model_clarity(sample, model)
    subset = responses[:k]
    labels = [r.get("label", "") for r in subset if r.get("label")]
    if not labels:
        return get_model_clarity(sample, model)
    majority = majority_label(dict(Counter(labels)))
    return EVASION_TO_CLARITY.get(majority, "Ambivalent")


def revote_ensemble_clarity(sample: Dict, gemini_weight: int) -> str:
    """Re-vote with a different GEMINI_WEIGHT."""
    grok_r0 = sample.get("grok_round0", {})
    gemini_r0 = sample.get("gemini_round0", {})

    vote_counts = grok_r0.get("vote_counts", {})
    weighted_votes = Counter()
    for ev_label, count in vote_counts.items():
        cl = EVASION_TO_CLARITY.get(ev_label, "Ambivalent")
        weighted_votes[cl] += count

    gemini_ev = gemini_r0.get("majority_label", "")
    gemini_cl = EVASION_TO_CLARITY.get(gemini_ev, "Ambivalent")
    weighted_votes[gemini_cl] += gemini_weight

    grok_cl = EVASION_TO_CLARITY.get(grok_r0.get("majority_label", ""), "Ambivalent")
    if grok_cl == gemini_cl:
        return grok_cl
    return majority_label(dict(weighted_votes))


def vote_evasion_majority(sample: Dict) -> str:
    """10-vote evasion majority (5 Grok + 5 Gemini individual responses)."""
    all_labels = []
    for model in ["grok_round0", "gemini_round0"]:
        for r in sample.get(model, {}).get("responses", []):
            lab = r.get("label", "")
            if lab:
                all_labels.append(lab)
    if not all_labels:
        return ""
    return majority_label(dict(Counter(all_labels)))


def vote_clarity_majority_10(sample: Dict) -> str:
    """10-vote clarity majority: map each response to clarity, then vote."""
    all_clarity = []
    for model in ["grok_round0", "gemini_round0"]:
        for r in sample.get(model, {}).get("responses", []):
            lab = r.get("label", "")
            if lab:
                all_clarity.append(EVASION_TO_CLARITY.get(lab, "Ambivalent"))
    if not all_clarity:
        return ""
    return majority_label(dict(Counter(all_clarity)))


# Helper: DCG re-application

def mean_gemini_response_length(sample: Dict) -> Optional[float]:
    """Compute average Gemini raw_response length (chars)."""
    gem = sample.get("gemini_round0", {})
    responses = gem.get("responses", [])
    if not responses:
        return None
    lengths = [len(str(r.get("raw_response", "") or "")) for r in responses]
    return sum(lengths) / len(lengths) if lengths else None


def percentile_q1(values: List[float]) -> float:
    """Compute 25th percentile (Q1) with linear interpolation."""
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    k = (n - 1) * 0.25
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(vals[int(k)])
    return float(vals[f] * (c - k) + vals[c] * (k - f))


def apply_dcg_batch(samples: List[Dict], percentile: float = 25.0,
                    grok_threshold: float = 1.0) -> List[str]:
    """Apply DCG to a batch, returning post-DCG clarity predictions."""
    # Compute theta1 from Gemini response lengths
    gem_lengths = []
    for s in samples:
        gl = mean_gemini_response_length(s)
        if gl is not None:
            gem_lengths.append(gl)

    if not gem_lengths:
        return [get_pre_dcg_clarity(s) for s in samples]

    # Compute percentile threshold
    vals = sorted(gem_lengths)
    n = len(vals)
    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        theta1 = float(vals[int(k)])
    else:
        theta1 = float(vals[f] * (c - k) + vals[c] * (k - f))

    results = []
    for s in samples:
        avg_len = mean_gemini_response_length(s)
        grok_cons = s.get("grok_round0", {}).get("consistency", 1.0) or 1.0

        if avg_len is None:
            results.append(get_pre_dcg_clarity(s))
            continue

        should_override = (avg_len > theta1) and (grok_cons < grok_threshold)

        grok_r0 = s.get("grok_round0", {})
        ens = s.get("ensemble", {})
        vote_counts = grok_r0.get("vote_counts", {})
        gemini_r0 = s.get("gemini_round0", {})

        weighted_votes = Counter()
        for ev_label, count in vote_counts.items():
            cl = EVASION_TO_CLARITY.get(ev_label, "Ambivalent")
            weighted_votes[cl] += count

        gemini_ev = gemini_r0.get("majority_label", "")
        gemini_weight = ens.get("gemini_weight", 4)
        if should_override:
            gemini_cl = "Ambivalent"
        else:
            gemini_cl = EVASION_TO_CLARITY.get(gemini_ev, "Ambivalent")

        weighted_votes[gemini_cl] += gemini_weight

        grok_cl = EVASION_TO_CLARITY.get(
            grok_r0.get("majority_label", ""), "Ambivalent")
        if grok_cl == gemini_cl:
            results.append(grok_cl)
        else:
            results.append(majority_label(dict(weighted_votes)))

    return results


# Helper: Formatting

def fmt_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


def fmt_f1(val: float) -> str:
    return f"{val:.4f}"


def print_results_table(title: str, metrics: Dict[str, Any]):
    """Pretty-print a results table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {fmt_pct(metrics['accuracy'])}  |  "
          f"Macro F1: {fmt_f1(metrics['macro_f1'])}  |  n={metrics['n']}")
    print(f"  {'─' * 60}")
    print(f"  {'Class':<20s} {'P':>8s} {'R':>8s} {'F1':>8s} {'Support':>8s}")
    print(f"  {'─' * 60}")
    for lab in CLARITY_LABELS:
        pc = metrics["per_class"].get(lab, {})
        print(f"  {lab:<20s} {pc.get('precision', 0):.4f}   "
              f"{pc.get('recall', 0):.4f}   {pc.get('f1', 0):.4f}   "
              f"{pc.get('support', 0):>5d}")
    print()


# SECTION A: Main Results Tables

def section_a(test_data, eval_data, test_gold, eval_gold):
    """Section A: Main results tables for both sets, pre/post DCG."""
    print("\n" + "=" * 70)
    print("  SECTION A: MAIN RESULTS TABLES")
    print("=" * 70)

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        # Post-DCG (final ensemble output — already has DCG applied)
        post_dcg_preds = [get_ensemble_clarity(s) for s in data]
        post_dcg_metrics = compute_prf(gold, post_dcg_preds)
        print_results_table(
            f"{set_name} SET — Post-DCG (Final System)", post_dcg_metrics)

        # Pre-DCG (reconstruct without DCG override)
        pre_dcg_preds = [get_pre_dcg_clarity(s) for s in data]
        pre_dcg_metrics = compute_prf(gold, pre_dcg_preds)
        print_results_table(
            f"{set_name} SET — Pre-DCG (Stage 0 Only)", pre_dcg_metrics)

        # DCG improvement
        delta = post_dcg_metrics["macro_f1"] - pre_dcg_metrics["macro_f1"]
        print(f"  DCG Improvement: {delta:+.4f} Macro F1 "
              f"({fmt_f1(pre_dcg_metrics['macro_f1'])} → "
              f"{fmt_f1(post_dcg_metrics['macro_f1'])})")

        # Evasion metrics (if gold available)
        gold_ev = get_gold_evasion_multi(data, set_name.lower())
        pred_ev = [get_ensemble_evasion(s) for s in data]
        if gold_ev and all(gold_ev):
            ev_metrics = compute_evasion_macro_f1(gold_ev, pred_ev)
            print(f"  Evasion: Acc={fmt_pct(ev_metrics['accuracy'])}  "
                  f"Macro F1={fmt_f1(ev_metrics['macro_f1'])}")

        results[set_name.lower()] = {
            "pre_dcg": pre_dcg_metrics,
            "post_dcg": post_dcg_metrics,
            "delta_f1": delta,
        }

    return results


# SECTION B: Component Ablation

def section_b(test_data, eval_data, test_gold, eval_gold):
    """Section B: Component ablation — run on BOTH sets."""
    print("\n" + "=" * 70)
    print("  SECTION B: COMPONENT ABLATION")
    print("=" * 70)

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        print(f"\n{'─' * 70}")
        print(f"  {set_name} SET (n={len(data)})")
        print(f"{'─' * 70}")

        ablation = {}

        # B1: Grok-only
        grok_preds = [get_model_clarity(s, "grok") for s in data]
        grok_m = compute_prf(gold, grok_preds)
        ablation["B1_grok_only"] = grok_m
        print(f"  B1 Grok-only:          F1={fmt_f1(grok_m['macro_f1'])}  "
              f"Acc={fmt_pct(grok_m['accuracy'])}")

        # B2: Gemini-only
        gem_preds = [get_model_clarity(s, "gemini") for s in data]
        gem_m = compute_prf(gold, gem_preds)
        ablation["B2_gemini_only"] = gem_m
        print(f"  B2 Gemini-only:        F1={fmt_f1(gem_m['macro_f1'])}  "
              f"Acc={fmt_pct(gem_m['accuracy'])}")

        # B3: Ensemble pre-DCG
        pre_preds = [get_pre_dcg_clarity(s) for s in data]
        pre_m = compute_prf(gold, pre_preds)
        ablation["B3_ensemble_pre_dcg"] = pre_m
        print(f"  B3 Ensemble (no DCG):  F1={fmt_f1(pre_m['macro_f1'])}  "
              f"Acc={fmt_pct(pre_m['accuracy'])}")

        # B4: Full system post-DCG
        post_preds = [get_ensemble_clarity(s) for s in data]
        post_m = compute_prf(gold, post_preds)
        ablation["B4_full_system_dcg"] = post_m
        print(f"  B4 Full system (+DCG): F1={fmt_f1(post_m['macro_f1'])}  "
              f"Acc={fmt_pct(post_m['accuracy'])}")

        # B5: k-sweep (subsample from k=5)
        print(f"\n  B5 k-sweep (self-consistency):")
        for k in [1, 3, 5]:
            # For each model, subsample k responses → get clarity majority
            # Then ensemble the two with standard weights
            k_preds = []
            for s in data:
                grok_cl = subsample_k_clarity(s, "grok", k)
                gem_cl = subsample_k_clarity(s, "gemini", k)
                if grok_cl == gem_cl:
                    k_preds.append(grok_cl)
                else:
                    # Weighted voting with subsampled Grok votes
                    grok_responses = s.get("grok_round0", {}).get("responses", [])[:k]
                    grok_labels = [r.get("label", "") for r in grok_responses]
                    wv = Counter()
                    for lab in grok_labels:
                        if lab:
                            wv[EVASION_TO_CLARITY.get(lab, "Ambivalent")] += 1
                    wv[gem_cl] += min(k - 1, 4)  # Scale gemini weight proportionally
                    k_preds.append(majority_label(dict(wv)) if wv else grok_cl)
            k_m = compute_prf(gold, k_preds)
            ablation[f"B5_k{k}"] = k_m
            print(f"    k={k}: F1={fmt_f1(k_m['macro_f1'])}  "
                  f"Acc={fmt_pct(k_m['accuracy'])}")

        # B6: GEMINI_WEIGHT sweep
        print(f"\n  B6 GEMINI_WEIGHT sweep:")
        for w in [0, 1, 2, 3, 4, 5, 6]:
            w_preds = [revote_ensemble_clarity(s, w) for s in data]
            w_m = compute_prf(gold, w_preds)
            ablation[f"B6_weight{w}"] = w_m
            marker = " ← current" if w == 4 else ""
            print(f"    w={w}: F1={fmt_f1(w_m['macro_f1'])}  "
                  f"Acc={fmt_pct(w_m['accuracy'])}{marker}")

        # B7: Clarity-first vs alternative voting
        print(f"\n  B7 Voting strategy comparison:")

        # Current: clarity-first (S0)
        print(f"    Clarity-first (S0):   F1={fmt_f1(pre_m['macro_f1'])}  "
              f"Acc={fmt_pct(pre_m['accuracy'])} ← current")

        # 10-vote evasion majority → clarity
        ev_maj_preds = [EVASION_TO_CLARITY.get(vote_evasion_majority(s), "Ambivalent")
                        for s in data]
        ev_maj_m = compute_prf(gold, ev_maj_preds)
        ablation["B7_evasion_majority"] = ev_maj_m
        print(f"    Evasion majority:     F1={fmt_f1(ev_maj_m['macro_f1'])}  "
              f"Acc={fmt_pct(ev_maj_m['accuracy'])}")

        # 10-vote clarity majority
        cl_maj_preds = [vote_clarity_majority_10(s) for s in data]
        cl_maj_m = compute_prf(gold, cl_maj_preds)
        ablation["B7_clarity_10vote"] = cl_maj_m
        print(f"    Clarity 10-vote:      F1={fmt_f1(cl_maj_m['macro_f1'])}  "
              f"Acc={fmt_pct(cl_maj_m['accuracy'])}")

        # Hidden agreements: evasion-disagree but clarity-agree
        n_ev_agree = sum(1 for s in data
                         if get_model_evasion(s, "grok") ==
                         get_model_evasion(s, "gemini"))
        n_ev_disagree_cl_agree = sum(
            1 for s in data
            if get_model_evasion(s, "grok") != get_model_evasion(s, "gemini")
            and get_model_clarity(s, "grok") == get_model_clarity(s, "gemini"))
        n_ev_disagree_cl_disagree = sum(
            1 for s in data
            if get_model_evasion(s, "grok") != get_model_evasion(s, "gemini")
            and get_model_clarity(s, "grok") != get_model_clarity(s, "gemini"))

        total = len(data)
        print(f"\n    Hidden agreements analysis:")
        print(f"      Evasion agree:                   "
              f"{n_ev_agree}/{total} ({100*n_ev_agree/total:.1f}%)")
        print(f"      Evasion disagree, clarity agree:  "
              f"{n_ev_disagree_cl_agree}/{total} "
              f"({100*n_ev_disagree_cl_agree/total:.1f}%) ← hidden")
        print(f"      Evasion disagree, clarity disagree:"
              f" {n_ev_disagree_cl_disagree}/{total} "
              f"({100*n_ev_disagree_cl_disagree/total:.1f}%)")

        # Accuracy on hidden agreements
        if n_ev_disagree_cl_agree > 0:
            hidden_correct = sum(
                1 for s, g in zip(data, gold)
                if get_model_evasion(s, "grok") != get_model_evasion(s, "gemini")
                and get_model_clarity(s, "grok") == get_model_clarity(s, "gemini")
                and get_model_clarity(s, "grok") == g)
            print(f"      Hidden agreement accuracy:        "
                  f"{hidden_correct}/{n_ev_disagree_cl_agree} "
                  f"({100*hidden_correct/n_ev_disagree_cl_agree:.1f}%)")

        ablation["B7_hidden_agreements"] = {
            "evasion_agree": n_ev_agree,
            "evasion_disagree_clarity_agree": n_ev_disagree_cl_agree,
            "evasion_disagree_clarity_disagree": n_ev_disagree_cl_disagree,
        }

        # B8 & B9: Debate and Hu variants (test set only, from metrics JSONs)
        if set_name == "TEST":
            print(f"\n  B8-B9 Ablation variants (test set only):")

            # B8: Debate
            debate_metrics_path = (BASE_DIR / "apex_debate_ablation" /
                                   "apex_debate_debate_round_caps_metrics.json")
            if debate_metrics_path.exists():
                dm = json.loads(debate_metrics_path.read_text(encoding="utf-8"))
                # Use cap=3 (full debate)
                cap3 = dm.get("caps", {}).get("3", {})
                cl = cap3.get("clarity", {})
                print(f"    B8 Debate (cap=3):    F1={fmt_f1(cl.get('macro_f1', 0))}  "
                      f"Acc={fmt_pct(cl.get('accuracy', 0))}")
                for cap_k in ["1", "2", "3"]:
                    cap_d = dm.get("caps", {}).get(cap_k, {})
                    cap_cl = cap_d.get("clarity", {})
                    print(f"      cap={cap_k}: F1={fmt_f1(cap_cl.get('macro_f1', 0))}  "
                          f"Acc={fmt_pct(cap_cl.get('accuracy', 0))}")
                ablation["B8_debate"] = cl

            # B9: Hu debate
            hu_metrics_path = (BASE_DIR / "Tianyu_Hu_Debate_ablation" /
                               "debate_llm_judges_metrics.json")
            if hu_metrics_path.exists():
                hm = json.loads(hu_metrics_path.read_text(encoding="utf-8"))
                cl_hu = hm.get("clarity", {})
                print(f"    B9 Hu 7-agent:       F1={fmt_f1(cl_hu.get('macro_f1', 0))}  "
                      f"Acc={fmt_pct(cl_hu.get('instance_accuracy', 0))}")
                ablation["B9_hu_debate"] = cl_hu

        results[set_name.lower()] = ablation

    return results


# SECTION C: S0 Findings (eval primary, test for dev context — run on BOTH)

def section_c(test_data, eval_data, test_gold, eval_gold):
    """Section C: 15 S0-level findings."""
    print("\n" + "=" * 70)
    print("  SECTION C: S0 FINDINGS")
    print("=" * 70)

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        print(f"\n{'─' * 70}")
        print(f"  {set_name} SET (n={len(data)})")
        print(f"{'─' * 70}")
        findings = {}

        # Precompute common data
        grok_cl = [get_model_clarity(s, "grok") for s in data]
        gem_cl = [get_model_clarity(s, "gemini") for s in data]
        grok_ev = [get_model_evasion(s, "grok") for s in data]
        gem_ev = [get_model_evasion(s, "gemini") for s in data]
        ens_cl = [get_pre_dcg_clarity(s) for s in data]

        # C1: Agreement = confidence proxy
        agree_indices = [i for i in range(len(data))
                         if grok_cl[i] == gem_cl[i]]
        disagree_indices = [i for i in range(len(data))
                            if grok_cl[i] != gem_cl[i]]
        agree_acc = (sum(1 for i in agree_indices if ens_cl[i] == gold[i])
                     / max(len(agree_indices), 1))
        disagree_acc = (sum(1 for i in disagree_indices if ens_cl[i] == gold[i])
                        / max(len(disagree_indices), 1))
        print(f"\n  C1: Agreement = confidence proxy")
        print(f"    Models agree:    {len(agree_indices)}/{len(data)} "
              f"({100*len(agree_indices)/len(data):.1f}%)  "
              f"accuracy={100*agree_acc:.1f}%")
        print(f"    Models disagree: {len(disagree_indices)}/{len(data)} "
              f"({100*len(disagree_indices)/len(data):.1f}%)  "
              f"accuracy={100*disagree_acc:.1f}%")
        findings["C1"] = {"agree_n": len(agree_indices),
                          "agree_acc": agree_acc,
                          "disagree_n": len(disagree_indices),
                          "disagree_acc": disagree_acc}

        # C2: Error boundary distribution
        errors = [(gold[i], ens_cl[i]) for i in range(len(data))
                  if gold[i] != ens_cl[i]]
        boundary_counts = Counter()
        for g, p in errors:
            pair = tuple(sorted([g, p]))
            boundary_counts[pair] += 1
        total_errors = len(errors)
        print(f"\n  C2: Error boundary distribution ({total_errors} errors)")
        for pair, count in boundary_counts.most_common():
            print(f"    {pair[0]} <-> {pair[1]}: "
                  f"{count}/{total_errors} ({100*count/max(total_errors,1):.1f}%)")
        findings["C2"] = {str(k): v for k, v in boundary_counts.items()}

        # C3: Complementary error profiles (per-evasion-class accuracy)
        gold_ev_multi = get_gold_evasion_multi(data, set_name.lower())
        print(f"\n  C3: Per-evasion-class accuracy (Grok vs Gemini)")
        evasion_classes = sorted(set(l for row in gold_ev_multi for l in row))
        c3_data = {}
        for ev_class in evasion_classes:
            indices = [i for i, g in enumerate(gold_ev_multi) if ev_class in g]
            if not indices:
                continue
            grok_correct = sum(1 for i in indices
                               if grok_ev[i] in gold_ev_multi[i])
            gem_correct = sum(1 for i in indices
                              if gem_ev[i] in gold_ev_multi[i])
            n = len(indices)
            grok_acc = grok_correct / n
            gem_acc = gem_correct / n
            diff = grok_acc - gem_acc
            winner = "Grok" if diff > 0.01 else ("Gemini" if diff < -0.01 else "Tied")
            print(f"    {ev_class:<22s}  Grok={100*grok_acc:.0f}%  "
                  f"Gem={100*gem_acc:.0f}%  diff={100*diff:+.0f}%  ({winner})")
            c3_data[ev_class] = {"grok_acc": grok_acc, "gem_acc": gem_acc,
                                 "n": n, "winner": winner}
        findings["C3"] = c3_data

        # C4: Opposite bias profiles (prediction distribution vs gold)
        gold_ev_flat = [g[0] if g else "" for g in gold_ev_multi]
        grok_dist = Counter(grok_ev)
        gem_dist = Counter(gem_ev)
        gold_dist = Counter(gold_ev_flat)
        print(f"\n  C4: Prediction distribution bias")
        print(f"    {'Label':<22s} {'Gold':>5s} {'Grok':>5s} {'dG':>5s} "
              f"{'Gem':>5s} {'dGem':>5s}")
        for lab in evasion_classes:
            g_count = gold_dist.get(lab, 0)
            gr_count = grok_dist.get(lab, 0)
            ge_count = gem_dist.get(lab, 0)
            print(f"    {lab:<22s} {g_count:>5d} {gr_count:>5d} "
                  f"{gr_count-g_count:>+5d} {ge_count:>5d} "
                  f"{ge_count-g_count:>+5d}")
        findings["C4"] = {"grok_dist": dict(grok_dist),
                          "gem_dist": dict(gem_dist),
                          "gold_dist": dict(gold_dist)}

        # C5: Precision/Recall complementarity
        grok_m = compute_prf(gold, grok_cl)
        gem_m = compute_prf(gold, gem_cl)
        print(f"\n  C5: Precision/Recall complementarity")
        print(f"    {'Class':<20s} {'Grok P':>7s} {'Grok R':>7s}  "
              f"{'Gem P':>7s} {'Gem R':>7s}")
        for lab in CLARITY_LABELS:
            gp = grok_m["per_class"][lab]
            gep = gem_m["per_class"][lab]
            print(f"    {lab:<20s} {gp['precision']:.3f}   {gp['recall']:.3f}  "
                  f"  {gep['precision']:.3f}   {gep['recall']:.3f}")
        findings["C5"] = {"grok": grok_m["per_class"],
                          "gemini": gem_m["per_class"]}

        # C6: Complementarity Venn
        grok_ev_correct = [grok_ev[i] in gold_ev_multi[i]
                           for i in range(len(data))]
        gem_ev_correct = [gem_ev[i] in gold_ev_multi[i]
                          for i in range(len(data))]
        both_right = sum(1 for g, e in zip(grok_ev_correct, gem_ev_correct)
                         if g and e)
        grok_only = sum(1 for g, e in zip(grok_ev_correct, gem_ev_correct)
                        if g and not e)
        gem_only = sum(1 for g, e in zip(grok_ev_correct, gem_ev_correct)
                       if not g and e)
        both_wrong = sum(1 for g, e in zip(grok_ev_correct, gem_ev_correct)
                         if not g and not e)
        n = len(data)
        print(f"\n  C6: Complementarity Venn (evasion)")
        print(f"    Both correct:    {both_right}/{n} ({100*both_right/n:.1f}%)")
        print(f"    Grok only right: {grok_only}/{n} ({100*grok_only/n:.1f}%)")
        print(f"    Gemini only:     {gem_only}/{n} ({100*gem_only/n:.1f}%)")
        print(f"    Both wrong:      {both_wrong}/{n} ({100*both_wrong/n:.1f}%)")
        findings["C6"] = {"both_right": both_right, "grok_only": grok_only,
                          "gem_only": gem_only, "both_wrong": both_wrong}

        # C7: Oracle gap (in clarity disagreements)
        print(f"\n  C7: Oracle gap (clarity disagreements)")
        oracle_right = 0
        actual_right = 0
        grok_right_in_disagree = 0
        gem_right_in_disagree = 0
        neither_right = 0
        for i in disagree_indices:
            grok_ok = grok_cl[i] == gold[i]
            gem_ok = gem_cl[i] == gold[i]
            if grok_ok or gem_ok:
                oracle_right += 1
            if ens_cl[i] == gold[i]:
                actual_right += 1
            if grok_ok:
                grok_right_in_disagree += 1
            if gem_ok:
                gem_right_in_disagree += 1
            if not grok_ok and not gem_ok:
                neither_right += 1
        n_dis = max(len(disagree_indices), 1)
        print(f"    Disagreements: {len(disagree_indices)}")
        print(f"    Oracle accuracy:  {oracle_right}/{n_dis} "
              f"({100*oracle_right/n_dis:.1f}%)")
        print(f"    Actual accuracy:  {actual_right}/{n_dis} "
              f"({100*actual_right/n_dis:.1f}%)")
        print(f"    Gap: {oracle_right - actual_right} samples")
        print(f"    Grok right: {grok_right_in_disagree} "
              f"({100*grok_right_in_disagree/n_dis:.1f}%)  "
              f"Gemini right: {gem_right_in_disagree} "
              f"({100*gem_right_in_disagree/n_dis:.1f}%)  "
              f"Neither: {neither_right}")
        findings["C7"] = {"oracle": oracle_right, "actual": actual_right,
                          "gap": oracle_right - actual_right,
                          "n_disagree": len(disagree_indices)}

        # C8: Self-consistency vs accuracy
        print(f"\n  C8: Self-consistency vs accuracy")
        for model_name in ["grok", "gemini"]:
            cons_buckets = defaultdict(lambda: [0, 0])
            for i, s in enumerate(data):
                cons = s.get(f"{model_name}_round0", {}).get("consistency", 1.0)
                if cons is None:
                    cons = 1.0
                bucket = round(cons, 1)
                ev_correct = (get_model_evasion(s, model_name)
                              in gold_ev_multi[i])
                cons_buckets[bucket][1] += 1
                if ev_correct:
                    cons_buckets[bucket][0] += 1
            print(f"    {model_name.capitalize()}:")
            for bucket in sorted(cons_buckets.keys(), reverse=True):
                correct, total = cons_buckets[bucket]
                acc = correct / max(total, 1)
                print(f"      cons={bucket:.1f}: {correct}/{total} "
                      f"({100*acc:.1f}%) evasion accuracy")
        findings["C8"] = "computed"

        # C9: CoT step consistency paradox
        print(f"\n  C9: CoT step-label consistency paradox")
        for model_name in ["grok", "gemini"]:
            consistent_correct = 0
            consistent_total = 0
            inconsistent_correct = 0
            inconsistent_total = 0
            for i, s in enumerate(data):
                responses = s.get(f"{model_name}_round0", {}).get("responses", [])
                for r in responses:
                    step_con = r.get("step_label_consistent")
                    label = r.get("label", "")
                    if step_con is None or not label:
                        continue
                    correct = label in gold_ev_multi[i]
                    if step_con:
                        consistent_total += 1
                        if correct:
                            consistent_correct += 1
                    else:
                        inconsistent_total += 1
                        if correct:
                            inconsistent_correct += 1
            con_acc = consistent_correct / max(consistent_total, 1)
            incon_acc = inconsistent_correct / max(inconsistent_total, 1)
            print(f"    {model_name.capitalize()}: "
                  f"step-consistent={100*con_acc:.1f}% ({consistent_total})  "
                  f"step-INconsistent={100*incon_acc:.1f}% ({inconsistent_total})")
        findings["C9"] = "computed"

        # C10: Confidence is flat/uninformative
        print(f"\n  C10: Confidence distribution")
        for model_name in ["grok", "gemini"]:
            conf_buckets = defaultdict(lambda: [0, 0])
            for i, s in enumerate(data):
                responses = s.get(f"{model_name}_round0", {}).get("responses", [])
                for r in responses:
                    conf = r.get("confidence")
                    if conf is None:
                        continue
                    correct = r.get("label", "") in gold_ev_multi[i]
                    conf_buckets[conf][1] += 1
                    if correct:
                        conf_buckets[conf][0] += 1
            print(f"    {model_name.capitalize()}:")
            total_resp = sum(v[1] for v in conf_buckets.values())
            for bucket in sorted(conf_buckets.keys()):
                correct, total = conf_buckets[bucket]
                acc = correct / max(total, 1)
                pct_of_all = 100 * total / max(total_resp, 1)
                print(f"      conf={bucket}: {total} ({pct_of_all:.1f}% of all)  "
                      f"accuracy={100*acc:.1f}%")
        findings["C10"] = "computed"

        # C11 & C12: Annotator agreement (test set only — has agreement field)
        if set_name == "TEST":
            print(f"\n  C11: Annotator agreement vs model accuracy")
            agree_levels = defaultdict(lambda: {"n": 0, "grok_cl_correct": 0,
                                                 "gem_cl_correct": 0,
                                                 "ens_correct": 0,
                                                 "grok_ev_correct": 0,
                                                 "gem_ev_correct": 0,
                                                 "models_agree": 0})
            for i, s in enumerate(data):
                ag = s.get("gold", {}).get("clarity_agreement", "unknown")
                agree_levels[ag]["n"] += 1
                if grok_cl[i] == gold[i]:
                    agree_levels[ag]["grok_cl_correct"] += 1
                if gem_cl[i] == gold[i]:
                    agree_levels[ag]["gem_cl_correct"] += 1
                if ens_cl[i] == gold[i]:
                    agree_levels[ag]["ens_correct"] += 1
                if grok_ev[i] in gold_ev_multi[i]:
                    agree_levels[ag]["grok_ev_correct"] += 1
                if gem_ev[i] in gold_ev_multi[i]:
                    agree_levels[ag]["gem_ev_correct"] += 1
                if grok_cl[i] == gem_cl[i]:
                    agree_levels[ag]["models_agree"] += 1

            for ag in ["unanimous", "majority", "all_different"]:
                d_ag = agree_levels.get(ag)
                if not d_ag or d_ag["n"] == 0:
                    print(f"    {ag}: n=0 (not present in data)")
                    continue
                n_ag = d_ag["n"]
                print(f"    {ag}: n={d_ag['n']}  "
                      f"ens_cl_acc={100*d_ag['ens_correct']/n_ag:.1f}%  "
                      f"grok_ev_acc={100*d_ag['grok_ev_correct']/n_ag:.1f}%  "
                      f"gem_ev_acc={100*d_ag['gem_ev_correct']/n_ag:.1f}%  "
                      f"models_agree={100*d_ag['models_agree']/n_ag:.1f}%")
            findings["C11"] = {k: dict(v) for k, v in agree_levels.items()}

            # C12: 3-way split paradox (higher evasion accuracy)
            print(f"\n  C12: Annotator 3-way split paradox")
            for ag in ["unanimous", "majority", "all_different"]:
                d_ag = agree_levels.get(ag)
                if not d_ag or d_ag["n"] == 0:
                    print(f"    {ag}: n=0 (not present)")
                    continue
                n_ag = d_ag["n"]
                print(f"    {ag}: evasion_acc="
                      f"{100*d_ag['grok_ev_correct']/n_ag:.1f}% (Grok), "
                      f"clarity_acc="
                      f"{100*d_ag['ens_correct']/n_ag:.1f}% (Ensemble)")
        else:
            print(f"\n  C11/C12: Annotator agreement — eval set has only 2 "
                  f"annotators (from task2_eval_labels.txt)")
            # Derive agreement from eval evasion labels (2 annotators)
            eval_ev_labels = gold_ev_multi
            n_unanimous = sum(1 for labels in eval_ev_labels
                              if len(set(labels)) == 1)
            n_disagree = sum(1 for labels in eval_ev_labels
                             if len(set(labels)) > 1)
            print(f"    Evasion unanimous: {n_unanimous}/{len(data)} "
                  f"({100*n_unanimous/len(data):.1f}%)")
            print(f"    Evasion disagree:  {n_disagree}/{len(data)} "
                  f"({100*n_disagree/len(data):.1f}%)")
            # Accuracy comparison
            unan_idx = [i for i, labels in enumerate(eval_ev_labels)
                        if len(set(labels)) == 1]
            disag_idx = [i for i, labels in enumerate(eval_ev_labels)
                         if len(set(labels)) > 1]
            if unan_idx:
                unan_acc = sum(1 for i in unan_idx if ens_cl[i] == gold[i]) / len(unan_idx)
                print(f"    Unanimous clarity acc: {100*unan_acc:.1f}%")
            if disag_idx:
                disag_acc = sum(1 for i in disag_idx if ens_cl[i] == gold[i]) / len(disag_idx)
                print(f"    Disagree clarity acc:  {100*disag_acc:.1f}%")

        # C13: Voting strategy comparison (in disagreements only)
        print(f"\n  C13: Voting strategy comparison (disagreements only)")
        strategies = {}
        for i in disagree_indices:
            s = data[i]
            g = gold[i]
            # Fixed 5/4 (current)
            strategies.setdefault("fixed_5_4", [0, 0])
            strategies["fixed_5_4"][1] += 1
            if ens_cl[i] == g:
                strategies["fixed_5_4"][0] += 1
            # Always Grok
            strategies.setdefault("always_grok", [0, 0])
            strategies["always_grok"][1] += 1
            if grok_cl[i] == g:
                strategies["always_grok"][0] += 1
            # Always Gemini
            strategies.setdefault("always_gemini", [0, 0])
            strategies["always_gemini"][1] += 1
            if gem_cl[i] == g:
                strategies["always_gemini"][0] += 1
            # Higher consistency wins
            strategies.setdefault("higher_cons", [0, 0])
            strategies["higher_cons"][1] += 1
            grok_cons = s.get("grok_round0", {}).get("consistency", 0.0) or 0.0
            gem_cons = s.get("gemini_round0", {}).get("consistency", 0.0) or 0.0
            pick = grok_cl[i] if grok_cons >= gem_cons else gem_cl[i]
            if pick == g:
                strategies["higher_cons"][0] += 1
            # Consistency-weighted
            strategies.setdefault("cons_weighted", [0, 0])
            strategies["cons_weighted"][1] += 1
            wv = Counter()
            for r in s.get("grok_round0", {}).get("responses", []):
                lab = r.get("label", "")
                if lab:
                    wv[EVASION_TO_CLARITY.get(lab, "Ambivalent")] += grok_cons
            for r in s.get("gemini_round0", {}).get("responses", []):
                lab = r.get("label", "")
                if lab:
                    wv[EVASION_TO_CLARITY.get(lab, "Ambivalent")] += gem_cons
            if wv and majority_label(dict(wv)) == g:
                strategies["cons_weighted"][0] += 1
            # Oracle
            strategies.setdefault("oracle", [0, 0])
            strategies["oracle"][1] += 1
            if grok_cl[i] == g or gem_cl[i] == g:
                strategies["oracle"][0] += 1

        for name, (correct, total) in sorted(strategies.items()):
            acc = correct / max(total, 1)
            marker = " <-- current" if name == "fixed_5_4" else ""
            print(f"    {name:<20s}: {correct}/{total} "
                  f"({100*acc:.1f}%){marker}")
        findings["C13"] = {k: {"correct": v[0], "total": v[1]}
                           for k, v in strategies.items()}

        # C14: 2D Agreement x Consistency sweet spot
        print(f"\n  C14: 2D Agreement x Consistency sweet spot")
        grid = defaultdict(lambda: [0, 0])
        for i, s in enumerate(data):
            grok_cons = s.get("grok_round0", {}).get("consistency", 1.0) or 1.0
            agree = grok_cl[i] == gem_cl[i]
            if grok_cons >= 0.9:
                cons_level = "high"
            elif grok_cons >= 0.6:
                cons_level = "medium"
            else:
                cons_level = "low"
            key = f"{'agree' if agree else 'disagree'}+{cons_level}"
            grid[key][1] += 1
            if ens_cl[i] == gold[i]:
                grid[key][0] += 1

        for key in sorted(grid.keys()):
            correct, total = grid[key]
            acc = correct / max(total, 1)
            print(f"    {key:<25s}: {correct}/{total} ({100*acc:.1f}%)")
        findings["C14"] = {k: {"correct": v[0], "total": v[1]}
                           for k, v in grid.items()}

        # C15: Confusion Resolution impact
        print(f"\n  C15: Confusion Resolution module impact")
        cr_triggered = 0
        cr_changed = 0
        for s in data:
            for model_name in ["grok", "gemini"]:
                responses = s.get(f"{model_name}_round0", {}).get("responses", [])
                for r in responses:
                    parse_info = r.get("parse", {})
                    if isinstance(parse_info, dict):
                        fb = parse_info.get("is_fallback", False)
                        fb_reason = parse_info.get("fallback_reason", "")
                        if fb and "confusion" in str(fb_reason).lower():
                            cr_triggered += 1
        print(f"    CR-like fallbacks detected: {cr_triggered}")
        # Check if vote_inputs differ from raw responses (indicates CR)
        cr_vote_changes = 0
        for s in data:
            for model_name in ["grok", "gemini"]:
                r0 = s.get(f"{model_name}_round0", {})
                vote_inputs = r0.get("vote_inputs", {})
                vote_counts = r0.get("vote_counts", {})
                if vote_inputs and vote_counts and vote_inputs.get("vote_counts", {}) != vote_counts:
                    cr_vote_changes += 1
        print(f"    Samples with vote_inputs != vote_counts: {cr_vote_changes}")
        findings["C15"] = {"cr_triggered": cr_triggered,
                           "vote_changes": cr_vote_changes}

        results[set_name.lower()] = findings

    return results


# ============================================================================
# SECTION D: Debate Ablation (test set ONLY)
# ============================================================================

def section_d(test_data, test_gold):
    """Section D: Debate ablation analysis (test set only)."""
    print("\n" + "█" * 70)
    print("  SECTION D: DEBATE ABLATION (test set only)")
    print("█" * 70)

    debate_path = BASE_DIR / "apex_debate_ablation" / "test_set_full.json"
    if not debate_path.exists():
        print("  Debate data not found!")
        return {}

    debate_data = json.loads(debate_path.read_text(encoding="utf-8"))
    assert len(debate_data) == len(test_gold)
    gold = test_gold
    gold_ev_multi = get_gold_evasion_multi(debate_data, "test")

    # S0 baseline (pre-debate)
    s0_preds = [get_pre_dcg_clarity(s) for s in debate_data]
    s0_metrics = compute_prf(gold, s0_preds)

    # Debate predictions
    debate_preds = [s.get("debate", {}).get("final_clarity", "") or
                    get_pre_dcg_clarity(s) for s in debate_data]
    debate_metrics = compute_prf(gold, debate_preds)

    # D1: Debate = net zero
    print(f"\n  D1: Debate = net zero")
    print(f"    S0 F1:     {fmt_f1(s0_metrics['macro_f1'])}  "
          f"Acc={fmt_pct(s0_metrics['accuracy'])}")
    print(f"    +Debate F1: {fmt_f1(debate_metrics['macro_f1'])}  "
          f"Acc={fmt_pct(debate_metrics['accuracy'])}")
    delta = debate_metrics["macro_f1"] - s0_metrics["macro_f1"]
    print(f"    Delta: {delta:+.4f}")

    # Count helped, hurt, same
    helped = sum(1 for i in range(len(gold))
                 if s0_preds[i] != gold[i] and debate_preds[i] == gold[i])
    hurt = sum(1 for i in range(len(gold))
               if s0_preds[i] == gold[i] and debate_preds[i] != gold[i])
    changed = sum(1 for i in range(len(gold))
                  if s0_preds[i] != debate_preds[i])
    print(f"    Changed: {changed}  Helped: {helped}  Hurt: {hurt}  "
          f"Net: {helped - hurt:+d}")

    # Identify triggered samples
    triggered = [i for i, s in enumerate(debate_data)
                 if s.get("debate", {}).get("triggered")]
    print(f"    Debate triggered: {len(triggered)}/{len(debate_data)}")

    # D2: Gemini is submissive debater
    print(f"\n  D2: Who flips in debate?")
    grok_flipped = 0
    gem_flipped = 0
    both_flipped = 0
    neither_flipped = 0
    gem_flip_correct = 0
    grok_flip_correct = 0

    for i in triggered:
        s = debate_data[i]
        rounds = s.get("debate", {}).get("rounds", [])
        if not rounds:
            continue
        r0_grok = get_model_evasion(s, "grok")
        r0_gemini = get_model_evasion(s, "gemini")

        last_round = rounds[-1]
        final_grok = last_round.get("grok", {}).get("majority_label", r0_grok)
        final_gemini = last_round.get("gemini", {}).get("majority_label", r0_gemini)

        grok_changed = final_grok != r0_grok
        gem_changed = final_gemini != r0_gemini

        if grok_changed and gem_changed:
            both_flipped += 1
        elif grok_changed:
            grok_flipped += 1
            if final_grok in gold_ev_multi[i]:
                grok_flip_correct += 1
        elif gem_changed:
            gem_flipped += 1
            if final_gemini in gold_ev_multi[i]:
                gem_flip_correct += 1
        else:
            neither_flipped += 1

    print(f"    Only Grok flipped:  {grok_flipped}  "
          f"(correct: {grok_flip_correct})")
    print(f"    Only Gemini flipped: {gem_flipped}  "
          f"(correct: {gem_flip_correct})")
    print(f"    Both flipped:       {both_flipped}")
    print(f"    Neither flipped:    {neither_flipped}")

    # D3: Direction of changes
    print(f"\n  D3: Direction of evasion changes")
    change_directions = Counter()
    for i in triggered:
        s = debate_data[i]
        r0_ev = get_ensemble_evasion(s)
        debate_ev_final = s.get("debate", {}).get("final_clarity", "")
        # Track evasion label changes per model
        rounds = s.get("debate", {}).get("rounds", [])
        for rd in rounds:
            for model_name in ["grok", "gemini"]:
                r0_lab = get_model_evasion(s, model_name)
                rd_lab = rd.get(model_name, {}).get("majority_label", r0_lab)
                if rd_lab != r0_lab:
                    change_directions[f"{r0_lab} -> {rd_lab}"] += 1
    for change, count in change_directions.most_common(10):
        print(f"    {change}: {count}")

    # D4: Convergence != correctness
    print(f"\n  D4: Convergence vs correctness")
    from collections import Counter as C2
    decision_counts = Counter(s.get("debate", {}).get("final_decision", "")
                              for s in debate_data)
    for decision in ["AGREE_ROUND0", "AGREE_ROUND1", "AGREE_ROUND2",
                      "AGREE_ROUND3", "DEBATE_AGGREGATED_R2",
                      "DEBATE_AGGREGATED_R3"]:
        indices = [i for i, s in enumerate(debate_data)
                   if s.get("debate", {}).get("final_decision") == decision]
        if not indices:
            continue
        correct = sum(1 for i in indices if debate_preds[i] == gold[i])
        s0_correct = sum(1 for i in indices if s0_preds[i] == gold[i])
        n = len(indices)
        print(f"    {decision:<25s}: n={n:>3d}  "
              f"debate_acc={100*correct/n:.1f}%  s0_acc={100*s0_correct/n:.1f}%"
              f"  net={correct-s0_correct:+d}")

    # D5: Debate doesn't preserve correct model
    print(f"\n  D5: Does debate preserve the correct model?")
    grok_was_right = 0
    grok_right_preserved = 0
    gem_was_right = 0
    gem_right_preserved = 0
    for i in triggered:
        s = debate_data[i]
        grok_right = get_model_clarity(s, "grok") == gold[i]
        gem_right = get_model_clarity(s, "gemini") == gold[i]
        if grok_right:
            grok_was_right += 1
            if debate_preds[i] == gold[i]:
                grok_right_preserved += 1
        if gem_right:
            gem_was_right += 1
            if debate_preds[i] == gold[i]:
                gem_right_preserved += 1
    print(f"    Grok was right: {grok_was_right}, preserved: "
          f"{grok_right_preserved} "
          f"({100*grok_right_preserved/max(grok_was_right,1):.1f}%)")
    print(f"    Gemini was right: {gem_was_right}, preserved: "
          f"{gem_right_preserved} "
          f"({100*gem_right_preserved/max(gem_was_right,1):.1f}%)")

    # D6: API cost
    print(f"\n  D6: API cost analysis")
    total_debate_calls = sum(
        len(s.get("debate", {}).get("rounds", [])) * 2
        for s in debate_data if s.get("debate", {}).get("triggered"))
    baseline_calls = len(debate_data) * 10  # 2 models x k=5
    print(f"    Baseline API calls: {baseline_calls}")
    print(f"    Extra debate calls: {total_debate_calls}")
    print(f"    Overhead: +{100*total_debate_calls/baseline_calls:.1f}%")

    # D7: Progressive cap = identical
    print(f"\n  D7: Progressive cap results")
    cap_metrics_path = (BASE_DIR / "apex_debate_ablation" /
                        "apex_debate_debate_round_caps_metrics.json")
    if cap_metrics_path.exists():
        cm = json.loads(cap_metrics_path.read_text(encoding="utf-8"))
        for cap in ["1", "2", "3"]:
            cl = cm.get("caps", {}).get(cap, {}).get("clarity", {})
            print(f"    Cap R{cap}: F1={fmt_f1(cl.get('macro_f1', 0))}  "
                  f"Delta from S0: "
                  f"{cl.get('macro_f1', 0) - s0_metrics['macro_f1']:+.4f}")

    return {"debate_f1": debate_metrics["macro_f1"],
            "s0_f1": s0_metrics["macro_f1"],
            "triggered": len(triggered),
            "helped": helped, "hurt": hurt}


# SECTION E: Hu Debate Comparison (test set ONLY)

def section_e(test_data, test_gold):
    """Section E: Hu 7-agent debate comparison (test set only)."""
    print("\n" + "=" * 70)
    print("  SECTION E: HU DEBATE COMPARISON (test set only)")
    print("=" * 70)

    hu_path = BASE_DIR / "Tianyu_Hu_Debate_ablation" / "debate_llm_judges_full.json"
    if not hu_path.exists():
        print("  Hu debate data not found!")
        return {}

    hu_data = json.loads(hu_path.read_text(encoding="utf-8"))
    assert len(hu_data) == len(test_gold)
    gold = test_gold
    gold_ev_multi = get_gold_evasion_multi(hu_data, "test")

    # APEX S0 baseline
    s0_preds = [get_pre_dcg_clarity(s) for s in test_data]
    s0_metrics = compute_prf(gold, s0_preds)

    # Hu predictions
    hu_preds = [s.get("final_clarity", "") for s in hu_data]
    hu_metrics = compute_prf(gold, hu_preds)

    # E1: APEX S0 > Hu
    print(f"\n  E1: APEX S0 vs Hu 7-agent")
    print(f"    APEX S0:     F1={fmt_f1(s0_metrics['macro_f1'])}  "
          f"Acc={fmt_pct(s0_metrics['accuracy'])}")
    print(f"    Hu 7-agent:  F1={fmt_f1(hu_metrics['macro_f1'])}  "
          f"Acc={fmt_pct(hu_metrics['accuracy'])}")
    delta = s0_metrics["macro_f1"] - hu_metrics["macro_f1"]
    print(f"    APEX advantage: +{delta:.4f} F1")

    # Per-class comparison
    print(f"    Per-class F1:")
    for lab in CLARITY_LABELS:
        s0_f1 = s0_metrics["per_class"][lab]["f1"]
        hu_f1 = hu_metrics["per_class"][lab]["f1"]
        print(f"      {lab:<20s}: S0={s0_f1:.4f}  Hu={hu_f1:.4f}  "
              f"delta={s0_f1-hu_f1:+.4f}")

    # E2: Debate hurts BOTH architectures
    print(f"\n  E2: Debate hurts both architectures")
    # Hu without debate = round0 only
    hu_r0_preds = []
    for s in hu_data:
        r0 = s.get("round0", {})
        cl_dist = r0.get("clarity_distribution", {})
        if cl_dist:
            hu_r0_preds.append(max(cl_dist, key=cl_dist.get))
        else:
            hu_r0_preds.append(s.get("final_clarity", ""))
    hu_r0_metrics = compute_prf(gold, hu_r0_preds)
    hu_debate_delta = hu_metrics["macro_f1"] - hu_r0_metrics["macro_f1"]
    print(f"    Hu R0 (no debate): F1={fmt_f1(hu_r0_metrics['macro_f1'])}")
    print(f"    Hu +debate:        F1={fmt_f1(hu_metrics['macro_f1'])}  "
          f"delta={hu_debate_delta:+.4f}")

    # E3: Grok k=5 SC > Hu 7×Grok
    print(f"\n  E3: Single model k=5 vs 7-agent debate")
    # Grok k=5 from test data
    grok_preds = [get_model_clarity(s, "grok") for s in test_data]
    grok_metrics = compute_prf(gold, grok_preds)
    print(f"    Grok k=5 alone: F1={fmt_f1(grok_metrics['macro_f1'])}  "
          f"API calls=5/sample")
    print(f"    Hu 7×Grok:      F1={fmt_f1(hu_metrics['macro_f1'])}  "
          f"API calls=7+/sample")
    print(f"    k=5 advantage:  {grok_metrics['macro_f1']-hu_metrics['macro_f1']:+.4f}")

    # E4: Groupthink — unanimity analysis
    print(f"\n  E4: Unanimity = groupthink?")
    r0_unanimous = sum(1 for s in hu_data
                       if s.get("round0", {}).get("unanimous"))
    print(f"    Round 0 unanimous: {r0_unanimous}/{len(hu_data)} "
          f"({100*r0_unanimous/len(hu_data):.1f}%)")

    # Check unanimity accuracy
    r0_unan_correct = sum(1 for s, g in zip(hu_data, gold)
                          if s.get("round0", {}).get("unanimous")
                          and s.get("final_clarity") == g)
    r0_unan_total = max(r0_unanimous, 1)
    print(f"    Round 0 unanimous accuracy: "
          f"{100*r0_unan_correct/r0_unan_total:.1f}%")

    # Per-round unanimity for triggered samples
    for s in hu_data:
        if s.get("debate_triggered"):
            for rd in s.get("debate_rounds", []):
                if rd.get("unanimous"):
                    # Check if this convergence was correct
                    pass

    # Decision methods
    decision_dist = Counter(s.get("decision_method", "") for s in hu_data)
    print(f"    Decision methods: {dict(decision_dist)}")

    # E5: Agent drift
    print(f"\n  E5: Agent drift in round 1")
    total_agents_changed = 0
    total_agents_in_debate = 0
    for s in hu_data:
        if not s.get("debate_triggered"):
            continue
        rounds = s.get("debate_rounds", [])
        if not rounds:
            continue
        r0_agents = s.get("round0", {}).get("agents", [])
        r1_agents = rounds[0].get("agents", [])
        for a_r0, a_r1 in zip(r0_agents, r1_agents):
            total_agents_in_debate += 1
            if a_r0.get("evasion_label") != a_r1.get("evasion_label"):
                total_agents_changed += 1
    if total_agents_in_debate > 0:
        print(f"    Agents in debate: {total_agents_in_debate}")
        print(f"    Changed label R0→R1: {total_agents_changed} "
              f"({100*total_agents_changed/total_agents_in_debate:.1f}%)")

    # E6: Bias amplification — per-class comparison
    print(f"\n  E6: Bias amplification (per-evasion-class)")
    hu_ev_preds = [s.get("final_evasion", "") for s in hu_data]
    hu_ev_dist = Counter(hu_ev_preds)
    grok_ev_preds = [get_model_evasion(s, "grok") for s in test_data]
    grok_ev_dist = Counter(grok_ev_preds)
    gold_ev_flat = [g[0] if g else "" for g in gold_ev_multi]
    gold_ev_dist = Counter(gold_ev_flat)

    evasion_classes = sorted(set(gold_ev_flat) | set(hu_ev_preds) | set(grok_ev_preds))
    print(f"    {'Label':<22s} {'Gold':>5s} {'Grok':>5s} {'Hu':>5s}")
    for lab in evasion_classes:
        if not lab:
            continue
        print(f"    {lab:<22s} {gold_ev_dist.get(lab,0):>5d} "
              f"{grok_ev_dist.get(lab,0):>5d} {hu_ev_dist.get(lab,0):>5d}")

    # E7: APEX wins specifically on Explicit
    print(f"\n  E7: Per-class advantage (APEX vs Hu)")
    for lab in CLARITY_LABELS:
        s0_f1 = s0_metrics["per_class"][lab]["f1"]
        hu_f1 = hu_metrics["per_class"][lab]["f1"]
        winner = "APEX" if s0_f1 > hu_f1 else "Hu"
        print(f"    {lab:<20s}: APEX={s0_f1:.4f} vs Hu={hu_f1:.4f} "
              f"({winner} +{abs(s0_f1-hu_f1):.4f})")

    # API cost comparison
    total_hu_calls = sum(s.get("total_api_calls", 7) for s in hu_data)
    apex_calls = len(test_data) * 10  # 2 models x k=5
    print(f"\n    API calls: APEX={apex_calls}  Hu={total_hu_calls}  "
          f"Hu/APEX={total_hu_calls/apex_calls:.1f}x")

    return {"s0_f1": s0_metrics["macro_f1"],
            "hu_f1": hu_metrics["macro_f1"],
            "hu_r0_f1": hu_r0_metrics["macro_f1"]}


# SECTION F: Generalization Findings (both sets)

def section_f(test_data, eval_data, test_gold, eval_gold):
    """Section F: Generalization findings across systems and sets."""
    print("\n" + "=" * 70)
    print("  SECTION F: GENERALIZATION FINDINGS")
    print("=" * 70)

    # Load Hu data for cross-system comparison
    hu_path = BASE_DIR / "Tianyu_Hu_Debate_ablation" / "debate_llm_judges_full.json"
    hu_data = json.loads(hu_path.read_text(encoding="utf-8")) if hu_path.exists() else None

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        print(f"\n{'─' * 70}")
        print(f"  {set_name} SET (n={len(data)})")
        print(f"{'─' * 70}")
        findings = {}

        gold_ev_multi = get_gold_evasion_multi(data, set_name.lower())

        # F1: Class imbalance
        print(f"\n  F1: Class imbalance")
        cl_dist = Counter(gold)
        n = len(gold)
        for lab in CLARITY_LABELS:
            print(f"    {lab:<20s}: {cl_dist.get(lab,0):>4d} "
                  f"({100*cl_dist.get(lab,0)/n:.1f}%)")

        if set_name == "TEST":
            gold_ev_flat = [g[0] if g else "" for g in gold_ev_multi]
            ev_dist = Counter(gold_ev_flat)
            print(f"    Evasion distribution:")
            for lab, count in ev_dist.most_common():
                if lab:
                    print(f"      {lab:<22s}: {count:>3d} "
                          f"({100*count/n:.1f}%)")
        findings["F1"] = dict(cl_dist)

        # F2: Annotator unanimity
        print(f"\n  F2: Annotator unanimity")
        if set_name == "TEST":
            agree_dist = Counter(
                s.get("gold", {}).get("agreement", "unknown") for s in data)
            for ag, count in agree_dist.most_common():
                print(f"    {ag:<20s}: {count:>3d} ({100*count/n:.1f}%)")

            # Per-evasion-class unanimity
            print(f"    Per-class unanimity rate:")
            ev_agree = defaultdict(lambda: [0, 0])
            for s in data:
                ev_labs = s.get("gold", {}).get("evasion_labels", [])
                ag = s.get("gold", {}).get("agreement", "")
                majority = s.get("gold", {}).get("majority_label", "")
                if majority:
                    ev_agree[majority][1] += 1
                    if ag == "unanimous":
                        ev_agree[majority][0] += 1
            for ev_class in sorted(ev_agree.keys()):
                unan, total = ev_agree[ev_class]
                print(f"      {ev_class:<22s}: {unan}/{total} "
                      f"({100*unan/max(total,1):.0f}%)")
        else:
            n_unan = sum(1 for labels in gold_ev_multi
                         if len(set(labels)) == 1)
            print(f"    Evasion unanimous (2 annotators): {n_unan}/{n} "
                  f"({100*n_unan/n:.1f}%)")

        # F3: Universally hard samples
        print(f"\n  F3: Universally hard samples")
        grok_cl = [get_model_clarity(s, "grok") for s in data]
        gem_cl = [get_model_clarity(s, "gemini") for s in data]
        ens_cl = [get_pre_dcg_clarity(s) for s in data]

        if set_name == "TEST" and hu_data:
            hu_preds = [s.get("final_clarity", "") for s in hu_data]
            all_wrong = sum(
                1 for i in range(n)
                if grok_cl[i] != gold[i]
                and gem_cl[i] != gold[i]
                and ens_cl[i] != gold[i]
                and hu_preds[i] != gold[i])
            print(f"    Wrong for ALL 4 systems: {all_wrong}/{n} "
                  f"({100*all_wrong/n:.1f}%)")

            # Also check: wrong for at least 3
            at_least_3_wrong = sum(
                1 for i in range(n)
                if sum([grok_cl[i] != gold[i], gem_cl[i] != gold[i],
                        ens_cl[i] != gold[i], hu_preds[i] != gold[i]]) >= 3)
            print(f"    Wrong for 3+ systems:    {at_least_3_wrong}/{n} "
                  f"({100*at_least_3_wrong/n:.1f}%)")
        else:
            all_wrong = sum(
                1 for i in range(n)
                if grok_cl[i] != gold[i]
                and gem_cl[i] != gold[i]
                and ens_cl[i] != gold[i])
            print(f"    Wrong for Grok+Gem+Ens: {all_wrong}/{n} "
                  f"({100*all_wrong/n:.1f}%)")
        findings["F3"] = all_wrong

        # F4: Evasion taxonomy = error buffer
        print(f"\n  F4: Evasion taxonomy as error buffer")
        grok_ev = [get_model_evasion(s, "grok") for s in data]
        gem_ev = [get_model_evasion(s, "gemini") for s in data]

        for model_name, ev_preds in [("Grok", grok_ev), ("Gemini", gem_ev)]:
            ev_wrong = sum(1 for i in range(n)
                           if ev_preds[i] not in gold_ev_multi[i])
            ev_wrong_but_cl_right = sum(
                1 for i in range(n)
                if ev_preds[i] not in gold_ev_multi[i]
                and EVASION_TO_CLARITY.get(ev_preds[i], "Ambivalent") ==
                gold[i])
            buffer_rate = ev_wrong_but_cl_right / max(ev_wrong, 1)
            print(f"    {model_name}: {ev_wrong} evasion errors, "
                  f"{ev_wrong_but_cl_right} absorbed "
                  f"({100*buffer_rate:.1f}% buffer rate)")

        # F5: Explicit↔Implicit = 1/3 of all evasion errors
        print(f"\n  F5: Explicit<->Implicit error rate")
        for model_name, ev_preds in [("Grok", grok_ev), ("Gemini", gem_ev),
                                      ("Ensemble", [get_ensemble_evasion(s)
                                                    for s in data])]:
            ev_errors = sum(1 for i in range(n)
                            if ev_preds[i] not in gold_ev_multi[i])
            exp_imp = sum(
                1 for i in range(n)
                if ev_preds[i] not in gold_ev_multi[i]
                and ((ev_preds[i] == "Explicit" and "Implicit" in gold_ev_multi[i])
                     or (ev_preds[i] == "Implicit" and "Explicit" in gold_ev_multi[i])))
            print(f"    {model_name}: {exp_imp}/{max(ev_errors,1)} "
                  f"({100*exp_imp/max(ev_errors,1):.1f}%) Explicit<->Implicit")

        # F6: Debate degrades subjective classification (covered in D+E)
        print(f"\n  F6: Debate degrades subjective classification")
        print(f"    (See Sections D and E for detailed analysis)")

        # F7: 2 different models > 7 identical
        print(f"\n  F7: Diversity > quantity")
        if set_name == "TEST" and hu_data:
            ens_m = compute_prf(gold, ens_cl)
            hu_m = compute_prf(gold, [s.get("final_clarity", "")
                                      for s in hu_data])
            print(f"    APEX (2 models, 10 calls): F1={fmt_f1(ens_m['macro_f1'])}")
            print(f"    Hu (7 agents, 7+ calls):   F1={fmt_f1(hu_m['macro_f1'])}")
            print(f"    Advantage: +{ens_m['macro_f1']-hu_m['macro_f1']:.4f}")
        else:
            print(f"    (Test set only — no Hu eval data)")

        # F8: System >= human agreement
        print(f"\n  F8: System vs human agreement")
        ens_m = compute_prf(gold, ens_cl)
        post_dcg = [get_ensemble_clarity(s) for s in data]
        post_dcg_m = compute_prf(gold, post_dcg)
        print(f"    S0 accuracy: {fmt_pct(ens_m['accuracy'])}")
        print(f"    S0+DCG accuracy: {fmt_pct(post_dcg_m['accuracy'])}")
        if set_name == "TEST":
            # Annotator agreement rate
            agree_counts = Counter(
                s.get("gold", {}).get("agreement", "") for s in data)
            unanimous_pct = agree_counts.get("unanimous", 0) / n
            print(f"    Annotator agreement: unanimous={fmt_pct(unanimous_pct)}")
            # Cohen's kappa proxy: annotator pairwise agreement
            # Use clarity_agreement as proxy
            cl_agree = Counter(
                s.get("gold", {}).get("clarity_agreement", "") for s in data)
            cl_unan_pct = cl_agree.get("unanimous", 0) / n
            print(f"    Clarity unanimous: {fmt_pct(cl_unan_pct)}")
            print(f"    → System accuracy ({fmt_pct(post_dcg_m['accuracy'])}) "
                  f"vs annotator clarity unanimity ({fmt_pct(cl_unan_pct)})")

        # F9: Human difficulty ↔ model difficulty
        print(f"\n  F9: Human-model difficulty correlation")
        if set_name == "TEST":
            for ag_level in ["unanimous", "majority"]:
                indices = [i for i, s in enumerate(data)
                           if s.get("gold", {}).get("clarity_agreement") == ag_level]
                if not indices:
                    continue
                acc = sum(1 for i in indices if ens_cl[i] == gold[i]) / len(indices)
                model_agree = sum(1 for i in indices
                                  if grok_cl[i] == gem_cl[i]) / len(indices)
                print(f"    Human {ag_level}: {len(indices)} samples  "
                      f"S0_acc={100*acc:.1f}%  models_agree={100*model_agree:.1f}%")
        else:
            unan_ev = [i for i, labels in enumerate(gold_ev_multi)
                       if len(set(labels)) == 1]
            disag_ev = [i for i, labels in enumerate(gold_ev_multi)
                        if len(set(labels)) > 1]
            if unan_ev:
                unan_acc = sum(1 for i in unan_ev if ens_cl[i] == gold[i]) / len(unan_ev)
                print(f"    Annotator unanimous ({len(unan_ev)}): "
                      f"S0_acc={100*unan_acc:.1f}%")
            if disag_ev:
                disag_acc = sum(1 for i in disag_ev if ens_cl[i] == gold[i]) / len(disag_ev)
                print(f"    Annotator disagree ({len(disag_ev)}):  "
                      f"S0_acc={100*disag_acc:.1f}%")

        results[set_name.lower()] = findings

    return results


# SECTION G: Error Analysis (both sets)

def section_g(test_data, eval_data, test_gold, eval_gold):
    """Section G: Detailed error analysis."""
    print("\n" + "=" * 70)
    print("  SECTION G: ERROR ANALYSIS")
    print("=" * 70)

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        print(f"\n{'─' * 70}")
        print(f"  {set_name} SET (n={len(data)})")
        print(f"{'─' * 70}")
        findings = {}
        n = len(data)

        ens_cl = [get_ensemble_clarity(s) for s in data]  # post-DCG
        pre_dcg = [get_pre_dcg_clarity(s) for s in data]
        gold_ev_multi = get_gold_evasion_multi(data, set_name.lower())
        grok_ev = [get_model_evasion(s, "grok") for s in data]
        gem_ev = [get_model_evasion(s, "gemini") for s in data]
        grok_cl = [get_model_clarity(s, "grok") for s in data]
        gem_cl = [get_model_clarity(s, "gemini") for s in data]

        # G1: Clarity confusion matrix (post-DCG)
        print(f"\n  G1: Clarity confusion matrix (post-DCG)")
        print(f"    {'Predicted →':<15s}", end="")
        for lab in CLARITY_LABELS:
            print(f" {lab[:6]:>8s}", end="")
        print()
        for gold_lab in CLARITY_LABELS:
            print(f"    {gold_lab:<15s}", end="")
            for pred_lab in CLARITY_LABELS:
                count = sum(1 for g, p in zip(gold, ens_cl)
                            if g == gold_lab and p == pred_lab)
                print(f" {count:>8d}", end="")
            print()

        # G2: Evasion confusion matrix (top errors)
        print(f"\n  G2: Top evasion confusions (ensemble)")
        ens_ev = [get_ensemble_evasion(s) for s in data]
        ev_confusions = Counter()
        for i in range(n):
            if ens_ev[i] not in gold_ev_multi[i]:
                gold_first = gold_ev_multi[i][0] if gold_ev_multi[i] else "?"
                ev_confusions[(gold_first, ens_ev[i])] += 1
        print(f"    {'Gold → Predicted':<45s} {'Count':>5s}")
        for (g, p), count in ev_confusions.most_common(15):
            # Check if this crosses clarity boundary
            g_cl = EVASION_TO_CLARITY.get(g, "?")
            p_cl = EVASION_TO_CLARITY.get(p, "?")
            crossing = "CROSS" if g_cl != p_cl else "same"
            print(f"    {g:<22s} → {p:<18s} {count:>5d}  ({crossing})")

        # G3: Per-class error breakdown (which gold class is hardest)
        print(f"\n  G3: Per-class error rate (post-DCG)")
        for lab in CLARITY_LABELS:
            support = sum(1 for g in gold if g == lab)
            errors = sum(1 for g, p in zip(gold, ens_cl)
                         if g == lab and p != lab)
            err_rate = errors / max(support, 1)
            # Where do errors go?
            error_dest = Counter(p for g, p in zip(gold, ens_cl)
                                 if g == lab and p != lab)
            print(f"    {lab:<20s}: {errors}/{support} "
                  f"({100*err_rate:.1f}% error rate)")
            for dest, cnt in error_dest.most_common():
                print(f"      → {dest}: {cnt}")

        # G4: Error by answer length (quartiles)
        print(f"\n  G4: Error rate by answer length")
        answer_lengths = [s.get("text_features", {}).get("answer_word_count", 0)
                          for s in data]
        # Split into quartiles
        sorted_lens = sorted(set(answer_lengths))
        if sorted_lens:
            q1_idx = len(sorted_lens) // 4
            q2_idx = len(sorted_lens) // 2
            q3_idx = 3 * len(sorted_lens) // 4
            q1_val = sorted_lens[min(q1_idx, len(sorted_lens)-1)]
            q2_val = sorted_lens[min(q2_idx, len(sorted_lens)-1)]
            q3_val = sorted_lens[min(q3_idx, len(sorted_lens)-1)]

            buckets = {"short (≤Q1)": [], "medium (Q1-Q2)": [],
                       "long (Q2-Q3)": [], "very long (>Q3)": []}
            for i in range(n):
                al = answer_lengths[i]
                if al <= q1_val:
                    buckets["short (≤Q1)"].append(i)
                elif al <= q2_val:
                    buckets["medium (Q1-Q2)"].append(i)
                elif al <= q3_val:
                    buckets["long (Q2-Q3)"].append(i)
                else:
                    buckets["very long (>Q3)"].append(i)

            print(f"    Quartiles: Q1={q1_val}, Q2={q2_val}, Q3={q3_val} words")
            for bucket_name, indices in buckets.items():
                if not indices:
                    continue
                errors = sum(1 for i in indices if ens_cl[i] != gold[i])
                acc = 1 - errors / len(indices)
                print(f"    {bucket_name:<22s}: n={len(indices):>3d}  "
                      f"acc={100*acc:.1f}%  err={errors}")

        # G5: Error by question length
        print(f"\n  G5: Error rate by question length")
        q_lengths = [s.get("text_features", {}).get("question_word_count", 0)
                     for s in data]
        q_buckets = {"short (≤10)": [], "medium (11-20)": [],
                     "long (21-40)": [], "very long (>40)": []}
        for i in range(n):
            ql = q_lengths[i]
            if ql <= 10:
                q_buckets["short (≤10)"].append(i)
            elif ql <= 20:
                q_buckets["medium (11-20)"].append(i)
            elif ql <= 40:
                q_buckets["long (21-40)"].append(i)
            else:
                q_buckets["very long (>40)"].append(i)
        for bucket_name, indices in q_buckets.items():
            if not indices:
                continue
            errors = sum(1 for i in indices if ens_cl[i] != gold[i])
            acc = 1 - errors / len(indices)
            print(f"    {bucket_name:<22s}: n={len(indices):>3d}  "
                  f"acc={100*acc:.1f}%  err={errors}")

        # G6: Systematic error patterns
        print(f"\n  G6: Systematic error patterns")
        # Errors where both models agree but are wrong
        both_agree_wrong = sum(
            1 for i in range(n)
            if grok_cl[i] == gem_cl[i]
            and grok_cl[i] != gold[i])
        both_agree_total = sum(
            1 for i in range(n)
            if grok_cl[i] == gem_cl[i])
        print(f"    Both models agree but WRONG: {both_agree_wrong}/"
              f"{both_agree_total} "
              f"({100*both_agree_wrong/max(both_agree_total,1):.1f}%)")

        # Errors where DCG makes it worse
        dcg_hurt = sum(1 for i in range(n)
                       if pre_dcg[i] == gold[i] and ens_cl[i] != gold[i])
        dcg_helped = sum(1 for i in range(n)
                         if pre_dcg[i] != gold[i] and ens_cl[i] == gold[i])
        print(f"    DCG helped: {dcg_helped}, DCG hurt: {dcg_hurt}, "
              f"net: {dcg_helped - dcg_hurt:+d}")

        # High-confidence errors (both models cons=1.0 but wrong)
        hc_wrong = sum(
            1 for i, s in enumerate(data)
            if grok_cl[i] == gem_cl[i]
            and (s.get("grok_round0", {}).get("consistency", 0) or 0) >= 1.0
            and (s.get("gemini_round0", {}).get("consistency", 0) or 0) >= 1.0
            and ens_cl[i] != gold[i])
        hc_total = sum(
            1 for i, s in enumerate(data)
            if grok_cl[i] == gem_cl[i]
            and (s.get("grok_round0", {}).get("consistency", 0) or 0) >= 1.0
            and (s.get("gemini_round0", {}).get("consistency", 0) or 0) >= 1.0)
        print(f"    High-confidence (agree+cons=1.0) wrong: "
              f"{hc_wrong}/{hc_total} "
              f"({100*hc_wrong/max(hc_total,1):.1f}%)")

        # G7: Model-specific weakness
        print(f"\n  G7: Model-specific errors (one right, other wrong)")
        grok_right_gem_wrong = []
        gem_right_grok_wrong = []
        for i in range(n):
            if grok_cl[i] == gold[i] and gem_cl[i] != gold[i]:
                grok_right_gem_wrong.append(i)
            elif gem_cl[i] == gold[i] and grok_cl[i] != gold[i]:
                gem_right_grok_wrong.append(i)

        print(f"    Grok right, Gemini wrong: {len(grok_right_gem_wrong)}")
        # What does Gemini predict when wrong?
        gem_wrong_preds = Counter(gem_cl[i] for i in grok_right_gem_wrong)
        for pred, cnt in gem_wrong_preds.most_common():
            print(f"      Gemini predicts: {pred} ({cnt})")

        print(f"    Gemini right, Grok wrong: {len(gem_right_grok_wrong)}")
        grok_wrong_preds = Counter(grok_cl[i] for i in gem_right_grok_wrong)
        for pred, cnt in grok_wrong_preds.most_common():
            print(f"      Grok predicts: {pred} ({cnt})")

        # Does ensemble pick the right one?
        ens_picks_right_grok = sum(1 for i in grok_right_gem_wrong
                                   if ens_cl[i] == gold[i])
        ens_picks_right_gem = sum(1 for i in gem_right_grok_wrong
                                  if ens_cl[i] == gold[i])
        print(f"    Ensemble picks correct (Grok wins): "
              f"{ens_picks_right_grok}/{len(grok_right_gem_wrong)} "
              f"({100*ens_picks_right_grok/max(len(grok_right_gem_wrong),1):.1f}%)")
        print(f"    Ensemble picks correct (Gem wins):  "
              f"{ens_picks_right_gem}/{len(gem_right_grok_wrong)} "
              f"({100*ens_picks_right_gem/max(len(gem_right_grok_wrong),1):.1f}%)")

        # G8: Boundary ambiguity — errors near CR/Amb boundary
        print(f"\n  G8: CR↔Amb boundary analysis")
        cr_amb_errors = [i for i in range(n)
                         if gold[i] != ens_cl[i]
                         and set([gold[i], ens_cl[i]]) ==
                         {"Clear Reply", "Ambivalent"}]
        cr_amb_total = len(cr_amb_errors)
        all_errors = sum(1 for i in range(n) if gold[i] != ens_cl[i])
        print(f"    CR↔Amb errors: {cr_amb_total}/{all_errors} "
              f"({100*cr_amb_total/max(all_errors,1):.1f}% of all errors)")

        # Among CR↔Amb errors, what's the gold evasion?
        if cr_amb_errors:
            cr_amb_gold_ev = Counter()
            for i in cr_amb_errors:
                if gold_ev_multi[i]:
                    cr_amb_gold_ev[gold_ev_multi[i][0]] += 1
            print(f"    Gold evasion of CR↔Amb errors:")
            for ev_lab, cnt in cr_amb_gold_ev.most_common():
                print(f"      {ev_lab}: {cnt}")

        # G9: Errors correlated with annotator agreement
        print(f"\n  G9: Error rate by annotator agreement")
        if set_name == "TEST":
            for ag_level in ["unanimous", "majority"]:
                indices = [i for i, s in enumerate(data)
                           if s.get("gold", {}).get("clarity_agreement") == ag_level]
                if not indices:
                    continue
                errors = sum(1 for i in indices if ens_cl[i] != gold[i])
                err_rate = errors / len(indices)
                print(f"    {ag_level}: {errors}/{len(indices)} "
                      f"({100*err_rate:.1f}% error rate)")
        else:
            unan_idx = [i for i, labels in enumerate(gold_ev_multi)
                        if len(set(labels)) == 1]
            disag_idx = [i for i, labels in enumerate(gold_ev_multi)
                         if len(set(labels)) > 1]
            for label, indices in [("Evasion unanimous", unan_idx),
                                   ("Evasion disagree", disag_idx)]:
                if not indices:
                    continue
                errors = sum(1 for i in indices if ens_cl[i] != gold[i])
                print(f"    {label}: {errors}/{len(indices)} "
                      f"({100*errors/len(indices):.1f}% error rate)")

        # G10: Hardest samples (wrong for ALL approaches)
        print(f"\n  G10: Hardest samples")
        hard_indices = [i for i in range(n)
                        if grok_cl[i] != gold[i]
                        and gem_cl[i] != gold[i]
                        and ens_cl[i] != gold[i]]
        print(f"    Universally hard: {len(hard_indices)}/{n}")

        # What classes are these?
        hard_gold = Counter(gold[i] for i in hard_indices)
        print(f"    Gold distribution of hard samples: {dict(hard_gold)}")

        # What do models predict for these?
        hard_grok = Counter(grok_cl[i] for i in hard_indices)
        hard_gem = Counter(gem_cl[i] for i in hard_indices)
        print(f"    Grok predictions: {dict(hard_grok)}")
        print(f"    Gemini predictions: {dict(hard_gem)}")

        # Average answer length of hard vs easy
        if hard_indices:
            hard_ans_len = sum(answer_lengths[i] for i in hard_indices) / len(hard_indices)
            easy_indices = [i for i in range(n) if i not in hard_indices]
            easy_ans_len = sum(answer_lengths[i] for i in easy_indices) / max(len(easy_indices), 1)
            print(f"    Avg answer length: hard={hard_ans_len:.0f} words, "
                  f"easy={easy_ans_len:.0f} words")

        results[set_name.lower()] = findings

    return results


# SECTION H: Thinking Tokens Analysis (both sets)

def section_h(test_data, eval_data, test_gold, eval_gold):
    """Section H: Gemini thinking tokens analysis."""
    print("\n" + "=" * 70)
    print("  SECTION H: THINKING TOKENS ANALYSIS")
    print("=" * 70)

    results = {}

    for set_name, data, gold in [("EVAL", eval_data, eval_gold),
                                  ("TEST", test_data, test_gold)]:
        print(f"\n{'─' * 70}")
        print(f"  {set_name} SET (n={len(data)})")
        print(f"{'─' * 70}")

        gold_ev_multi = get_gold_evasion_multi(data, set_name.lower())
        ens_cl = [get_ensemble_clarity(s) for s in data]

        # H1: Gemini thinking token usage stats
        print(f"\n  H1: Gemini thinking token usage")
        all_thoughts = []
        all_completion = []
        all_total = []
        per_sample_avg_thoughts = {}  # keyed by sample index

        for i, s in enumerate(data):
            gem_responses = s.get("gemini_round0", {}).get("responses", [])
            sample_thoughts = []
            for r in gem_responses:
                usage = r.get("usage", {})
                tt = usage.get("thoughts_tokens", 0) or 0
                ct = usage.get("completion_tokens", 0) or 0
                tot = usage.get("total_tokens", 0) or 0
                all_thoughts.append(tt)
                all_completion.append(ct)
                all_total.append(tot)
                sample_thoughts.append(tt)
            if sample_thoughts:
                per_sample_avg_thoughts[i] = (
                    sum(sample_thoughts) / len(sample_thoughts))

        if all_thoughts:
            avg_t = sum(all_thoughts) / len(all_thoughts)
            avg_c = sum(all_completion) / len(all_completion)
            avg_tot = sum(all_total) / len(all_total)
            min_t = min(all_thoughts)
            max_t = max(all_thoughts)
            sorted_t = sorted(all_thoughts)
            med_t = sorted_t[len(sorted_t)//2]
            ratio = avg_t / max(avg_c, 1)

            print(f"    Responses: {len(all_thoughts)}")
            print(f"    Thinking tokens:  mean={avg_t:.0f}  "
                  f"median={med_t}  min={min_t}  max={max_t}")
            print(f"    Completion tokens: mean={avg_c:.0f}")
            print(f"    Total tokens:     mean={avg_tot:.0f}")
            print(f"    Think/Complete ratio: {ratio:.2f}x")
            print(f"    Think % of total: "
                  f"{100*avg_t/max(avg_tot,1):.1f}%")

        # H2: Thinking tokens vs accuracy
        print(f"\n  H2: Thinking tokens vs accuracy")
        if per_sample_avg_thoughts:
            # Split into quartiles (only samples that have thinking tokens)
            valid_indices = sorted(per_sample_avg_thoughts.keys(),
                                   key=lambda i: per_sample_avg_thoughts[i])
            sorted_samples = valid_indices
            q_size = len(sorted_samples) // 4

            for qi, q_name in enumerate(["Q1 (fewest)", "Q2", "Q3",
                                          "Q4 (most)"]):
                start = qi * q_size
                end = start + q_size if qi < 3 else len(sorted_samples)
                q_indices = sorted_samples[start:end]
                if not q_indices:
                    continue
                q_correct = sum(1 for i in q_indices if ens_cl[i] == gold[i])
                q_acc = q_correct / len(q_indices)
                q_avg_t = sum(per_sample_avg_thoughts[i]
                              for i in q_indices) / len(q_indices)
                print(f"    {q_name:<15s}: n={len(q_indices)}  "
                      f"avg_think={q_avg_t:.0f}  "
                      f"clarity_acc={100*q_acc:.1f}%")

            # Per-evasion correct/wrong comparison
            print(f"\n    Thinking tokens: correct vs wrong responses")
            correct_thoughts = []
            wrong_thoughts = []
            for i, s in enumerate(data):
                gem_responses = s.get("gemini_round0", {}).get("responses", [])
                for r in gem_responses:
                    tt = r.get("usage", {}).get("thoughts_tokens", 0) or 0
                    label = r.get("label", "")
                    if not label:
                        continue
                    if label in gold_ev_multi[i]:
                        correct_thoughts.append(tt)
                    else:
                        wrong_thoughts.append(tt)

            if correct_thoughts and wrong_thoughts:
                avg_correct = sum(correct_thoughts) / len(correct_thoughts)
                avg_wrong = sum(wrong_thoughts) / len(wrong_thoughts)
                print(f"    Correct responses: mean={avg_correct:.0f} "
                      f"(n={len(correct_thoughts)})")
                print(f"    Wrong responses:   mean={avg_wrong:.0f} "
                      f"(n={len(wrong_thoughts)})")
                print(f"    Delta: {avg_wrong - avg_correct:+.0f} tokens")

                # Cohen's d
                import statistics
                if len(correct_thoughts) > 1 and len(wrong_thoughts) > 1:
                    sd_c = statistics.stdev(correct_thoughts)
                    sd_w = statistics.stdev(wrong_thoughts)
                    pooled_sd = math.sqrt(
                        (sd_c**2 * (len(correct_thoughts)-1) +
                         sd_w**2 * (len(wrong_thoughts)-1)) /
                        (len(correct_thoughts) + len(wrong_thoughts) - 2))
                    if pooled_sd > 0:
                        d = (avg_wrong - avg_correct) / pooled_sd
                        print(f"    Cohen's d: {d:.3f}")

        # Grok reasoning tokens (total - prompt - completion)
        print(f"\n    Grok reasoning overhead:")
        grok_reasoning = []
        for s in data:
            grok_responses = s.get("grok_round0", {}).get("responses", [])
            for r in grok_responses:
                usage = r.get("usage", {})
                total = usage.get("total_tokens", 0) or 0
                prompt = usage.get("prompt_tokens", 0) or 0
                completion = usage.get("completion_tokens", 0) or 0
                reasoning = total - prompt - completion
                if reasoning > 0:
                    grok_reasoning.append(reasoning)
        if grok_reasoning:
            avg_gr = sum(grok_reasoning) / len(grok_reasoning)
            print(f"    Grok reasoning tokens: mean={avg_gr:.0f} "
                  f"(n={len(grok_reasoning)})")
        else:
            print(f"    Grok: no reasoning overhead detected")

        results[set_name.lower()] = {"avg_thoughts": avg_t if all_thoughts else 0}

    return results


# Main

def main():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace")

    print("CLARITY Paper: Comprehensive Analysis")
    print("=" * 70)
    print(f"Loading data from: {BASE_DIR}")

    # Load data
    eval_data = load_detailed_json("stage2_eval_set_detailed.json")
    test_data = load_detailed_json("stage2_test_set_detailed.json")
    print(f"  Eval set: {len(eval_data)} samples")
    print(f"  Test set: {len(test_data)} samples")

    # Get gold labels
    eval_gold = get_gold_clarity(eval_data, "eval")
    test_gold = get_gold_clarity(test_data, "test")
    print(f"  Eval gold: {Counter(eval_gold)}")
    print(f"  Test gold: {Counter(test_gold)}")

    # Run sections
    all_results = {}
    all_results["A"] = section_a(test_data, eval_data, test_gold, eval_gold)
    all_results["B"] = section_b(test_data, eval_data, test_gold, eval_gold)
    all_results["C"] = section_c(test_data, eval_data, test_gold, eval_gold)
    all_results["D"] = section_d(test_data, test_gold)
    all_results["E"] = section_e(test_data, test_gold)
    all_results["F"] = section_f(test_data, eval_data, test_gold, eval_gold)
    all_results["G"] = section_g(test_data, eval_data, test_gold, eval_gold)
    all_results["H"] = section_h(test_data, eval_data, test_gold, eval_gold)

    # Save results
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "results_ALL.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()

