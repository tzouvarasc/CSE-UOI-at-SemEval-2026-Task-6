# -*- coding: utf-8 -*-
"""
Hu et al. (2025) multi-agent debate 
"""

import os
import json
import csv
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# Hu et al. hyperparameters

N_AGENTS = 7
MAX_ROUNDS = 10
TEMPERATURE = 1.0
MAX_MODEL_LENGTH = 16000
KS_THRESHOLD = 0.05
KS_CONSECUTIVE = 2

MAX_TOKENS = 4000
MAX_JUDGE_RESPONSE_CHARS = 2000
CLARITIES = sorted(["Ambivalent", "Clear Non-Reply", "Clear Reply"])


EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "Deflection",
    "Partial/half-answer",
    "General",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
]

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


# Grok client (reused)

class GrokClient:
    """Wrapper for xAI Grok 4.1 Fast Reasoning API."""

    def __init__(self, api_key: str, max_retries: int = 3):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model = "grok-4-1-fast-reasoning"
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str, max_tokens: int = 1500, temperature: float = 0.4) -> str:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_error = e
                print(f"  [Grok retry {attempt+1}/{self.max_retries}]: {e}")
        print(f"  [Grok FAILED]: {last_error}")
        return ""


# Classification prompt (reused exactly)

COT_EVASION_SYSTEM_PROMPT = """You are an expert at classifying political interview responses for clarity and evasion.

## EVASION TAXONOMY (9 categories):

1. **EXPLICIT**: The answer DIRECTLY states the requested information.
   - Contains yes/no, specific facts, numbers, or clear commitments
   - The EXACT information requested is provided
   - Example: Q: "Will you raise taxes?" A: "No, I will not raise taxes."

2. **IMPLICIT**: The answer is ON-TOPIC and you can INFER the answer from context.
   - MUST be about the SAME TOPIC as the question
   - Not stated directly, but a reasonable listener can deduce the answer
   - Example: Q: "Do you support the bill?" A: "I've always stood with working families on this issue." (implies support for THIS bill)

3. **PARTIAL/HALF-ANSWER**: Multi-part question where SOME parts are answered, others ignored.
   - Only applies to questions with multiple distinct parts
   - Example: Q: "Will you raise taxes and cut spending?" A: "We won't cut spending." (ignores taxes part)

4. **GENERAL**: ON-TOPIC but TOO VAGUE - empty platitudes with NO specifics.
   - MUST be about the SAME TOPIC as the question
   - You CANNOT infer any concrete answer
   - Example: Q: "What will you do about inflation?" A: "We're committed to economic stability."

5. **DODGING**: Answer is OFF-TOPIC - talks about DIFFERENT subject than what was asked.
   - THIS IS THE MOST IMPORTANT CHECK: Does the answer address WHAT WAS ASKED?
   - If the question asks about X but the answer talks about Y → DODGING
   - Example: Q: "Did you meet the lobbyist?" A: "Let me tell you about our infrastructure plan."
   - Example: Q: "Do you owe an apology?" A: "Let me talk about the PATRIOT Act..." → DODGING
   - Example: Q: "Is this negotiable?" A: "We must prevent a vacuum..." → DODGING (different topic)

6. **DEFLECTION**: Stays ON-TOPIC but SHIFTS BLAME to others.
   - Points fingers at opposition, predecessor, external factors
   - MUST still be about the same topic as the question
   - Example: Q: "Why did YOUR project fail?" A: "The previous administration left us a mess."

7. **DECLINING TO ANSWER**: EXPLICIT refusal to answer.
   - "No comment", "I won't discuss", "I can't speak to that"
   - Example: Q: "Can you confirm...?" A: "I'm not going to comment on ongoing investigations."

8. **CLAIMS IGNORANCE**: Says they DON'T KNOW the answer.
   - "I don't know", "I'm not aware", "I'll have to check"
   - Example: Q: "When did this happen?" A: "I don't have that date. I'll get back to you."

9. **CLARIFICATION**: Asks question BACK instead of answering.
   - "What do you mean by...?", "Are you asking about...?"
   - Example: Q: "Was it your decision?" A: "You mean the public fund?"

## CRITICAL DISTINCTION: DODGING vs IMPLICIT vs GENERAL

The #1 mistake is confusing these three. Here's the key test:

| Question | If answer talks about... | Label |
|----------|-------------------------|-------|
| "Will you do X?" | Something OTHER than X | **DODGING** (off-topic) |
| "Will you do X?" | X, but in a way you can infer the answer | **IMPLICIT** |
| "Will you do X?" | X, but too vague to infer anything | **GENERAL** |

**DODGING examples (answer talks about DIFFERENT subject):**
- Q: "Do you owe an apology?" A: "Let me discuss the PATRIOT Act..." → DODGING (apology ≠ PATRIOT Act)
- Q: "Is this negotiable?" A: "We can't create a vacuum for Hezbollah..." → DODGING (negotiable ≠ vacuum)
- Q: "Do walls feel like closing in?" A: "Senator Warner said..." → DODGING (feelings ≠ Senator Warner)

**IMPLICIT examples (on-topic, inferable):**
- Q: "Do you support the bill?" A: "I've always supported working families." → IMPLICIT (on-topic, implies yes)

**GENERAL examples (on-topic, no inference possible):**
- Q: "What's your plan?" A: "We will use all tools available." → GENERAL (on-topic, but too vague)

## YOUR TASK:
Classify the given Q&A pair. FIRST check if the answer is ON-TOPIC or OFF-TOPIC.

## OUTPUT FORMAT (follow exactly):
STEP1_QUESTION_TOPIC: <What SPECIFIC topic/subject does the question ask about?>
STEP2_ANSWER_TOPIC: <What topic/subject does the answer ACTUALLY discuss?>
STEP3_TOPIC_MATCH: <Do these topics MATCH? YES = on-topic (Explicit/Implicit/General/Partial/Deflection possible), NO = off-topic (likely DODGING)>
STEP4_DIRECT_CHECK: <If topics match: Is there a DIRECT yes/no or specific answer? If yes → Explicit>
STEP5_INFERENCE_CHECK: <If topics match: Can you INFER an answer? If yes → Implicit. If no → General>
STEP6_REFUSAL_CHECK: <Is there explicit refusal, ignorance claim, or clarification request? → Clear Non-Reply category>
STEP7_BLAME_CHECK: <If topics match: Does the answer shift blame to others? If yes → Deflection>
STEP8_MULTI_PART_CHECK: <If multi-part question, are ALL parts addressed? If some missing → Partial>
FINAL_LABEL: <One of: Explicit, Implicit, Partial/half-answer, General, Dodging, Deflection, Declining to answer, Claims ignorance, Clarification>
CONFIDENCE: <1-5, where 5 is highest confidence>"""


DEBATE_HISTORY_SECTION = """

## OTHER JUDGES' EVALUATIONS
Several other independent judges have classified this same Q&A pair. Below are their evaluations from the previous round:

{judge_responses}

Consider the other judges' reasoning carefully, but make your own independent assessment. You may maintain or change your classification. Only change if you find a genuine reason to — do not change merely because others disagree.
"""


# Utility functions (reused)

def normalize_label(label_text: str, allow_fallback: bool = True) -> str:
    """Normalize a label string to match official taxonomy."""
    if not label_text:
        return "General" if allow_fallback else ""

    label_text = label_text.strip().strip("*").strip('"').strip("'").strip()

    if not label_text:
        return "General" if allow_fallback else ""

    # Exact match
    for official_label in EVASION_LABELS:
        if label_text.lower() == official_label.lower():
            return official_label

    # Prefix match
    label_lower = label_text.lower()
    for official_label in EVASION_LABELS:
        if label_lower.startswith(official_label.lower()):
            return official_label

    # Special cases
    if "partial/half" in label_lower or "half-answer" in label_lower or "half answer" in label_lower:
        return "Partial/half-answer"
    if "declining to answer" in label_lower or "decline to answer" in label_lower:
        return "Declining to answer"
    if "claims ignorance" in label_lower or "claim ignorance" in label_lower:
        return "Claims ignorance"

    # Word-based matching
    words = set(label_lower.split())
    if "explicit" in words or label_lower == "explicit":
        return "Explicit"
    if "implicit" in words or label_lower == "implicit":
        return "Implicit"
    if "dodging" in words or label_lower == "dodging":
        return "Dodging"
    if "deflection" in words or label_lower == "deflection":
        return "Deflection"
    if "clarification" in words or label_lower == "clarification":
        return "Clarification"
    if "partial" in words:
        return "Partial/half-answer"
    if "general" in words and label_lower.strip() == "general":
        return "General"

    # Substring matching
    if "dodg" in label_lower:
        return "Dodging"
    if "deflect" in label_lower:
        return "Deflection"
    if "declin" in label_lower or "refus" in label_lower:
        return "Declining to answer"
    if "ignoran" in label_lower:
        return "Claims ignorance"
    if "clarif" in label_lower:
        return "Clarification"

    return "General" if allow_fallback else ""


def extract_label_from_response(response: str, field_name: str = "FINAL_LABEL") -> str:
    """Extract and normalize a label from a structured response."""
    if not response:
        return ""

    # Look for the specific field
    for line in response.split("\n"):
        line_stripped = line.strip()
        if field_name.lower() in line_stripped.lower():
            if ":" in line_stripped:
                parts = line_stripped.split(":", 1)
                if len(parts) > 1:
                    raw_label = parts[1].strip()
                    result = normalize_label(raw_label, allow_fallback=False)
                    if result:
                        return result

    # Try any label-like field
    label_keywords = ["LABEL", "FINAL_LABEL", "CHOICE", "ANSWER"]
    for line in response.split("\n"):
        line_stripped = line.strip()
        for keyword in label_keywords:
            if keyword in line_stripped.upper() and ":" in line_stripped:
                parts = line_stripped.split(":", 1)
                if len(parts) > 1:
                    raw_label = parts[1].strip()
                    result = normalize_label(raw_label, allow_fallback=False)
                    if result:
                        return result

    # Last resort: check if any line IS a label
    for line in response.split("\n"):
        line_stripped = line.strip()
        for official_label in EVASION_LABELS:
            if line_stripped.lower() == official_label.lower():
                return official_label

    return ""


def extract_confidence(response: str) -> int:
    """Extract confidence score (1-5) from response."""
    for line in response.split("\n"):
        if "CONFIDENCE" in line.upper() and ":" in line:
            parts = line.split(":", 1)
            if len(parts) > 1:
                try:
                    conf = int(parts[1].strip().split()[0])
                    return min(max(conf, 1), 5)
                except Exception:
                    pass
    return 3  # Default medium confidence


def extract_gold_labels(example: dict) -> List[str]:
    """Extract gold evasion labels from dataset example."""
    gold_labels = []
    for ann_key in ("annotator1", "annotator2", "annotator3"):
        lab = example.get(ann_key)
        if isinstance(lab, str) and lab.strip():
            gold_labels.append(lab.strip())

    if not gold_labels and "evasion_label" in example:
        lab = example["evasion_label"]
        if isinstance(lab, str) and lab.strip():
            gold_labels = [lab.strip()]

    return gold_labels


def f1_for_class(gold_annotations: List[List[str]], predictions: List[str], target_class: str) -> dict:
    """Calculate F1 score for a specific class (handles multi-annotator gold)."""
    TP = FP = FN = 0

    for gold, pred in zip(gold_annotations, predictions):
        gold_set = set(gold)

        if pred == target_class and target_class in gold_set:
            TP += 1
        elif pred == target_class and target_class not in gold_set:
            FP += 1
        elif target_class in gold_set and pred not in gold_set:
            FN += 1

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": TP, "fp": FP, "fn": FN}


def compute_macro_f1(gold_annotations: List[List[str]], predictions: List[str]) -> Tuple[float, List[str]]:
    """Compute macro F1 score (official Task 2 metric)."""
    classes = sorted(set(str(lbl) for row in gold_annotations for lbl in row if str(lbl).strip()))
    f1s = []
    for cls in classes:
        stats = f1_for_class(gold_annotations, predictions, cls)
        f1s.append(stats["f1"])
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    return macro_f1, classes


def compute_instance_accuracy(gold_annotations: List[List[str]], predictions: List[str]) -> float:
    """Compute instance accuracy (pred in gold set)."""
    correct = sum(1 for gold, pred in zip(gold_annotations, predictions) if pred in set(gold))
    return correct / len(gold_annotations) if gold_annotations else 0.0


# Hu debate helpers

def build_user_prompt(question: str, answer: str, full_question: Optional[str] = None) -> str:
    if full_question and full_question != question:
        return (
            f"FULL INTERVIEW QUESTION: {full_question}\n"
            f"SPECIFIC SUB-QUESTION: {question}\n"
            f"ANSWER: {answer}"
        )
    return f"QUESTION: {question}\nANSWER: {answer}"


def compute_ks_statistic(prev_results: List[str], curr_results: List[str]) -> float:
    """
    KS statistic between two rounds' clarity label distributions.

    prev_results, curr_results: lists of clarity labels (length 7 each)
    Returns: float (max CDF difference)
    """
    n = len(prev_results)
    if n == 0 or len(curr_results) == 0:
        return 1.0

    prev_counts = Counter(prev_results)
    curr_counts = Counter(curr_results)

    prev_cdf = 0.0
    curr_cdf = 0.0
    max_diff = 0.0

    for c in CLARITIES:
        prev_cdf += prev_counts.get(c, 0) / n
        curr_cdf += curr_counts.get(c, 0) / n
        max_diff = max(max_diff, abs(prev_cdf - curr_cdf))

    return max_diff


def check_unanimous(clarity_labels: List[str]) -> Tuple[bool, Optional[str]]:
    """Hu et al.: ALL 7 agents must agree. Returns (converged: bool, label: str|None)."""
    if len(set(clarity_labels)) == 1 and clarity_labels:
        return True, clarity_labels[0]
    return False, None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if pd is not None and pd.isna(value):
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return str(value)


def _truncate_response(response: str, max_chars: int = MAX_JUDGE_RESPONSE_CHARS) -> str:
    if len(response) <= max_chars:
        return response
    return response[:max_chars] + "\n...[truncated]"


def _clarity_distribution(round_results: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(r["clarity"] for r in round_results))


def _pick_evasion_for_clarity(
    results: List[Dict[str, Any]],
    target_clarity: str,
    history: Optional[List[List[Dict[str, Any]]]] = None,
) -> str:
    evasion_labels = [r["evasion"] for r in results if r["clarity"] == target_clarity]
    if not evasion_labels:
        evasion_labels = [r["evasion"] for r in results]

    evasion_counts = Counter(evasion_labels)
    if not evasion_counts:
        return "General"

    top_count = max(evasion_counts.values())
    winners = [label for label, count in evasion_counts.items() if count == top_count]
    if len(winners) == 1:
        return winners[0]

    if history:
        hist_counts = Counter()
        for round_results in history:
            for record in round_results:
                if EVASION_TO_CLARITY.get(record["evasion"], "Ambivalent") == target_clarity:
                    hist_counts[record["evasion"]] += 1
        if hist_counts:
            hist_top = max(hist_counts.get(label, 0) for label in winners)
            hist_winners = [label for label in winners if hist_counts.get(label, 0) == hist_top]
            if len(hist_winners) == 1:
                return hist_winners[0]
            return sorted(hist_winners)[0]

    return sorted(winners)[0]


def majority_vote_last_round(
    results: List[Dict[str, Any]],
    history: Optional[List[List[Dict[str, Any]]]] = None,
) -> Tuple[str, str]:
    """Hu et al.: majority vote of the LAST round ONLY, with requested tie-breakers."""
    clarity_labels = [r["clarity"] for r in results]
    counts = Counter(clarity_labels)
    if not counts:
        return "General", "Ambivalent"

    top_count = max(counts.values())
    top_clarities = [label for label, cnt in counts.items() if cnt == top_count]

    if len(top_clarities) == 1:
        winning_clarity = top_clarities[0]
    else:
        historical = Counter()
        if history:
            for round_results in history:
                historical.update(r["clarity"] for r in round_results)

        if historical:
            historical_top = max(historical.get(label, 0) for label in top_clarities)
            narrowed = [label for label in top_clarities if historical.get(label, 0) == historical_top]
        else:
            narrowed = top_clarities

        if len(narrowed) == 1:
            winning_clarity = narrowed[0]
        else:
            winning_clarity = "Ambivalent"

    winning_evasion = _pick_evasion_for_clarity(results, winning_clarity, history)
    return winning_evasion, winning_clarity


# Hu classifier

class HuFaithfulDebateClassifier:
    """
    Faithful implementation of Hu et al. (2025) multi-agent debate
    adapted for political evasion classification.

    Key differences from Debate:
    - 7 homogeneous Grok agents (not 2 heterogeneous models)
    - 1 response per agent per round (not k=5 self-consistency)
    - Unanimous convergence (not majority-label agreement)
    - Last-round majority vote fallback (not recency-weighted)
    - Full response history shown to each agent (not CoT summary)
    - Temperature 1.0 uniform (not 0.4/1.0 mixed)
    """

    def __init__(
        self,
        grok_client: GrokClient,
        n_agents: int = N_AGENTS,
        max_rounds: int = MAX_ROUNDS,
        ks_threshold: float = KS_THRESHOLD,
        ks_consecutive: int = KS_CONSECUTIVE,
    ):
        self.grok_client = grok_client
        self.n_agents = n_agents
        self.max_rounds = max_rounds
        self.ks_threshold = ks_threshold
        self.ks_consecutive = ks_consecutive

    def _run_single_agent(self, agent_id: int, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        response = self.grok_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,  # Must override Grok default 0.4
        )

        evasion_label = extract_label_from_response(response, field_name="FINAL_LABEL")
        if not evasion_label:
            evasion_label = "General"

        clarity_label = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
        confidence = extract_confidence(response)

        return {
            "agent_id": agent_id,
            "response": response,
            "evasion": evasion_label,
            "clarity": clarity_label,
            "confidence": confidence,
        }

    def _run_round_parallel(self, system_prompts: Dict[int, str], user_prompt: str) -> List[Dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            futures = {
                executor.submit(self._run_single_agent, agent_id, prompt, user_prompt): agent_id
                for agent_id, prompt in system_prompts.items()
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"  [Agent {agent_id} failed]: {exc}")
                    results.append(
                        {
                            "agent_id": agent_id,
                            "response": "",
                            "evasion": "General",
                            "clarity": EVASION_TO_CLARITY["General"],
                            "confidence": 3,
                        }
                    )

        results.sort(key=lambda x: x["agent_id"])
        return results

    def _format_other_judges(self, previous_round: List[Dict[str, Any]], current_agent_id: int) -> str:
        blocks = []
        judge_num = 1
        for agent_result in previous_round:
            if agent_result["agent_id"] == current_agent_id:
                continue
            response = _truncate_response(agent_result.get("response", ""))
            blocks.append(f"Judge {judge_num}:\n{response}")
            judge_num += 1
        return "\n---\n".join(blocks)

    def classify(
        self,
        question: str,
        answer: str,
        full_question: Optional[str] = None,
        sample_index: Optional[int] = None,
        total_samples: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        user_prompt = build_user_prompt(question, answer, full_question)
        sample_prefix = ""
        if sample_index is not None and total_samples is not None:
            sample_prefix = f"[Sample {sample_index + 1}/{total_samples}] "

        if verbose:
            print(f"{sample_prefix}Round 0: running {self.n_agents} independent agents")

        # ROUND 0
        round0_prompts = {agent_id: COT_EVASION_SYSTEM_PROMPT for agent_id in range(self.n_agents)}
        round0_results = self._run_round_parallel(round0_prompts, user_prompt)
        history = [round0_results]

        round0_clarity = [r["clarity"] for r in round0_results]
        round0_unanimous, round0_unanimous_label = check_unanimous(round0_clarity)

        round0_log = {
            "agents": [
                {
                    "agent_id": r["agent_id"],
                    "response": r["response"],
                    "evasion_label": r["evasion"],
                    "clarity_label": r["clarity"],
                }
                for r in round0_results
            ],
            "clarity_distribution": _clarity_distribution(round0_results),
            "unanimous": round0_unanimous,
        }

        if verbose:
            print(
                f"{sample_prefix}Round 0 status: unanimous={round0_unanimous}, "
                f"distribution={round0_log['clarity_distribution']}"
            )

        if round0_unanimous and round0_unanimous_label is not None:
            final_evasion = _pick_evasion_for_clarity(round0_results, round0_unanimous_label, history)
            if verbose:
                print(
                    f"{sample_prefix}Final: {final_evasion} -> {round0_unanimous_label} "
                    f"(stop=unanimous, decision=unanimous_round0)"
                )
            return {
                "round0": round0_log,
                "debate_rounds": [],
                "debate_triggered": False,
                "total_rounds": 1,
                "stop_reason": "unanimous",
                "final_evasion": final_evasion,
                "final_clarity": round0_unanimous_label,
                "decision_method": "unanimous_round0",
                "total_api_calls": self.n_agents,
            }

        # DEBATE ROUNDS
        debate_rounds_log = []
        below_threshold_consecutive = 0
        stop_reason = "max_rounds"
        unanimous_round_label = None

        for round_idx in range(1, self.max_rounds + 1):
            if verbose:
                print(f"{sample_prefix}Round {round_idx}: debating with prior responses")

            prev_round = history[-1]
            debate_prompts = {}
            for agent_id in range(self.n_agents):
                judge_responses = self._format_other_judges(prev_round, agent_id)
                debate_prompts[agent_id] = COT_EVASION_SYSTEM_PROMPT + DEBATE_HISTORY_SECTION.format(
                    judge_responses=judge_responses
                )

            curr_results = self._run_round_parallel(debate_prompts, user_prompt)
            history.append(curr_results)

            prev_by_agent = {r["agent_id"]: r for r in prev_round}
            curr_clarity = [r["clarity"] for r in curr_results]
            prev_clarity = [r["clarity"] for r in prev_round]

            ks_stat = compute_ks_statistic(prev_clarity, curr_clarity)
            if ks_stat < self.ks_threshold:
                below_threshold_consecutive += 1
            else:
                below_threshold_consecutive = 0

            unanimous, unanimous_label = check_unanimous(curr_clarity)
            round_log = {
                "round": round_idx,
                "agents": [
                    {
                        "agent_id": r["agent_id"],
                        "response": r["response"],
                        "evasion_label": r["evasion"],
                        "clarity_label": r["clarity"],
                        "changed": r["evasion"] != prev_by_agent[r["agent_id"]]["evasion"],
                    }
                    for r in curr_results
                ],
                "clarity_distribution": _clarity_distribution(curr_results),
                "unanimous": unanimous,
                "ks_statistic": round(ks_stat, 6),
            }
            debate_rounds_log.append(round_log)

            if verbose:
                print(
                    f"{sample_prefix}Round {round_idx} status: unanimous={unanimous}, "
                    f"ks={ks_stat:.6f}, below_threshold_consecutive={below_threshold_consecutive}"
                )

            if unanimous:
                stop_reason = "unanimous"
                unanimous_round_label = unanimous_label
                break

            if below_threshold_consecutive >= self.ks_consecutive:
                stop_reason = "adaptive_stability"
                break

        last_round_results = history[-1]
        if stop_reason == "unanimous" and unanimous_round_label is not None:
            final_clarity = unanimous_round_label
            final_evasion = _pick_evasion_for_clarity(last_round_results, final_clarity, history)
            decision_method = f"unanimous_round{len(history) - 1}"
        else:
            final_evasion, final_clarity = majority_vote_last_round(last_round_results, history)
            decision_method = "majority_vote_last_round"

        if verbose:
            print(
                f"{sample_prefix}Final: {final_evasion} -> {final_clarity} "
                f"(stop={stop_reason}, decision={decision_method})"
            )

        return {
            "round0": round0_log,
            "debate_rounds": debate_rounds_log,
            "debate_triggered": True,
            "total_rounds": len(history),
            "stop_reason": stop_reason,
            "final_evasion": final_evasion,
            "final_clarity": final_clarity,
            "decision_method": decision_method,
            "total_api_calls": len(history) * self.n_agents,
        }


# Main

def _build_metrics(
    results: List[Dict[str, Any]],
    pred_evasion: List[str],
    pred_clarity: List[str],
    n_agents: int,
    max_rounds: int,
) -> Dict[str, Any]:
    gold_evasion = []
    pred_evasion_gold = []
    gold_clarity = []
    pred_clarity_gold = []

    for result in results:
        gold_block = result.get("gold", {}) or {}
        labels = gold_block.get("evasion_labels", []) or []
        gold_clarity_label = gold_block.get("clarity_label")

        if labels:
            gold_evasion.append(labels)
            pred_evasion_gold.append(result["final_evasion"])

        if isinstance(gold_clarity_label, str) and gold_clarity_label.strip():
            gold_clarity.append([gold_clarity_label.strip()])
            pred_clarity_gold.append(result["final_clarity"])

    if not gold_evasion and not gold_clarity:
        return {
            "has_gold_labels": False,
            "samples_with_gold": 0,
            "samples_processed": len(results),
            "prediction_distribution_evasion": dict(Counter(pred_evasion)),
            "prediction_distribution_clarity": dict(Counter(pred_clarity)),
            "config": {
                "n_agents": n_agents,
                "max_rounds": max_rounds,
                "temperature": TEMPERATURE,
                "max_model_length": MAX_MODEL_LENGTH,
                "ks_threshold": KS_THRESHOLD,
                "ks_consecutive": KS_CONSECUTIVE,
                "grok_model": "grok-4-1-fast-reasoning",
            },
        }

    evasion_metrics: Optional[Dict[str, Any]]
    if gold_evasion:
        evasion_macro_f1, evasion_classes = compute_macro_f1(gold_evasion, pred_evasion_gold)
        evasion_per_class = {cls: f1_for_class(gold_evasion, pred_evasion_gold, cls) for cls in evasion_classes}
        evasion_metrics = {
            "instance_accuracy": compute_instance_accuracy(gold_evasion, pred_evasion_gold),
            "macro_f1": evasion_macro_f1,
            "classes": evasion_classes,
            "per_class": evasion_per_class,
            "prediction_distribution": dict(Counter(pred_evasion_gold)),
        }
    else:
        evasion_metrics = None

    clarity_metrics: Optional[Dict[str, Any]]
    if gold_clarity:
        clarity_macro_f1, clarity_classes = compute_macro_f1(gold_clarity, pred_clarity_gold)
        clarity_per_class = {cls: f1_for_class(gold_clarity, pred_clarity_gold, cls) for cls in clarity_classes}
        clarity_metrics = {
            "instance_accuracy": compute_instance_accuracy(gold_clarity, pred_clarity_gold),
            "macro_f1": clarity_macro_f1,
            "classes": clarity_classes,
            "per_class": clarity_per_class,
            "prediction_distribution": dict(Counter(pred_clarity_gold)),
        }
    else:
        clarity_metrics = None

    return {
        "has_gold_labels": True,
        "samples_with_gold": max(len(gold_evasion), len(gold_clarity)),
        "samples_processed": len(results),
        "evasion": evasion_metrics,
        "clarity": clarity_metrics,
        "config": {
            "n_agents": n_agents,
            "max_rounds": max_rounds,
            "temperature": TEMPERATURE,
            "max_model_length": MAX_MODEL_LENGTH,
            "ks_threshold": KS_THRESHOLD,
            "ks_consecutive": KS_CONSECUTIVE,
            "grok_model": "grok-4-1-fast-reasoning",
        },
    }


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hu et al. (2025) multi-agent debate"
    )
    parser.add_argument(
        "--grok_api_key",
        type=str,
        default=os.environ.get("XAI_API_KEY"),
        help="xAI API key for Grok (or set XAI_API_KEY)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="QEvasion dataset split (e.g., train/test)",
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Path to evaluation CSV (optional, no gold labels required)",
    )
    parser.add_argument(
        "--n_agents",
        type=int,
        default=N_AGENTS,
        help=f"Number of debate agents (default: {N_AGENTS})",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=MAX_ROUNDS,
        help=f"Maximum debate rounds after round 0 (default: {MAX_ROUNDS})",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="debate_llm_judges",
        help="Output prefix for JSON files",
    )
    args = parser.parse_args()

    if not args.grok_api_key:
        raise ValueError("Missing Grok API key. Use --grok_api_key or set XAI_API_KEY.")

    is_eval_csv = args.eval_csv is not None
    if is_eval_csv:
        print(f"Loading eval CSV: {args.eval_csv}")
        if pd is not None:
            df = pd.read_csv(args.eval_csv)
            records = df.to_dict("records")
        else:
            with open(args.eval_csv, "r", encoding="utf-8", newline="") as f:
                records = list(csv.DictReader(f))
    else:
        if load_dataset is None:
            raise ImportError("datasets package is required for --split mode. Install with: pip install datasets")
        print(f"Loading QEvasion dataset split='{args.split}'")
        ds = load_dataset("ailsntua/QEvasion", split=args.split)
        records = [ds[i] for i in range(len(ds))]

    if args.max_samples is not None:
        records = records[: args.max_samples]

    print(f"Loaded {len(records)} samples")
    print(
        f"Config: n_agents={args.n_agents}, max_rounds={args.max_rounds}, "
        f"temperature={TEMPERATURE}, max_model_length={MAX_MODEL_LENGTH}"
    )

    grok_client = GrokClient(args.grok_api_key)
    classifier = HuFaithfulDebateClassifier(
        grok_client=grok_client,
        n_agents=args.n_agents,
        max_rounds=args.max_rounds,
        ks_threshold=KS_THRESHOLD,
        ks_consecutive=KS_CONSECUTIVE,
    )

    all_results: List[Dict[str, Any]] = []
    pred_evasion: List[str] = []
    pred_clarity: List[str] = []
    gold_evasion_running: List[List[str]] = []
    pred_evasion_running: List[str] = []
    gold_clarity_running: List[List[str]] = []
    pred_clarity_running: List[str] = []

    for idx, example in enumerate(records):
        question = _safe_str(example.get("question", ""))
        answer = _safe_str(example.get("interview_answer", ""))
        full_question = _safe_str(example.get("interview_question", ""))
        gold_labels = extract_gold_labels(example)
        raw_gold_clarity = _safe_str(example.get("clarity_label", "")).strip()
        gold_clarity_label = raw_gold_clarity if raw_gold_clarity else None

        print(f"{'='*70}")
        print(f" SAMPLE {idx + 1}/{len(records)}")
        print(f"{'='*70}")
        print(f"   Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")

        trace = classifier.classify(
            question=question,
            answer=answer,
            full_question=full_question,
            sample_index=idx,
            total_samples=len(records),
            verbose=True,
        )

        final_evasion = trace["final_evasion"]
        final_clarity = trace["final_clarity"]

        pred_evasion.append(final_evasion)
        pred_clarity.append(final_clarity)

        correct = final_evasion in set(gold_labels) if gold_labels else None
        clarity_correct = final_clarity == gold_clarity_label if gold_clarity_label else None
        result = {
            "index": idx,
            "question": question,
            "answer": answer,
            "full_question": full_question,
            "gold": {
                "evasion_labels": gold_labels,
                "clarity_label": gold_clarity_label,
            },
            "round0": trace["round0"],
            "debate_rounds": trace["debate_rounds"],
            "debate_triggered": trace["debate_triggered"],
            "total_rounds": trace["total_rounds"],
            "stop_reason": trace["stop_reason"],
            "final_evasion": final_evasion,
            "final_clarity": final_clarity,
            "decision_method": trace["decision_method"],
            "total_api_calls": trace["total_api_calls"],
            "correct": correct,
        }
        all_results.append(result)

        print("")
        print("    RESULT:")
        print(f"      Final decision: {trace['decision_method']}")
        print(f"      Stop reason:    {trace['stop_reason']}")
        print(f"      Total rounds:   {trace['total_rounds']}")
        print(f"      API calls:      {trace['total_api_calls']}")
        print(f"      EVASION:  Pred={final_evasion:<20} Gold={gold_labels if gold_labels else 'N/A'}  "
              f"{'OK' if correct is True else ('WRONG' if correct is False else 'N/A')}")
        gold_clarity_display = gold_clarity_label if gold_clarity_label else "N/A"
        print(f"      CLARITY:  Pred={final_clarity:<20} Gold={gold_clarity_display:<15}  "
              f"{'OK' if clarity_correct is True else ('WRONG' if clarity_correct is False else 'N/A')}")

        if gold_labels:
            gold_evasion_running.append(gold_labels)
            pred_evasion_running.append(final_evasion)
        if gold_clarity_label:
            gold_clarity_running.append([gold_clarity_label])
            pred_clarity_running.append(final_clarity)

        if gold_evasion_running or gold_clarity_running:
            print("")
            print(f"    RUNNING METRICS ({idx + 1} samples):")
            if gold_evasion_running:
                evasion_acc_so_far = compute_instance_accuracy(gold_evasion_running, pred_evasion_running)
                evasion_f1_so_far, _ = compute_macro_f1(gold_evasion_running, pred_evasion_running)
                print(f"      Evasion Macro F1:  {evasion_f1_so_far:.4f}  (Accuracy: {evasion_acc_so_far:.4f})")
            else:
                print("      Evasion Macro F1:  N/A (no gold evasion labels yet)")

            if gold_clarity_running:
                clarity_acc_so_far = compute_instance_accuracy(gold_clarity_running, pred_clarity_running)
                clarity_f1_so_far, _ = compute_macro_f1(gold_clarity_running, pred_clarity_running)
                print(f"      Clarity Macro F1:  {clarity_f1_so_far:.4f}  (Accuracy: {clarity_acc_so_far:.4f})")
            else:
                print("      Clarity Macro F1:  N/A (no gold clarity labels yet)")
        print("")

    metrics = _build_metrics(
        results=all_results,
        pred_evasion=pred_evasion,
        pred_clarity=pred_clarity,
        n_agents=args.n_agents,
        max_rounds=args.max_rounds,
    )

    full_path = f"{args.output_prefix}_full.json"
    metrics_path = f"{args.output_prefix}_metrics.json"
    _ensure_parent_dir(full_path)
    _ensure_parent_dir(metrics_path)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(" FINAL METRICS")
    print("=" * 70)
    if metrics.get("has_gold_labels"):
        evasion_metrics = metrics.get("evasion")
        clarity_metrics = metrics.get("clarity")

        if evasion_metrics:
            print("\n EVASION METRICS (Task 2)")
            print("-" * 40)
            print(f"   Instance Accuracy: {evasion_metrics['instance_accuracy']:.4f}")
            print(f"   Macro F1:          {evasion_metrics['macro_f1']:.4f}")
            print(f"   Classes found:     {evasion_metrics['classes']}")
            print("   Prediction distribution:")
            for label, count in Counter(pred_evasion_running).most_common():
                print(f"     {label}: {count}")
        else:
            print("\n EVASION METRICS (Task 2)")
            print("-" * 40)
            print("   N/A (no gold evasion labels)")

        if clarity_metrics:
            print("\n CLARITY METRICS (Task 1)")
            print("-" * 40)
            print(f"   Instance Accuracy: {clarity_metrics['instance_accuracy']:.4f}")
            print(f"   Macro F1:          {clarity_metrics['macro_f1']:.4f}")
            print(f"   Classes found:     {clarity_metrics['classes']}")
            print("   Prediction distribution:")
            for label, count in Counter(pred_clarity_running).most_common():
                print(f"     {label}: {count}")
        else:
            print("\n CLARITY METRICS (Task 1)")
            print("-" * 40)
            print("   N/A (no gold clarity labels)")
    else:
        print("No gold labels detected. Metrics skipped.")
        print("\nPrediction distribution (Evasion):")
        for label, count in Counter(pred_evasion).most_common():
            print(f"  {label}: {count}")
        print("\nPrediction distribution (Clarity):")
        for label, count in Counter(pred_clarity).most_common():
            print(f"  {label}: {count}")

    print("\n" + "=" * 70)
    print(" SAVING RESULTS")
    print("=" * 70)
    print(f"Saved full logs: {full_path}")
    print(f"Saved metrics:   {metrics_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
