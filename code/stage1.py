# -*- coding: utf-8 -*-
"""
Strategy:
1. Run both Grok and Gemini classifiers with k=5 self-consistency
2. If both agree on clarity → use that 
3. If they disagree → weighted voting: Grok 5 votes + Gemini 4 votes
"""

import os
import json
import pickle
import argparse
import pandas as pd
from collections import Counter
from typing import Any, List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

EVASION_LABELS = [
    "Explicit",
    "Implicit",
    "Partial/half-answer",
    "General",
    "Dodging",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification"
]

EVASION_LABEL_DEFS = {
    "Explicit": "The answer directly addresses the sub-question in a clear, direct way.",
    "Implicit": "The answer addresses the sub-question indirectly, relying on context or implication.",
    "Dodging": "The speaker answers a different question or changes the topic, without addressing the sub-question.",
    "General": "The answer stays at a very high level of generality, with no concrete details that address the sub-question.",
    "Deflection": "The speaker shifts blame or responsibility to someone/something else instead of addressing the sub-question.",
    "Partial/half-answer": "The answer addresses only part of the sub-question and ignores another important part.",
    "Declining to answer": "The speaker explicitly refuses to answer (e.g., 'I won't comment on that').",
    "Claims ignorance": "The speaker says they (or others) do not know the information needed to answer.",
    "Clarification": "The speaker asks for clarification or rephrases the question without giving a substantive answer.",
}

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

# Grok Client

class GrokClient:
    """Wrapper for xAI Grok 4.1 Fast Reasoning API."""

    def __init__(self, api_key: str, max_retries: int = 3):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
        self.model = "grok-4-1-fast-reasoning"
        self.max_retries = max_retries

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 1500, temperature: float = 0.4,
             return_meta: bool = False):
        """Call Grok API with retry logic."""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                text = response.choices[0].message.content.strip()
                if not return_meta:
                    return text

                usage_obj = getattr(response, "usage", None)
                finish_reason = None
                if getattr(response, "choices", None):
                    finish_reason = getattr(response.choices[0], "finish_reason", None)

                usage_meta = {
                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                    "total_tokens": getattr(usage_obj, "total_tokens", None),
                }
                api_meta = {
                    "provider": "grok",
                    "model": self.model,
                    "finish_reason": finish_reason,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                }
                return {"text": text, "usage": usage_meta, "api": api_meta}
            except Exception as e:
                last_error = e
                print(f"  [Grok retry {attempt+1}/{self.max_retries}]: {e}")
        print(f"  [Grok FAILED]: {last_error}")
        if return_meta:
            return {
                "text": "",
                "usage": {},
                "api": {
                    "provider": "grok",
                    "model": self.model,
                    "finish_reason": "ERROR",
                    "attempt": self.max_retries,
                    "retry_count": self.max_retries - 1,
                    "error": str(last_error),
                },
            }
        return ""


# Gemini Client

class GeminiClient:

    def __init__(self, api_key: str, max_retries: int = 3, thinking_level: str = "high"):
        from google import genai
        from google.genai import types
        
        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3-flash-preview"
        self.max_retries = max_retries
        self.thinking_level = thinking_level

    def chat(self, system_prompt: str, user_prompt: str,
             max_tokens: int = 1500, temperature: float = 1.0,
             return_meta: bool = False):

        last_error = None
        for attempt in range(self.max_retries):
            try:

                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=full_prompt,
                    config=self.types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        thinking_config=self.types.ThinkingConfig(
                            thinking_level=self.thinking_level
                        )
                    )
                )
                text = response.text.strip()
                if not return_meta:
                    return text

                usage_obj = getattr(response, "usage_metadata", None)
                candidates = getattr(response, "candidates", None) or []
                finish_reason = None
                if candidates:
                    finish_reason = getattr(candidates[0], "finish_reason", None)
                    if finish_reason is None:
                        finish_reason = getattr(candidates[0], "finishReason", None)

                usage_meta = {
                    "prompt_tokens": getattr(usage_obj, "prompt_token_count", None),
                    "completion_tokens": getattr(usage_obj, "candidates_token_count", None),
                    "total_tokens": getattr(usage_obj, "total_token_count", None),
                    "thoughts_tokens": getattr(usage_obj, "thoughts_token_count", None),
                }

                if usage_meta["prompt_tokens"] is None and usage_obj is not None:
                    usage_meta["prompt_tokens"] = getattr(usage_obj, "promptTokenCount", None)
                if usage_meta["completion_tokens"] is None and usage_obj is not None:
                    usage_meta["completion_tokens"] = getattr(usage_obj, "candidatesTokenCount", None)
                if usage_meta["total_tokens"] is None and usage_obj is not None:
                    usage_meta["total_tokens"] = getattr(usage_obj, "totalTokenCount", None)
                if usage_meta["thoughts_tokens"] is None and usage_obj is not None:
                    usage_meta["thoughts_tokens"] = getattr(usage_obj, "thoughtsTokenCount", None)

                api_meta = {
                    "provider": "gemini",
                    "model": self.model,
                    "finish_reason": finish_reason,
                    "attempt": attempt + 1,
                    "retry_count": attempt,
                }
                return {"text": text, "usage": usage_meta, "api": api_meta}
            except Exception as e:
                last_error = e
                print(f"  [Gemini retry {attempt+1}/{self.max_retries}]: {e}")
        print(f"  [Gemini FAILED]: {last_error}")
        if return_meta:
            return {
                "text": "",
                "usage": {},
                "api": {
                    "provider": "gemini",
                    "model": self.model,
                    "finish_reason": "ERROR",
                    "attempt": self.max_retries,
                    "retry_count": self.max_retries - 1,
                    "error": str(last_error),
                },
            }
        return ""


# Chain-of-Thought Classification Prompt

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

# Label Extraction and Normalization

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
    for line in response.split('\n'):
        line_stripped = line.strip()
        if field_name.lower() in line_stripped.lower():
            if ':' in line_stripped:
                parts = line_stripped.split(':', 1)
                if len(parts) > 1:
                    raw_label = parts[1].strip()
                    result = normalize_label(raw_label, allow_fallback=False)
                    if result:
                        return result

    # Try any label-like field
    label_keywords = ["LABEL", "FINAL_LABEL", "CHOICE", "ANSWER"]
    for line in response.split('\n'):
        line_stripped = line.strip()
        for keyword in label_keywords:
            if keyword in line_stripped.upper() and ':' in line_stripped:
                parts = line_stripped.split(':', 1)
                if len(parts) > 1:
                    raw_label = parts[1].strip()
                    result = normalize_label(raw_label, allow_fallback=False)
                    if result:
                        return result

    # Last resort: check if any line IS a label
    for line in response.split('\n'):
        line_stripped = line.strip()
        for official_label in EVASION_LABELS:
            if line_stripped.lower() == official_label.lower():
                return official_label

    return ""


def extract_confidence(response: str) -> int:
    """Extract confidence score (1-5) from response."""
    for line in response.split('\n'):
        if 'CONFIDENCE' in line.upper() and ':' in line:
            parts = line.split(':', 1)
            if len(parts) > 1:
                try:
                    conf = int(parts[1].strip().split()[0])
                    return min(max(conf, 1), 5)
                except:
                    pass
    return 3 


# Parsing & Analysis Helpers

def extract_label_with_parse_meta(response: str, field_name: str = "FINAL_LABEL") -> Tuple[str, Dict[str, Any]]:
    """Extract label plus parsing trace metadata for logging."""
    parse_meta: Dict[str, Any] = {
        "target_field": field_name,
        "source": None,
        "label_field_detected": False,
        "raw_label": "",
        "normalized_label": "",
        "parse_success": False,
        "fallback_reason": None,
    }

    if not response:
        parse_meta["fallback_reason"] = "empty_response"
        return "", parse_meta

    # 1) Target field
    for line in response.split("\n"):
        line_stripped = line.strip()
        if field_name.lower() in line_stripped.lower():
            parse_meta["label_field_detected"] = True
            if ":" in line_stripped:
                raw_label = line_stripped.split(":", 1)[1].strip()
                parse_meta["raw_label"] = raw_label
                norm = normalize_label(raw_label, allow_fallback=False)
                if norm:
                    parse_meta["source"] = "target_field"
                    parse_meta["normalized_label"] = norm
                    parse_meta["parse_success"] = True
                    return norm, parse_meta

    # 2) Generic label-like fields
    label_keywords = ["LABEL", "FINAL_LABEL", "CHOICE", "ANSWER"]
    for line in response.split("\n"):
        line_stripped = line.strip()
        for keyword in label_keywords:
            if keyword in line_stripped.upper() and ":" in line_stripped:
                parse_meta["label_field_detected"] = True
                raw_label = line_stripped.split(":", 1)[1].strip()
                parse_meta["raw_label"] = raw_label
                norm = normalize_label(raw_label, allow_fallback=False)
                if norm:
                    parse_meta["source"] = "label_like_field"
                    parse_meta["normalized_label"] = norm
                    parse_meta["parse_success"] = True
                    return norm, parse_meta

    # 3) Exact label line
    for line in response.split("\n"):
        line_stripped = line.strip()
        for official_label in EVASION_LABELS:
            if line_stripped.lower() == official_label.lower():
                parse_meta["source"] = "exact_line_label"
                parse_meta["raw_label"] = line_stripped
                parse_meta["normalized_label"] = official_label
                parse_meta["parse_success"] = True
                return official_label, parse_meta

    parse_meta["fallback_reason"] = "label_not_found"
    return "", parse_meta


ROUND0_STEP_FIELDS = [
    "STEP1_QUESTION_TOPIC",
    "STEP2_ANSWER_TOPIC",
    "STEP3_TOPIC_MATCH",
    "STEP4_DIRECT_CHECK",
    "STEP5_INFERENCE_CHECK",
    "STEP6_REFUSAL_CHECK",
    "STEP7_BLAME_CHECK",
    "STEP8_MULTI_PART_CHECK",
]


def parse_cot_steps(response: str, fields: list) -> dict:
    """Parse structured step values from a CoT response."""
    steps = {}
    if not response:
        return steps
    for line in response.split('\n'):
        line_stripped = line.strip()
        for field in fields:
            if field in line_stripped and ':' in line_stripped:
                value = line_stripped.split(':', 1)[1].strip()
                steps[field] = value[:300]
                break
    return steps


def check_step_label_consistency(steps: dict, final_label: str) -> bool:
    """Check if parsed CoT steps are logically consistent with the final label."""
    if not steps:
        return True

    topic_match = steps.get("STEP3_TOPIC_MATCH", "").upper()
    direct_check = steps.get("STEP4_DIRECT_CHECK", "").upper()
    refusal_check = steps.get("STEP6_REFUSAL_CHECK", "").upper()

    if ("NO" in topic_match and not topic_match.startswith("NOT")) and final_label != "Dodging":
        if any(p in topic_match for p in ["NO -", "NO,", "NO.", "NO "]):
            if "SPECIFIC" not in topic_match and "DETAIL" not in topic_match:
                return False

    if topic_match.startswith("YES") and final_label == "Dodging":
        return False

    if any(marker in direct_check for marker in ["YES", "DIRECT ANSWER", "DIRECTLY STATES", "DIRECTLY ADDRESSES"]):
        if not direct_check.startswith("NO") and final_label not in ("Explicit", "Partial/half-answer"):
            return False

    if any(marker in refusal_check for marker in ["YES", "EXPLICIT REFUSAL", "REFUSES"]):
        if not refusal_check.startswith("NO") and final_label != "Declining to answer":
            return False

    return True


def compute_annotator_analysis(gold_evasion_labels: list) -> dict:
    """Compute annotator agreement statistics from gold labels."""
    if not gold_evasion_labels:
        return {
            "agreement": "unknown",
            "majority_label": None,
            "majority_count": 0,
            "unique_labels": 0,
            "clarity_labels": [],
            "clarity_agreement": "unknown",
            "evasion_crosses_clarity_boundary": False,
        }

    label_counts = Counter(gold_evasion_labels)
    majority_label = label_counts.most_common(1)[0][0]
    majority_count = label_counts.most_common(1)[0][1]
    unique_count = len(label_counts)

    n_annotators = len(gold_evasion_labels)
    if unique_count == 1:
        agreement = "unanimous"
    elif majority_count >= 2:
        agreement = "majority"
    elif n_annotators == 2:
        agreement = "split"
    else:
        agreement = "three_way_split"

    clarity_labels = [EVASION_TO_CLARITY.get(l, "Ambivalent") for l in gold_evasion_labels]
    clarity_counts = Counter(clarity_labels)

    if len(clarity_counts) == 1:
        clarity_agreement = "unanimous"
    elif clarity_counts.most_common(1)[0][1] >= 2:
        clarity_agreement = "majority"
    else:
        clarity_agreement = "split"

    return {
        "agreement": agreement,
        "majority_label": majority_label,
        "majority_count": majority_count,
        "unique_labels": unique_count,
        "clarity_labels": clarity_labels,
        "clarity_agreement": clarity_agreement,
        "evasion_crosses_clarity_boundary": len(clarity_counts) > 1,
    }


def compute_text_features(question: str, answer: str, full_question: str = "") -> dict:
    """Compute text length features for dataset analysis."""
    return {
        "answer_word_count": len(answer.split()) if answer else 0,
        "answer_char_count": len(answer) if answer else 0,
        "question_word_count": len(question.split()) if question else 0,
        "full_question_word_count": len(full_question.split()) if full_question else 0,
    }


def is_clarity_safe_evasion_error(pred_evasion: str, gold_evasion_labels: list) -> Optional[bool]:
    """Check if an evasion error is clarity-safe (doesn't cross clarity boundary)."""
    if pred_evasion in set(gold_evasion_labels):
        return None

    pred_clarity = EVASION_TO_CLARITY.get(pred_evasion, "Ambivalent")
    gold_clarities = set(EVASION_TO_CLARITY.get(g, "Ambivalent") for g in gold_evasion_labels)

    return pred_clarity in gold_clarities


def compute_vote_margin(weighted_votes: Dict[str, float]) -> float:
    """Top1 - Top2 margin from weighted vote dictionary."""
    if not weighted_votes:
        return 0.0
    sorted_values = sorted(weighted_votes.values(), reverse=True)
    top1 = sorted_values[0]
    top2 = sorted_values[1] if len(sorted_values) > 1 else 0.0
    return float(top1 - top2)


# ============================================================================
# Base Classifier (used by both Grok and Gemini)
# ============================================================================

class BaseAPEXClassifier:
    """Base classifier with self-consistency."""
    
    def __init__(self, client, k_samples: int = 5, 
                 model_name: str = "base"):
        self.client = client
        self.k_samples = k_samples
        self.model_name = model_name

    def classify(self, question: str, answer: str, 
                 full_question: str = None, is_grok: bool = True) -> Tuple[str, str, float, Dict]:
        """
        Classify a Q&A pair using evasion-first strategy.
        Returns:
            (evasion_label, clarity_label, confidence, metadata)
        """
        # Build user prompt
        user_prompt = f"""QUESTION: {question}

ANSWER: {answer}"""
        
        if full_question and full_question != question:
            user_prompt = f"""FULL INTERVIEW QUESTION: {full_question}

SPECIFIC SUB-QUESTION TO EVALUATE: {question}

ANSWER: {answer}"""

        # Stage 1: Self-Consistency - Sample k times IN PARALLEL
        def single_call(sample_idx):
            if is_grok:
                temp = 0.5 if sample_idx > 0 else 0.3
            else:
                temp = 1.0

            call_result = self.client.chat(
                COT_EVASION_SYSTEM_PROMPT,
                user_prompt,
                temperature=temp,
                return_meta=True,
            )

            if isinstance(call_result, dict):
                response = call_result.get("text", "")
                usage_meta = call_result.get("usage", {})
                api_meta = call_result.get("api", {})
            else:
                response = call_result or ""
                usage_meta = {}
                api_meta = {}

            label, parse_meta = extract_label_with_parse_meta(response, "FINAL_LABEL")
            is_fallback = False
            if not label:
                label = "General"
                is_fallback = True
                if not parse_meta.get("fallback_reason"):
                    parse_meta["fallback_reason"] = "label_parse_failed"

            conf = extract_confidence(response)
            parse_meta["confidence_parsed"] = conf
            parse_meta["is_fallback"] = is_fallback
            parse_meta["final_label"] = label
            return sample_idx, response, label, conf, is_fallback, usage_meta, api_meta, parse_meta

        predictions = [None] * self.k_samples
        confidences = [None] * self.k_samples
        responses = [None] * self.k_samples
        fallback_flags = [None] * self.k_samples
        usage_metas = [None] * self.k_samples
        api_metas = [None] * self.k_samples
        parse_metas = [None] * self.k_samples

        # Run k samples in parallel
        with ThreadPoolExecutor(max_workers=self.k_samples) as executor:
            futures = [executor.submit(single_call, i) for i in range(self.k_samples)]
            for future in as_completed(futures):
                idx, resp, label, conf, fallback, usage_meta, api_meta, parse_meta = future.result()
                predictions[idx] = label
                confidences[idx] = conf
                responses[idx] = resp
                fallback_flags[idx] = fallback
                usage_metas[idx] = usage_meta
                api_metas[idx] = api_meta
                parse_metas[idx] = parse_meta

        response_details = []
        for idx in range(self.k_samples):
            temp = (0.5 if idx > 0 else 0.3) if is_grok else 1.0
            parsed_steps = parse_cot_steps(responses[idx], ROUND0_STEP_FIELDS)
            consistent = check_step_label_consistency(parsed_steps, predictions[idx])

            response_details.append({
                "idx": idx,
                "label": predictions[idx],
                "confidence": confidences[idx],
                "temperature": temp,
                "is_fallback": fallback_flags[idx],
                "raw_response": responses[idx],
                "steps": parsed_steps,
                "step_label_consistent": consistent,
                "usage": usage_metas[idx] if usage_metas[idx] is not None else {},
                "api": api_metas[idx] if api_metas[idx] is not None else {},
                "parse": parse_metas[idx] if parse_metas[idx] is not None else {},
            })

        # Stage 2: Majority Vote
        label_counts = Counter(predictions)
        most_common = label_counts.most_common()
        primary_label = most_common[0][0]
        vote_count = most_common[0][1]
        consistency = vote_count / self.k_samples
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        normalized_confidence = (avg_confidence / 5.0) * consistency

        # Stage 3: Map to clarity
        clarity_label = EVASION_TO_CLARITY.get(primary_label, "Ambivalent")

        metadata = {
            "predictions": predictions,
            "confidences": confidences,
            "consistency": consistency,
            "vote_counts": dict(label_counts),
            "vote_inputs": {
                "k_samples": self.k_samples,
                "predictions": predictions,
                "confidences": confidences,
                "vote_counts": dict(label_counts),
                "majority_label": primary_label,
                "majority_votes": vote_count,
                "consistency": consistency,
            },
            "avg_confidence": avg_confidence,
            "raw_responses": responses,
            "response_details": response_details,
        }

        return primary_label, clarity_label, normalized_confidence, metadata


# ============================================================================
# APEX Ultimate Ensemble Classifier
# ============================================================================

class APEXUltimateClassifier:
    """
    Ensemble classifier with weighted voting.

    Strategy:
    1. Run both Grok and Gemini classifiers with k=5 self-consistency
    2. If both agree on clarity → use that (highest reliability)
    3. If they disagree → weighted voting: Grok 5 votes + Gemini 4 votes
    """
    
    GEMINI_WEIGHT = 4
    
    def __init__(self, grok_client: GrokClient, gemini_client: GeminiClient, 
                 k_samples: int = 5):
        self.grok_classifier = BaseAPEXClassifier(grok_client, k_samples, 
                                                   "grok")
        self.gemini_classifier = BaseAPEXClassifier(gemini_client, k_samples, 
                                                     "gemini")
        self.k_samples = k_samples
    
    def classify(self, question: str, answer: str, 
                 full_question: str = None) -> Tuple[str, str, float, Dict]:
        """
        Runs BOTH models IN PARALLEL for faster execution.
        
        Returns:
            (final_evasion_label, final_clarity_label, confidence, metadata)
        """
        print("    [PARALLEL] Running Grok + Gemini simultaneously...")
        
        # Run Grok and Gemini in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            grok_future = executor.submit(
                self.grok_classifier.classify,
                question, answer, full_question, True
            )
            gemini_future = executor.submit(
                self.gemini_classifier.classify,
                question, answer, full_question, False
            )
            
            grok_evasion, grok_clarity, grok_conf, grok_meta = grok_future.result()
            gemini_evasion, gemini_clarity, gemini_conf, gemini_meta = gemini_future.result()
        
        print(f"    [Grok] {grok_evasion} (cons={grok_meta['consistency']:.2f})")
        print(f"    [Gemini] {gemini_evasion} (cons={gemini_meta['consistency']:.2f})")
        
        weighted_votes = Counter()
        for evasion_label, count in grok_meta['vote_counts'].items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            weighted_votes[clarity] += count
        weighted_votes[gemini_clarity] += self.GEMINI_WEIGHT

        # Rule 1: If both agree on clarity → use that (highest reliability)
        if grok_clarity == gemini_clarity:
            final_clarity = grok_clarity
            final_evasion = grok_evasion  # Use Grok's more specific label
            decision_reason = "AGREE"
        else:
            # Rule 2: Weighted voting - Grok 5 votes + Gemini 4 votes
            # Get winner
            final_clarity = weighted_votes.most_common(1)[0][0]
            
            # Determine final evasion label based on clarity winner
            if final_clarity == grok_clarity:
                final_evasion = grok_evasion
                decision_reason = "WEIGHTED_GROK"
            else:
                final_evasion = gemini_evasion
                decision_reason = "WEIGHTED_GEMINI"
        
        # Confidence is average of both
        final_confidence = (grok_conf + gemini_conf) / 2
        
        metadata = {
            "grok": {
                "evasion_label": grok_evasion,
                "clarity_label": grok_clarity,
                "confidence": grok_conf,
                "consistency": grok_meta['consistency'],
                "vote_counts": grok_meta['vote_counts'],
                "vote_inputs": grok_meta.get("vote_inputs", {}),
                "avg_confidence": grok_meta.get("avg_confidence", 0.0),
                "response_details": grok_meta.get("response_details", []),
            },
            "gemini": {
                "evasion_label": gemini_evasion,
                "clarity_label": gemini_clarity,
                "confidence": gemini_conf,
                "consistency": gemini_meta['consistency'],
                "vote_counts": gemini_meta['vote_counts'],
                "vote_inputs": gemini_meta.get("vote_inputs", {}),
                "avg_confidence": gemini_meta.get("avg_confidence", 0.0),
                "response_details": gemini_meta.get("response_details", []),
            },
            "ensemble": {
                "decision_reason": decision_reason,
                "gemini_weight": self.GEMINI_WEIGHT,
                "final_clarity": final_clarity,
                "final_evasion": final_evasion,
                "final_clarity_votes_weighted": dict(weighted_votes),
                "final_vote_margin": compute_vote_margin(dict(weighted_votes)),
                "round0_static_clarity_votes_weighted": dict(weighted_votes),
                "api_calls_estimated": 2 * self.k_samples,
            }
        }
        
        return final_evasion, final_clarity, final_confidence, metadata


# Metrics Calculation

def f1_for_class(gold_annotations: List[List[str]], predictions: List[str], 
                 target_class: str) -> dict:
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


# Main Processing

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


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble Classifier with Weighted Voting"
    )
    parser.add_argument(
        "--grok_api_key",
        type=str,
        default=os.environ.get("XAI_API_KEY"),
        help="xAI API key for Grok (or set XAI_API_KEY env var)"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY"),
        help="Google Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Path to evaluation CSV file (if provided, uses CSV )"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="stage1",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--k_samples",
        type=int,
        default=5,
        help="Number of samples for self-consistency (default: 5)"
    )
    parser.add_argument(
        "--thinking_level",
        type=str,
        default="high",
        choices=["minimal", "low", "medium", "high"],
        help="Gemini 3 thinking level: minimal/low/medium/high (default: high)"
    )
    parser.add_argument(
        "--detailed_json",
        action="store_true",
        help="Save full nested detailed schema (text/gold/round0/ensemble/evaluation)"
    )

    args = parser.parse_args()

    if not args.grok_api_key:
        raise ValueError("Grok API key required. Set --grok_api_key or XAI_API_KEY env var")
    if not args.gemini_api_key:
        raise ValueError("Gemini API key required. Set --gemini_api_key or GEMINI_API_KEY env var")

    
    print("   1. Run Grok (k=5) and Gemini (k=5) independently")
    print("   2. If both agree → use that prediction")
    print(f"   3. If disagree → weighted voting: Grok 5 votes + Gemini {APEXUltimateClassifier.GEMINI_WEIGHT} votes")

    print("\n Initializing clients...")
    grok_client = GrokClient(args.grok_api_key)
    gemini_client = GeminiClient(args.gemini_api_key, thinking_level=args.thinking_level)
    classifier = APEXUltimateClassifier(grok_client, gemini_client, k_samples=args.k_samples)

    # Load data - either from CSV or HuggingFace dataset
    is_eval_mode = args.eval_csv is not None
    
    if is_eval_mode:
        print(f"\n Loading evaluation set from CSV: {args.eval_csv}...")
        df = pd.read_csv(args.eval_csv)
        # Convert DataFrame to list of dicts for consistent interface
        ds = df.to_dict('records')
        if args.max_samples:
            ds = ds[:args.max_samples]
        print(f"   Loaded {len(ds)} examples from CSV")
        print(f"   EVALUATION MODE: Gold labels are empty, metrics will be skipped")
    else:
        print(f"\n Loading QEvasion dataset (split: {args.split})...")
        ds = load_dataset("ailsntua/QEvasion", split=args.split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"   Loaded {len(ds)} examples")

    print(f"\n Starting ensemble classification...\n")

    all_results = []
    preds_evasion = []
    preds_clarity = []
    gold_multi = []

    for i, example in enumerate(ds):
        print(f"{'='*70}")
        print(f"SAMPLE {i+1}/{len(ds)}")
        print(f"{'='*70}")

        question = example.get("question", "") if isinstance(example, dict) else example["question"]
        answer = example.get("interview_answer", "") if isinstance(example, dict) else example["interview_answer"]
        full_question = example.get("interview_question", "") if isinstance(example, dict) else example.get("interview_question", "")
        
        # Handle NaN values from CSV
        if pd.isna(question):
            question = ""
        if pd.isna(answer):
            answer = ""
        if pd.isna(full_question):
            full_question = ""

        gold_labels = extract_gold_labels(example) if not is_eval_mode else []
        gold_multi.append(gold_labels)

        print(f"   Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")

        # Classify
        evasion_label, clarity_label, confidence, metadata = classifier.classify(
            question, answer, full_question
        )

        preds_evasion.append(evasion_label)
        preds_clarity.append(clarity_label)

        gold_clarity = example.get("clarity_label", "") if isinstance(example, dict) else ""
        if pd.isna(gold_clarity):
            gold_clarity = ""
        
        if is_eval_mode:
            is_evasion_correct = None
            is_clarity_correct = None
        else:
            is_evasion_correct = evasion_label in set(gold_labels) if gold_labels else False
            is_clarity_correct = clarity_label == gold_clarity

        # Store result
        if args.detailed_json:
            text_feats = compute_text_features(question, answer, full_question)
            text_feats["question"] = question
            text_feats["answer"] = answer
            text_feats["full_question"] = full_question

            annotator = compute_annotator_analysis(gold_labels) if (not is_eval_mode and gold_labels) else {}

            def build_round0_block(meta):
                return {
                    "majority_label": meta.get("evasion_label", ""),
                    "clarity": meta.get("clarity_label", ""),
                    "consistency": meta.get("consistency", 0.0),
                    "vote_counts": meta.get("vote_counts", {}),
                    "vote_inputs": meta.get("vote_inputs", {}),
                    "avg_confidence": meta.get("avg_confidence", 0.0),
                    "responses": meta.get("response_details", []),
                }

            if is_eval_mode:
                eval_block = {}
            else:
                clarity_safe = is_clarity_safe_evasion_error(evasion_label, gold_labels) if gold_labels else None
                eval_block = {
                    "evasion_correct": is_evasion_correct,
                    "clarity_correct": is_clarity_correct,
                    "grok_evasion_correct": (metadata["grok"].get("evasion_label", "") in set(gold_labels)) if gold_labels else None,
                    "gemini_evasion_correct": (metadata["gemini"].get("evasion_label", "") in set(gold_labels)) if gold_labels else None,
                    "grok_clarity_correct": (metadata["grok"].get("clarity_label", "") == gold_clarity) if gold_clarity else None,
                    "gemini_clarity_correct": (metadata["gemini"].get("clarity_label", "") == gold_clarity) if gold_clarity else None,
                    "clarity_safe_evasion_error": clarity_safe,
                }

            result = {
                "index": i,
                "text_features": text_feats,
                "gold": {
                    "evasion_labels": gold_labels,
                    "clarity_label": gold_clarity,
                    **annotator,
                } if not is_eval_mode else {},
                "grok_round0": build_round0_block(metadata["grok"]),
                "gemini_round0": build_round0_block(metadata["gemini"]),
                "ensemble": {
                    "final_evasion": evasion_label,
                    "final_clarity": clarity_label,
                    "confidence": confidence,
                    "decision_reason": metadata["ensemble"].get("decision_reason", "UNKNOWN"),
                    "final_clarity_votes_weighted": metadata["ensemble"].get("final_clarity_votes_weighted", {}),
                    "final_vote_margin": metadata["ensemble"].get("final_vote_margin", 0.0),
                    "round0_static_clarity_votes_weighted": metadata["ensemble"].get("round0_static_clarity_votes_weighted", {}),
                    "grok_weight": 5,
                    "gemini_weight": APEXUltimateClassifier.GEMINI_WEIGHT,
                    "api_calls_estimated": metadata["ensemble"].get("api_calls_estimated"),
                },
                "evaluation": eval_block,
            }
        else:
            result = {
                "index": i,
                "evasion_label": evasion_label,
                "clarity_label": clarity_label,
                "confidence": confidence,
                "grok_evasion": metadata["grok"]["evasion_label"],
                "grok_clarity": metadata["grok"]["clarity_label"],
                "grok_consistency": metadata["grok"]["consistency"],
                "grok_votes": metadata["grok"]["vote_counts"],
                "gemini_evasion": metadata["gemini"]["evasion_label"],
                "gemini_clarity": metadata["gemini"]["clarity_label"],
                "gemini_consistency": metadata["gemini"]["consistency"],
                "gemini_votes": metadata["gemini"]["vote_counts"],
                "decision_reason": metadata["ensemble"]["decision_reason"],
                "gold_evasion": gold_labels,
                "gold_clarity": gold_clarity
            }

        all_results.append(result)

        # Format vote counts for display
        def format_votes(votes_dict):
            if not votes_dict:
                return "N/A"
            sorted_votes = sorted(votes_dict.items(), key=lambda x: x[1], reverse=True)
            return ", ".join(f"{label}({count})" for label, count in sorted_votes)
        
        grok_votes_str = format_votes(metadata['grok']['vote_counts'])
        gemini_votes_str = format_votes(metadata['gemini']['vote_counts'])
        
        print(f"\n   RESULT:")
        print(f"      Grok votes:   {grok_votes_str}")
        print(f"      Grok final:   {metadata['grok']['evasion_label']} → {metadata['grok']['clarity_label']} (cons={metadata['grok']['consistency']:.2f})")
        print(f"      Gemini votes: {gemini_votes_str}")
        print(f"      Gemini final: {metadata['gemini']['evasion_label']} → {metadata['gemini']['clarity_label']} (cons={metadata['gemini']['consistency']:.2f})")
        print(f"      Decision: {metadata['ensemble']['decision_reason']}")
        print(f"")
        
        if is_eval_mode:
            print(f"      EVASION:  Pred={evasion_label:<20} (Gold: N/A - Eval mode)")
            print(f"      CLARITY:  Pred={clarity_label:<20} (Gold: N/A - Eval mode)")
        else:
            print(f"      EVASION:  Pred={evasion_label:<20} Gold={gold_labels}  {'✅' if is_evasion_correct else '❌'}")
            print(f"      CLARITY:  Pred={clarity_label:<20} Gold={gold_clarity:<15}  {'✅' if is_clarity_correct else '❌'}")
        
        # Compute running F1 scores only if not in eval mode
        if not is_eval_mode:
            golds_clarity_so_far = [ds[j].get("clarity_label", "").strip() if isinstance(ds[j], dict) else ds[j]["clarity_label"] for j in range(i + 1)]
            clarity_f1_so_far = f1_score(golds_clarity_so_far, preds_clarity, average="macro", zero_division=0)
            evasion_acc_so_far = compute_instance_accuracy(gold_multi, preds_evasion)
            evasion_f1_so_far, _ = compute_macro_f1(gold_multi, preds_evasion)
            
            print(f"")
            print(f"   RUNNING METRICS ({i+1} samples):")
            print(f"      Clarity Macro F1:  {clarity_f1_so_far:.4f}")
            print(f"      Evasion Macro F1:  {evasion_f1_so_far:.4f}  (Accuracy: {evasion_acc_so_far:.4f})")
        print()

    # Final Metrics (skip if evaluation mode)
    if is_eval_mode:
        print("\n" + "="*70)
        print("EVALUATION MODE - METRICS SKIPPED (no gold labels)")
        print("="*70)
        
        print("\n   Prediction distribution (Evasion):")
        pred_counts = Counter(preds_evasion)
        for label, count in pred_counts.most_common():
            print(f"     {label}: {count}")
        
        print("\n   Prediction distribution (Clarity):")
        clarity_counts = Counter(preds_clarity)
        for label, count in clarity_counts.most_common():
            print(f"     {label}: {count}")
        
        # Set placeholder values for metrics saving
        evasion_acc = None
        evasion_macro_f1 = None
        evasion_classes = None
        clarity_acc = None
        clarity_macro_f1 = None
    else:
        print("\n" + "="*70)
        print("FINAL METRICS")
        print("="*70)

        print("\nEVASION METRICS (Task 2)")
        print("-"*40)

        evasion_acc = compute_instance_accuracy(gold_multi, preds_evasion)
        evasion_macro_f1, evasion_classes = compute_macro_f1(gold_multi, preds_evasion)

        print(f"   Classes found: {evasion_classes}")
        print(f"   Instance Accuracy: {evasion_acc:.4f}")
        print(f"   Macro F1:          {evasion_macro_f1:.4f}")

        print("\n   Prediction distribution:")
        pred_counts = Counter(preds_evasion)
        for label, count in pred_counts.most_common():
            print(f"     {label}: {count}")

        print("\nCLARITY METRICS (Task 1)")
        print("-"*40)

        golds_clarity = []
        for ex in ds:
            gold_c = ex.get("clarity_label", "").strip() if isinstance(ex, dict) and "clarity_label" in ex else "Ambivalent"
            if pd.isna(gold_c):
                gold_c = "Ambivalent"
            golds_clarity.append(gold_c)

        clarity_acc = accuracy_score(golds_clarity, preds_clarity)
        clarity_macro_f1 = f1_score(golds_clarity, preds_clarity, average="macro")

        print(f"   Accuracy:  {clarity_acc:.4f}")
        print(f"   Macro F1:  {clarity_macro_f1:.4f}")

        print("\n   Prediction distribution:")
        clarity_counts = Counter(preds_clarity)
        for label, count in clarity_counts.most_common():
            print(f"     {label}: {count}")

        print("\n   Classification Report:")
        print(classification_report(golds_clarity, preds_clarity))

    # Save Results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    # Evasion pickle
    evasion_file = f"{args.output_prefix}_evasion.pkl"
    with open(evasion_file, "wb") as f:
        pickle.dump([list(range(len(preds_evasion))), preds_evasion], f)
    print(f" Saved evasion labels to: {evasion_file}")

    # Clarity pickle
    clarity_file = f"{args.output_prefix}_clarity.pkl"
    with open(clarity_file, "wb") as f:
        pickle.dump([list(range(len(preds_clarity))), preds_clarity], f)
    print(f" Saved clarity labels to: {clarity_file}")

    # Results JSON
    results_json_file = (
        f"{args.output_prefix}_detailed.json"
        if args.detailed_json
        else f"{args.output_prefix}_full.json"
    )
    with open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f" Saved results to: {results_json_file}")

    # Metrics JSON (only save if not eval mode or include placeholders)
    metrics_file = f"{args.output_prefix}_metrics.json"
    metrics = {
        "evasion": {
            "accuracy": evasion_acc,
            "macro_f1": evasion_macro_f1,
            "classes": evasion_classes
        },
        "clarity": {
            "accuracy": clarity_acc,
            "macro_f1": clarity_macro_f1
        },
        "config": {
            "k_samples": args.k_samples,
            "grok_model": "grok-4-1-fast-reasoning",
            "gemini_model": "gemini-3-flash-preview",
            "gemini_thinking_level": args.thinking_level,
            "gemini_weight": APEXUltimateClassifier.GEMINI_WEIGHT,
            "strategy": "weighted_voting_agree_first",
            "is_eval_mode": is_eval_mode,
            "detailed_json": args.detailed_json,
        },
        "samples_processed": len(ds)
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f" Saved metrics to: {metrics_file}")

    print("\n" + "="*70)
    print("PROCESSING COMPLETE!")
    if is_eval_mode:
        print(f"   Processed {len(ds)} samples in EVALUATION mode")
        print(f"   Predictions saved to {clarity_file} (upload to Codabench)")
    else:
        print(f"   Clarity Macro F1: {clarity_macro_f1:.4f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
