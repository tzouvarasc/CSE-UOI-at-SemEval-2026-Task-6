# -*- coding: utf-8 -*-
"""
Modes:
1. `ultimate`: static weighted voting baseline
2. `debate`: iterative multi-agent debate with adaptive stopping
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

# Global Trackers

FALLBACK_COUNTER = {"round0_grok": 0, "round0_gemini": 0, "debate_grok": 0, "debate_gemini": 0}

# Clarity label mapping: eval set uses different names than train set
EVAL_CLARITY_MAP = {
    "Direct Reply": "Clear Reply",
    "Direct Non-Reply": "Clear Non-Reply",
    "Ambivalent": "Ambivalent",
    # pass-through for already-correct labels
    "Clear Reply": "Clear Reply",
    "Clear Non-Reply": "Clear Non-Reply",
}

# Evasion label mapping: eval CSV uses slightly different names
EVAL_EVASION_MAP = {
    "Claims Ignorance": "Claims ignorance",
    "Partial": "Partial/half-answer",
    "Diffusion": "Deflection",
}

# Configuration & Constants

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
             max_tokens: int = 7000, temperature: float = 0.4,
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
             max_tokens: int = 12000, temperature: float = 1.0,
             return_meta: bool = False):

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Combine system and user prompts for Gemini
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

                # Fallback for SDK versions exposing camelCase fields.
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


# Chain-of-Thought Evasion Classification Prompt

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


# Debate Prompts (Debate)

DEBATE_SYSTEM_PROMPT = """You are an expert political evasion classifier participating in a structured deliberation.

## YOUR PREVIOUS CLASSIFICATION
You classified this Q&A pair as: **{own_label}** ({own_clarity})
Your consistency score: {own_consistency} (out of {k} samples, {own_votes} agreed)

Your reasoning summary:
{own_cot_summary}

## OPPOSING CLASSIFICATION
Another independent expert classifier analyzed the SAME Q&A pair and reached a DIFFERENT conclusion:
Classification: **{other_label}** ({other_clarity})
Their consistency score: {other_consistency} (out of {k} samples, {other_votes} agreed)

Their reasoning summary:
{other_cot_summary}

## THE SPECIFIC DISAGREEMENT
This is a {own_clarity} vs {other_clarity} disagreement.
{boundary_specific_guidance}

## DELIBERATION RULES
1. CAREFULLY read the other expert's reasoning. They may have noticed something you missed.
2. DO NOT change your answer just because someone disagrees. Only change if you find a genuine flaw in your OWN reasoning.
3. If you change your answer, you MUST explain EXACTLY what was wrong with your previous reasoning.
4. If you maintain your answer, you MUST explain why the other expert's reasoning is flawed.
5. Consider: Would a HUMAN ANNOTATOR (non-expert, following annotation guidelines) classify this the same way you did?

## OUTPUT FORMAT
STEP1_RECONSIDERED_TOPIC_MATCH: <re-examine if answer is on-topic>
STEP2_OPPOSING_ARGUMENT_STRENGTH: <STRONG/MODERATE/WEAK - how convincing is the other expert?>
STEP3_OWN_REASONING_FLAW: <Did you find a flaw in your OWN reasoning? YES/NO - if YES, explain>
STEP4_DECISION: <MAINTAIN or CHANGE>
STEP5_JUSTIFICATION: <Why you maintain or what changed your mind>
FINAL_LABEL: <one of 9 evasion labels>
CONFIDENCE: <1-5>"""


BOUNDARY_GUIDANCE = {
    ("Ambivalent", "Clear Reply"): """
THE EXPLICIT vs IMPLICIT/GENERAL BOUNDARY (where 84% of all errors occur):

ASK YOURSELF THESE SPECIFIC QUESTIONS:
1. Does the answer contain the LITERAL information requested? (specific yes/no, a number, a name, a date, a concrete commitment)
   - YES -> likely Explicit
   - NO -> NOT Explicit, even if on-topic

2. Can a reasonable person INFER the answer from what's said?
   - YES -> Implicit (Ambivalent)
   - NO -> General (Ambivalent)

COMMON FALSE POSITIVE PATTERN (models predicting Explicit when humans don't):
The politician says something substantive and on-topic, and the model interprets deep semantic content as "explicit." But human annotators look for LITERAL, SURFACE-LEVEL directness. A thorough, on-topic response is NOT necessarily Explicit - it may be Implicit (inferable but not stated) or General (on-topic but vague).

COMMON FALSE NEGATIVE PATTERN (models missing Explicit when humans see it):
The politician gives a direct answer but embedded in a longer response with qualifications. The model focuses on the hedging/qualifications and classifies as Implicit or General. But the core direct answer IS there - human annotators would catch it.

CRITICAL TEST: Read ONLY the first 1-2 sentences of the answer. Does the answer to the question appear there explicitly? If you need to read the whole answer and reason about it - it's probably NOT Explicit.""",

    ("Ambivalent", "Clear Non-Reply"): """
THE AMBIVALENT vs CLEAR NON-REPLY BOUNDARY:

Key distinction: Does the politician ENGAGE with the topic at all?
- If they talk about the topic, even vaguely -> Ambivalent (General/Implicit/Dodging)
- If they REFUSE, claim ignorance, or redirect -> Clear Non-Reply

This boundary has fewer errors (9 total). Focus on whether there's an EXPLICIT refusal marker:
- "I won't comment" / "no comment" -> Clear Non-Reply (Declining)
- "I don't know" / "I'm not sure" -> Clear Non-Reply (Claims ignorance)
- "What do you mean by...?" -> Clear Non-Reply (Clarification)
- Everything else where they at least attempt to talk -> Ambivalent""",

    ("Clear Non-Reply", "Clear Reply"): """
This disagreement is unusual - one says the politician answered directly, the other says they refused entirely.
Re-read carefully: is there ANY direct answer in the text? If yes -> Clear Reply.
Is there ONLY a refusal? If yes -> Clear Non-Reply.
"""
}


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
    return 3  # Default medium confidence


# Debate Helpers

def extract_cot_summary(raw_response: str) -> str:
    """
    Extract key reasoning steps from a full CoT response.
    Returns a concise summary suitable for debate context.
    """
    if not raw_response:
        return ""

    summary_parts = []
    key_fields = [
        "STEP1_QUESTION_TOPIC",
        "STEP3_TOPIC_MATCH",
        "STEP4_DIRECT_CHECK",
        "STEP5_INFERENCE_CHECK",
        "FINAL_LABEL",
        "CONFIDENCE",
    ]

    for line in raw_response.split("\n"):
        for field in key_fields:
            if field in line:
                summary_parts.append(line.strip())
                break

    if summary_parts:
        return "\n".join(summary_parts)

    return raw_response[:500] + "..." if len(raw_response) > 500 else raw_response


def compute_distribution_shift(prev: Dict[str, float], curr: Dict[str, float]) -> float:
    """Return KS-style max absolute difference between two vote distributions."""
    all_labels = set(prev.keys()) | set(curr.keys())
    prev_total = sum(prev.values())
    curr_total = sum(curr.values())
    if prev_total == 0 or curr_total == 0:
        return 1.0

    max_diff = 0.0
    for label in all_labels:
        p = prev.get(label, 0) / prev_total
        c = curr.get(label, 0) / curr_total
        max_diff = max(max_diff, abs(p - c))
    return max_diff


def check_debate_stability(vote_history: List[Dict[str, float]], threshold: float = 0.05) -> bool:
    """
    Simplified stability detection for 2-agent debate.

    vote_history: list of dicts, each dict = {clarity_label: total_weighted_votes}
    Returns True if the latest transition appears stable.
    """
    if len(vote_history) < 2:
        return False
    return compute_distribution_shift(vote_history[-2], vote_history[-1]) < threshold


def aggregate_debate_votes(all_rounds: List[Dict[str, Any]],
                           grok_weight: float,
                           gemini_weight: float) -> str:
    """
    Aggregate votes across all debate rounds with recency weighting.

    all_rounds: list of dicts with structure:
        {
            'round': int,
            'grok_votes': Counter,  # {evasion_label: count}
            'gemini_votes': Counter,
            ...
        }
    """
    round_weights = {0: 1.0, 1: 1.5, 2: 2.0}
    clarity_votes = Counter()

    for round_data in all_rounds:
        r = round_data.get("round", 0)
        recency_w = round_weights.get(r, 1.0)

        for evasion_label, count in round_data.get("grok_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            clarity_votes[clarity] += count * grok_weight * recency_w

        for evasion_label, count in round_data.get("gemini_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            clarity_votes[clarity] += count * gemini_weight * recency_w

    if not clarity_votes:
        return "Ambivalent"
    return clarity_votes.most_common(1)[0][0]


def aggregate_debate_clarity_votes(all_rounds: List[Dict[str, Any]],
                                   grok_weight: float,
                                   gemini_weight: float) -> Dict[str, float]:
    """Return recency-weighted clarity vote totals across debate rounds."""
    round_weights = {0: 1.0, 1: 1.5, 2: 2.0}
    clarity_votes = Counter()

    for round_data in all_rounds:
        r = round_data.get("round", 0)
        recency_w = round_weights.get(r, 1.0)

        for evasion_label, count in round_data.get("grok_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            clarity_votes[clarity] += count * grok_weight * recency_w

        for evasion_label, count in round_data.get("gemini_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            clarity_votes[clarity] += count * gemini_weight * recency_w

    return dict(clarity_votes)


def compute_vote_margin(weighted_votes: Dict[str, float]) -> float:
    """Top1 - Top2 margin from weighted vote dictionary."""
    if not weighted_votes:
        return 0.0
    sorted_values = sorted(weighted_votes.values(), reverse=True)
    top1 = sorted_values[0]
    top2 = sorted_values[1] if len(sorted_values) > 1 else 0.0
    return float(top1 - top2)


def extract_debate_decision(response: str) -> str:
    """Extract MAINTAIN/CHANGE from debate response.
    
    Handles multiple formats:
    - STEP4_DECISION: MAINTAIN
    - **Decision**: Maintain  
    - I will maintain my classification
    - After reconsidering... I change my label
    """
    if not response:
        return "UNKNOWN"
    
    resp_upper = response.upper()
    
    # 1. Try exact field match first
    for line in response.split("\n"):
        line_stripped = line.strip()
        if any(kw in line_stripped.upper() for kw in ["STEP4", "DECISION"]):
            line_up = line_stripped.upper()
            if "MAINTAIN" in line_up:
                return "MAINTAIN"
            if "CHANGE" in line_up and "UNCHANGED" not in line_up:
                return "CHANGE"
    
    # 2. Search in last 40% of response (where decision usually appears)
    decision_zone = response[int(len(response) * 0.6):]
    zone_upper = decision_zone.upper()
    
    has_maintain = "MAINTAIN" in zone_upper
    has_change = ("CHANGE" in zone_upper and "UNCHANGED" not in zone_upper 
                  and "NO CHANGE" not in zone_upper)
    
    if has_maintain and not has_change:
        return "MAINTAIN"
    if has_change and not has_maintain:
        return "CHANGE"
    
    # 3. Check for indirect signals
    maintain_signals = ["I STAND BY", "I CONFIRM", "MY ORIGINAL", "REMAIN WITH",
                        "KEEP MY", "STILL BELIEVE", "UPHOLD"]
    change_signals = ["I REVISE", "I UPDATE", "CONVINCED BY", "PERSUADED",
                      "SWITCH TO", "NOW CLASSIFY", "CORRECT MY"]
    
    for sig in maintain_signals:
        if sig in zone_upper:
            return "MAINTAIN"
    for sig in change_signals:
        if sig in zone_upper:
            return "CHANGE"
    
    return "UNKNOWN"


# Step Parsing & Analysis Helpers

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

DEBATE_STEP_FIELDS = [
    "STEP1_RECONSIDERED_TOPIC_MATCH",
    "STEP2_OPPOSING_ARGUMENT_STRENGTH",
    "STEP3_OWN_REASONING_FLAW",
    "STEP4_DECISION",
    "STEP5_JUSTIFICATION",
]


def parse_cot_steps(response: str, fields: list) -> dict:
    """Parse structured step values from a CoT response.
    
    Extracts the value after the colon for each field found.
    Returns dict mapping field_name -> value_string (max 300 chars).
    """
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
    """Check if parsed CoT steps are logically consistent with the final label.
    
    Rules:
    - STEP3 says off-topic (NO) → label should be Dodging
    - STEP3 says on-topic (YES) → label should NOT be Dodging
    - STEP4 says direct/yes → label should be Explicit
    - STEP6 says refusal → label should be Declining to answer
    
    Returns True if consistent or if steps are empty, False if contradiction detected.
    """
    if not steps:
        return True
    
    topic_match = steps.get("STEP3_TOPIC_MATCH", "").upper()
    direct_check = steps.get("STEP4_DIRECT_CHECK", "").upper()
    refusal_check = steps.get("STEP6_REFUSAL_CHECK", "").upper()
    
    # Rule 1: Off-topic should be Dodging
    if ("NO" in topic_match and not topic_match.startswith("NOT")) and final_label != "Dodging":
        # Exclude "NO specifics" etc — only match "NO" as topic mismatch indicator
        # Check it's not just "NO specifics" or similar by looking for common patterns
        if any(p in topic_match for p in ["NO -", "NO,", "NO.", "NO "]):
            if "SPECIFIC" not in topic_match and "DETAIL" not in topic_match:
                return False
    
    # Rule 2: On-topic should NOT be Dodging
    if topic_match.startswith("YES") and final_label == "Dodging":
        return False
    
    # Rule 3: Direct answer detected but not Explicit
    if any(marker in direct_check for marker in ["YES", "DIRECT ANSWER", "DIRECTLY STATES", "DIRECTLY ADDRESSES"]):
        if not direct_check.startswith("NO") and final_label not in ("Explicit", "Partial/half-answer"):
            return False
    
    # Rule 4: Refusal detected but not Declining
    if any(marker in refusal_check for marker in ["YES", "EXPLICIT REFUSAL", "REFUSES"]):
        if not refusal_check.startswith("NO") and final_label != "Declining to answer":
            return False
    
    return True


def compute_annotator_analysis(gold_evasion_labels: list) -> dict:
    """Compute annotator agreement statistics from gold labels.
    
    Args:
        gold_evasion_labels: list of 1-3 evasion label strings from annotators
    
    Returns dict with agreement analysis fields.
    """
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
    """Check if an evasion error is 'clarity-safe' (doesn't cross clarity boundary).
    
    Returns:
        None if prediction is correct (not an error)
        True if error but pred and gold map to SAME clarity class
        False if error AND crosses clarity boundary
    """
    if pred_evasion in set(gold_evasion_labels):
        return None
    
    pred_clarity = EVASION_TO_CLARITY.get(pred_evasion, "Ambivalent")
    gold_clarities = set(EVASION_TO_CLARITY.get(g, "Ambivalent") for g in gold_evasion_labels)
    
    return pred_clarity in gold_clarities


# Base Classifier (used by both Grok and Gemini)

class BaseAPEXClassifier:
    """Base  classifier with self-consistency."""
    
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
                key = "round0_grok" if is_grok else "round0_gemini"
                FALLBACK_COUNTER[key] += 1
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


#  Ultimate Ensemble Classifier

class APEXUltimateClassifier:

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


#  Debate Classifier

class APEXDebateClassifier:

    MAX_DEBATE_ROUNDS = 2
    DEBATE_K_SAMPLES = 3
    STABILITY_THRESHOLD = 0.05
    STABILITY_REQUIRED_CONSECUTIVE = 2

    ROUND_WEIGHTS = {0: 1.0, 1: 1.5, 2: 2.0}
    GROK_BASE_WEIGHT = 5
    GEMINI_BASE_WEIGHT = 4

    def __init__(self,
                 grok_client: GrokClient,
                 gemini_client: GeminiClient,
                 k_samples: int = 5,
                 max_debate_rounds: int = 2,
                 debate_k_samples: int = 3):
        self.grok_client = grok_client
        self.gemini_client = gemini_client
        self.grok_classifier = BaseAPEXClassifier(
            grok_client, k_samples, "grok"
        )
        self.gemini_classifier = BaseAPEXClassifier(
            gemini_client, k_samples, "gemini"
        )
        self.k_samples = k_samples
        self.max_debate_rounds = max(1, max_debate_rounds)
        self.debate_k_samples = max(1, debate_k_samples)

    def _build_user_prompt(self, question: str, answer: str, full_question: str = None) -> str:
        user_prompt = f"QUESTION: {question}\nANSWER: {answer}"
        if full_question and full_question != question:
            user_prompt = (
                f"FULL INTERVIEW QUESTION: {full_question}\n"
                f"SPECIFIC SUB-QUESTION: {question}\n"
                f"ANSWER: {answer}"
            )
        return user_prompt

    def _select_majority_response(self, metadata: Dict[str, Any], majority_label: str) -> str:
        predictions = metadata.get("predictions", [])
        raw_responses = metadata.get("raw_responses", [])
        for pred, raw in zip(predictions, raw_responses):
            if pred == majority_label and raw:
                return raw
        return raw_responses[0] if raw_responses else ""

    def _compute_vote_distribution(self, round_data: Dict[str, Any]) -> Dict[str, float]:
        votes = Counter()
        for evasion_label, count in round_data.get("grok_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            votes[clarity] += count * self.GROK_BASE_WEIGHT
        for evasion_label, count in round_data.get("gemini_votes", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            votes[clarity] += count * self.GEMINI_BASE_WEIGHT
        return dict(votes)

    def _round0_static_weighted(self,
                                grok_clarity: str,
                                grok_evasion: str,
                                grok_meta: Dict[str, Any],
                                gemini_clarity: str,
                                gemini_evasion: str) -> Tuple[str, str, str]:
        if grok_clarity == gemini_clarity:
            return grok_clarity, grok_evasion, "AGREE"

        votes = Counter(self._round0_static_clarity_votes(grok_meta, gemini_clarity))

        final_clarity = votes.most_common(1)[0][0] if votes else "Ambivalent"
        if final_clarity == grok_clarity:
            return final_clarity, grok_evasion, "WEIGHTED_GROK"
        return final_clarity, gemini_evasion, "WEIGHTED_GEMINI"

    def _round0_static_clarity_votes(self,
                                     grok_meta: Dict[str, Any],
                                     gemini_clarity: str) -> Dict[str, float]:
        votes = Counter()
        for evasion_label, count in grok_meta.get("vote_counts", {}).items():
            clarity = EVASION_TO_CLARITY.get(evasion_label, "Ambivalent")
            votes[clarity] += count
        votes[gemini_clarity] += self.GEMINI_BASE_WEIGHT
        return dict(votes)

    def _run_debate_samples(self, client, system_prompt: str, user_prompt: str,
                            k: int = 3, is_grok: bool = True) -> Dict[str, Any]:
        """Run k samples for a debate round."""

        predictions = [None] * k
        raw_responses = [None] * k
        decisions = [None] * k
        confidences = [None] * k
        fallback_flags = [None] * k
        usage_metas = [None] * k
        api_metas = [None] * k
        parse_metas = [None] * k

        def single_call(idx: int):
            temp = 0.4 if is_grok else 1.0
            call_result = client.chat(
                system_prompt, user_prompt, max_tokens=12000, temperature=temp, return_meta=True
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
                key = "debate_grok" if is_grok else "debate_gemini"
                FALLBACK_COUNTER[key] += 1
                if not parse_meta.get("fallback_reason"):
                    parse_meta["fallback_reason"] = "label_parse_failed"
            decision = extract_debate_decision(response)
            conf = extract_confidence(response)
            parse_meta["confidence_parsed"] = conf
            parse_meta["decision_parsed"] = decision
            parse_meta["is_fallback"] = is_fallback
            parse_meta["final_label"] = label
            return idx, response, label, decision, conf, is_fallback, usage_meta, api_meta, parse_meta

        with ThreadPoolExecutor(max_workers=k) as executor:
            futures = [executor.submit(single_call, i) for i in range(k)]
            for future in as_completed(futures):
                idx, response, label, decision, conf, is_fallback, usage_meta, api_meta, parse_meta = future.result()
                predictions[idx] = label
                raw_responses[idx] = response
                decisions[idx] = decision
                confidences[idx] = conf
                fallback_flags[idx] = is_fallback
                usage_metas[idx] = usage_meta
                api_metas[idx] = api_meta
                parse_metas[idx] = parse_meta

        response_details = []
        for idx in range(k):
            temp = 0.4 if is_grok else 1.0
            parsed_steps = parse_cot_steps(raw_responses[idx], DEBATE_STEP_FIELDS)
            
            response_details.append({
                "idx": idx,
                "label": predictions[idx],
                "confidence": confidences[idx],
                "decision": decisions[idx],
                "temperature": temp,
                "is_fallback": fallback_flags[idx],
                "raw_response": raw_responses[idx],
                "steps": parsed_steps,
                "usage": usage_metas[idx] if usage_metas[idx] is not None else {},
                "api": api_metas[idx] if api_metas[idx] is not None else {},
                "parse": parse_metas[idx] if parse_metas[idx] is not None else {},
            })

        valid_predictions = [p for p in predictions if p]
        vote_counts = Counter(valid_predictions)
        if not vote_counts:
            vote_counts = Counter({"General": 1})
        majority_label = vote_counts.most_common(1)[0][0]

        return {
            "majority_label": majority_label,
            "vote_counts": dict(vote_counts),
            "vote_inputs": {
                "k_samples": k,
                "predictions": predictions,
                "confidences": confidences,
                "decisions": decisions,
                "vote_counts": dict(vote_counts),
                "majority_label": majority_label,
                "majority_votes": vote_counts[majority_label],
                "consistency": vote_counts[majority_label] / max(1, k),
            },
            "predictions": predictions,
            "raw_responses": raw_responses,
            "decisions": decisions,
            "confidences": confidences,
            "consistency": vote_counts[majority_label] / max(1, k),
            "response_details": response_details,
        }

    def _build_debate_metadata(self,
                               grok_round0: Dict[str, Any],
                               gemini_round0: Dict[str, Any],
                               final_evasion: str,
                               final_clarity: str,
                               decision_reason: str,
                               final_clarity_votes_weighted: Dict[str, float],
                               final_vote_margin: float,
                               round0_static_clarity_votes_weighted: Dict[str, float],
                               api_calls_estimated: int,
                               debate_log: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "grok": grok_round0,
            "gemini": gemini_round0,
            "ensemble": {
                "decision_reason": decision_reason,
                "final_clarity": final_clarity,
                "final_evasion": final_evasion,
                "final_clarity_votes_weighted": final_clarity_votes_weighted,
                "final_vote_margin": final_vote_margin,
                "round0_static_clarity_votes_weighted": round0_static_clarity_votes_weighted,
                "grok_weight": self.GROK_BASE_WEIGHT,
                "gemini_weight": self.GEMINI_BASE_WEIGHT,
                "api_calls_estimated": api_calls_estimated,
            },
            "debate": debate_log,
        }

    def classify(self, question: str, answer: str,
                 full_question: str = None) -> Tuple[str, str, float, Dict]:
        """
        Debate classification pipeline.

        Returns:
            (final_evasion_label, final_clarity_label, confidence, metadata)
        """
        print("    [PARALLEL] Round 0: Running Grok + Gemini simultaneously...")

        estimated_api_calls = 2 * self.k_samples

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

        print(f"    [Round0 Grok] {grok_evasion} (cons={grok_meta['consistency']:.2f})")
        print(f"    [Round0 Gemini] {gemini_evasion} (cons={gemini_meta['consistency']:.2f})")

        round0_static_clarity, round0_static_evasion, round0_static_reason = self._round0_static_weighted(
            grok_clarity, grok_evasion, grok_meta, gemini_clarity, gemini_evasion
        )
        round0_static_clarity_votes_weighted = self._round0_static_clarity_votes(
            grok_meta, gemini_clarity
        )

        debate_history = [{
            "round": 0,
            "grok_votes": dict(Counter(grok_meta.get("vote_counts", {}))),
            "gemini_votes": dict(Counter(gemini_meta.get("vote_counts", {}))),
            "grok_evasion": grok_evasion,
            "gemini_evasion": gemini_evasion,
            "grok_clarity": grok_clarity,
            "gemini_clarity": gemini_clarity,
        }]

        grok_round0 = {
            "evasion_label": grok_evasion,
            "clarity_label": grok_clarity,
            "confidence": grok_conf,
            "consistency": grok_meta.get("consistency", 0.0),
            "vote_counts": grok_meta.get("vote_counts", {}),
            "vote_inputs": grok_meta.get("vote_inputs", {}),
            "avg_confidence": grok_meta.get("avg_confidence", 0.0),
            "response_details": grok_meta.get("response_details", []),
        }
        gemini_round0 = {
            "evasion_label": gemini_evasion,
            "clarity_label": gemini_clarity,
            "confidence": gemini_conf,
            "consistency": gemini_meta.get("consistency", 0.0),
            "vote_counts": gemini_meta.get("vote_counts", {}),
            "vote_inputs": gemini_meta.get("vote_inputs", {}),
            "avg_confidence": gemini_meta.get("avg_confidence", 0.0),
            "response_details": gemini_meta.get("response_details", []),
        }

        debate_log = {
            "sample_id": None,
            "question": question,
            "answer": answer,
            "round0_grok": {
                "label": grok_evasion,
                "clarity": grok_clarity,
                "consistency": grok_meta.get("consistency", 0.0),
                "votes": grok_meta.get("vote_counts", {}),
                "vote_inputs": grok_meta.get("vote_inputs", {}),
            },
            "round0_gemini": {
                "label": gemini_evasion,
                "clarity": gemini_clarity,
                "consistency": gemini_meta.get("consistency", 0.0),
                "votes": gemini_meta.get("vote_counts", {}),
                "vote_inputs": gemini_meta.get("vote_inputs", {}),
            },
            "debate_triggered": False,
            "debate_reason": "round0_agreement" if grok_clarity == gemini_clarity else "clarity_disagreement",
            "rounds": [],
            "round0_static_clarity": round0_static_clarity,
            "round0_static_evasion": round0_static_evasion,
            "round0_static_decision": round0_static_reason,
            "final_decision": None,
            "final_clarity": None,
            "gold_clarity": None,
            "debate_helped": None,
            "api_calls_estimated": estimated_api_calls,
        }

        if grok_clarity == gemini_clarity:
            final_evasion = grok_evasion
            final_clarity = grok_clarity
            decision_reason = "AGREE_ROUND0"
            debate_log["final_decision"] = decision_reason
            debate_log["final_clarity"] = final_clarity
            final_clarity_votes_weighted = aggregate_debate_clarity_votes(
                debate_history,
                grok_weight=self.GROK_BASE_WEIGHT,
                gemini_weight=self.GEMINI_BASE_WEIGHT,
            )
            final_vote_margin = compute_vote_margin(final_clarity_votes_weighted)
            metadata = self._build_debate_metadata(
                grok_round0,
                gemini_round0,
                final_evasion,
                final_clarity,
                decision_reason,
                final_clarity_votes_weighted,
                final_vote_margin,
                round0_static_clarity_votes_weighted,
                estimated_api_calls,
                debate_log,
            )
            return final_evasion, final_clarity, (grok_conf + gemini_conf) / 2, metadata

        debate_log["debate_triggered"] = True

        grok_cot = extract_cot_summary(self._select_majority_response(grok_meta, grok_evasion))
        gemini_cot = extract_cot_summary(self._select_majority_response(gemini_meta, gemini_evasion))

        boundary_key = tuple(sorted([grok_clarity, gemini_clarity]))
        boundary_guidance = BOUNDARY_GUIDANCE.get(
            boundary_key, "Examine the disagreement carefully and reassess your reasoning."
        )

        vote_distributions = [self._compute_vote_distribution(debate_history[-1])]
        stable_streak = 0

        current_grok = {
            "label": grok_evasion,
            "clarity": grok_clarity,
            "consistency": grok_meta.get("consistency", 0.0),
            "votes": grok_meta.get("vote_counts", {}),
        }
        current_gemini = {
            "label": gemini_evasion,
            "clarity": gemini_clarity,
            "consistency": gemini_meta.get("consistency", 0.0),
            "votes": gemini_meta.get("vote_counts", {}),
        }

        user_prompt = self._build_user_prompt(question, answer, full_question)

        for debate_round in range(1, self.max_debate_rounds + 1):
            print(f"    [DEBATE] Round {debate_round}/{self.max_debate_rounds}")

            grok_debate_prompt = DEBATE_SYSTEM_PROMPT.format(
                own_label=current_grok["label"],
                own_clarity=current_grok["clarity"],
                own_consistency=f"{current_grok['consistency']:.0%}",
                k=self.k_samples,
                own_votes=f"{int(current_grok['consistency'] * self.k_samples)}/{self.k_samples}",
                own_cot_summary=grok_cot,
                other_label=current_gemini["label"],
                other_clarity=current_gemini["clarity"],
                other_consistency=f"{current_gemini['consistency']:.0%}",
                other_votes=f"{int(current_gemini['consistency'] * self.k_samples)}/{self.k_samples}",
                other_cot_summary=gemini_cot,
                boundary_specific_guidance=boundary_guidance,
            )
            gemini_debate_prompt = DEBATE_SYSTEM_PROMPT.format(
                own_label=current_gemini["label"],
                own_clarity=current_gemini["clarity"],
                own_consistency=f"{current_gemini['consistency']:.0%}",
                k=self.k_samples,
                own_votes=f"{int(current_gemini['consistency'] * self.k_samples)}/{self.k_samples}",
                own_cot_summary=gemini_cot,
                other_label=current_grok["label"],
                other_clarity=current_grok["clarity"],
                other_consistency=f"{current_grok['consistency']:.0%}",
                other_votes=f"{int(current_grok['consistency'] * self.k_samples)}/{self.k_samples}",
                other_cot_summary=grok_cot,
                boundary_specific_guidance=boundary_guidance,
            )

            with ThreadPoolExecutor(max_workers=2) as executor:
                grok_future = executor.submit(
                    self._run_debate_samples,
                    self.grok_client,
                    grok_debate_prompt,
                    user_prompt,
                    self.debate_k_samples,
                    True,
                )
                gemini_future = executor.submit(
                    self._run_debate_samples,
                    self.gemini_client,
                    gemini_debate_prompt,
                    user_prompt,
                    self.debate_k_samples,
                    False,
                )
                grok_debate_results = grok_future.result()
                gemini_debate_results = gemini_future.result()

            estimated_api_calls += 2 * self.debate_k_samples

            prev_grok_label = current_grok["label"]
            prev_gemini_label = current_gemini["label"]
            prev_grok_clarity = current_grok["clarity"]
            prev_gemini_clarity = current_gemini["clarity"]

            current_grok = {
                "label": grok_debate_results["majority_label"],
                "clarity": EVASION_TO_CLARITY.get(grok_debate_results["majority_label"], "Ambivalent"),
                "consistency": grok_debate_results["consistency"],
                "votes": grok_debate_results["vote_counts"],
            }
            current_gemini = {
                "label": gemini_debate_results["majority_label"],
                "clarity": EVASION_TO_CLARITY.get(gemini_debate_results["majority_label"], "Ambivalent"),
                "consistency": gemini_debate_results["consistency"],
                "votes": gemini_debate_results["vote_counts"],
            }

            round_data = {
                "round": debate_round,
                "grok_votes": dict(Counter(grok_debate_results["vote_counts"])),
                "gemini_votes": dict(Counter(gemini_debate_results["vote_counts"])),
                "grok_evasion": current_grok["label"],
                "gemini_evasion": current_gemini["label"],
                "grok_clarity": current_grok["clarity"],
                "gemini_clarity": current_gemini["clarity"],
            }
            debate_history.append(round_data)

            vote_distributions.append(self._compute_vote_distribution(round_data))
            stability_metric = compute_distribution_shift(vote_distributions[-2], vote_distributions[-1])
            stable_transition = check_debate_stability(vote_distributions, self.STABILITY_THRESHOLD)
            if stable_transition:
                stable_streak += 1
            else:
                stable_streak = 0

            debate_log["rounds"].append({
                "round": debate_round,
                "boundary_type": f"{min(prev_grok_clarity, prev_gemini_clarity)} vs {max(prev_grok_clarity, prev_gemini_clarity)}",
                "grok": {
                    "majority_label": current_grok["label"],
                    "clarity": current_grok["clarity"],
                    "consistency": current_grok["consistency"],
                    "vote_counts": grok_debate_results["vote_counts"],
                    "vote_inputs": grok_debate_results.get("vote_inputs", {}),
                    "changed_from_previous": current_grok["label"] != prev_grok_label,
                    "responses": grok_debate_results["response_details"],
                },
                "gemini": {
                    "majority_label": current_gemini["label"],
                    "clarity": current_gemini["clarity"],
                    "consistency": current_gemini["consistency"],
                    "vote_counts": gemini_debate_results["vote_counts"],
                    "vote_inputs": gemini_debate_results.get("vote_inputs", {}),
                    "changed_from_previous": current_gemini["label"] != prev_gemini_label,
                    "responses": gemini_debate_results["response_details"],
                },
                "stability_metric": stability_metric,
                "stable": stable_transition,
            })

            if current_grok["clarity"] == current_gemini["clarity"]:
                final_evasion = current_grok["label"]
                final_clarity = current_grok["clarity"]
                decision_reason = f"AGREE_ROUND{debate_round}"
                debate_log["final_decision"] = decision_reason
                debate_log["final_clarity"] = final_clarity
                debate_log["api_calls_estimated"] = estimated_api_calls
                final_clarity_votes_weighted = aggregate_debate_clarity_votes(
                    debate_history,
                    grok_weight=self.GROK_BASE_WEIGHT,
                    gemini_weight=self.GEMINI_BASE_WEIGHT,
                )
                final_vote_margin = compute_vote_margin(final_clarity_votes_weighted)
                metadata = self._build_debate_metadata(
                    grok_round0,
                    gemini_round0,
                    final_evasion,
                    final_clarity,
                    decision_reason,
                    final_clarity_votes_weighted,
                    final_vote_margin,
                    round0_static_clarity_votes_weighted,
                    estimated_api_calls,
                    debate_log,
                )
                return final_evasion, final_clarity, 0.9, metadata

            if stable_streak >= self.STABILITY_REQUIRED_CONSECUTIVE:
                print(
                    f"    [DEBATE] Stable vote distribution for "
                    f"{self.STABILITY_REQUIRED_CONSECUTIVE} transitions, stopping early."
                )
                break

            grok_cot = extract_cot_summary(self._select_majority_response(
                grok_debate_results, current_grok["label"]
            ))
            gemini_cot = extract_cot_summary(self._select_majority_response(
                gemini_debate_results, current_gemini["label"]
            ))

        final_clarity = aggregate_debate_votes(
            debate_history,
            grok_weight=self.GROK_BASE_WEIGHT,
            gemini_weight=self.GEMINI_BASE_WEIGHT,
        )
        if final_clarity == current_grok["clarity"]:
            final_evasion = current_grok["label"]
        elif final_clarity == current_gemini["clarity"]:
            final_evasion = current_gemini["label"]
        else:
            final_evasion = current_grok["label"]

        decision_reason = f"DEBATE_AGGREGATED_R{len(debate_history) - 1}"
        debate_log["final_decision"] = decision_reason
        debate_log["final_clarity"] = final_clarity
        debate_log["api_calls_estimated"] = estimated_api_calls
        final_clarity_votes_weighted = aggregate_debate_clarity_votes(
            debate_history,
            grok_weight=self.GROK_BASE_WEIGHT,
            gemini_weight=self.GEMINI_BASE_WEIGHT,
        )
        final_vote_margin = compute_vote_margin(final_clarity_votes_weighted)

        metadata = self._build_debate_metadata(
            grok_round0,
            gemini_round0,
            final_evasion,
            final_clarity,
            decision_reason,
            final_clarity_votes_weighted,
            final_vote_margin,
            round0_static_clarity_votes_weighted,
            estimated_api_calls,
            debate_log,
        )
        return final_evasion, final_clarity, 0.7, metadata


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
        description=" Debate/Ultimate political evasion classifier"
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
        help="Path to evaluation CSV file (if provided, uses CSV instead of HuggingFace dataset)"
    )
    parser.add_argument(
        "--eval_csv_gold",
        type=str,
        default=None,
        help="Path to evaluation CSV WITH gold labels (annotator1/2 + clarity_label). "
             "Runs full gold-label evaluation with metrics, same JSON schema as test set."
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
        default="debate",
        help="Prefix for output files"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="debate",
        choices=["ultimate", "debate"],
        help="Classifier mode: ultimate (static weighted voting) or debate (iterative deliberation)"
    )
    parser.add_argument(
        "--k_samples",
        type=int,
        default=5,
        help="Number of samples for self-consistency (default: 5)"
    )
    parser.add_argument(
        "--max_debate_rounds",
        type=int,
        default=2,
        help="Maximum debate rounds for disagreement cases (default: 2)"
    )
    parser.add_argument(
        "--debate_k",
        type=int,
        default=3,
        help="Number of samples per model in each debate round (default: 3)"
    )
    parser.add_argument(
        "--thinking_level",
        type=str,
        default="high",
        choices=["minimal", "low", "medium", "high"],
        help="Gemini 3 thinking level: minimal/low/medium/high (default: high)"
    )
    parser.add_argument(
        "--save_reasoning",
        action="store_true",
        help="Save detailed reasoning to file"
    )

    args = parser.parse_args()

    if not args.grok_api_key:
        raise ValueError("Grok API key required. Set --grok_api_key or XAI_API_KEY env var")
    if not args.gemini_api_key:
        raise ValueError("Gemini API key required. Set --gemini_api_key or GEMINI_API_KEY env var")

    print("\n" + "=" * 70)
    if args.mode == "debate":
        print(" Iterative Multi-Agent Deliberation")
        print("=" * 70)
        print("\nDebate Strategy:")
        print(f"   1. Round 0: Grok (k={args.k_samples}) + Gemini (k={args.k_samples}) independently")
        print("   2. If clarity agrees -> early exit")
        print(f"   3. If disagree -> debate rounds (max={args.max_debate_rounds}, k={args.debate_k})")
        print("   4. Aggregate with recency-weighted voting across rounds")
    else:
        
        print(f"   1. Run Grok (k={args.k_samples}) and Gemini (k={args.k_samples}) independently")
        print("   2. If both agree -> use that prediction")
        print(
            f"   3. If disagree -> weighted voting: Grok 5 votes + Gemini {APEXUltimateClassifier.GEMINI_WEIGHT} votes"
        )

    print("\n Initializing clients...")
    grok_client = GrokClient(args.grok_api_key)
    gemini_client = GeminiClient(args.gemini_api_key, thinking_level=args.thinking_level)
    if args.mode == "debate":
        classifier = APEXDebateClassifier(
            grok_client,
            gemini_client,
            k_samples=args.k_samples,
            max_debate_rounds=args.max_debate_rounds,
            debate_k_samples=args.debate_k,
        )
    else:
        classifier = APEXUltimateClassifier(
            grok_client,
            gemini_client,
            k_samples=args.k_samples,
        )

    # Load data - either from CSV or HuggingFace dataset
    is_eval_gold_mode = args.eval_csv_gold is not None
    is_eval_mode = args.eval_csv is not None and not is_eval_gold_mode
    
    if is_eval_gold_mode:
        # NEW: Eval CSV WITH gold labels — full evaluation with metrics
        print(f"\n Loading evaluation set WITH GOLD from CSV: {args.eval_csv_gold}...")
        df = pd.read_csv(args.eval_csv_gold)
        # Map clarity labels to match train set naming convention
        df["clarity_label"] = df["clarity_label"].map(
            lambda x: EVAL_CLARITY_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else "Ambivalent"
        )
        # Map evasion labels (annotator columns) to match train set naming
        for ann_col in ["annotator1", "annotator2", "annotator3"]:
            if ann_col in df.columns:
                df[ann_col] = df[ann_col].apply(
                    lambda x: EVAL_EVASION_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) and str(x).strip() else ""
                )
        ds = df.to_dict('records')
        if args.max_samples:
            ds = ds[:args.max_samples]
        # Count gold availability
        n_with_ann = sum(1 for r in ds if str(r.get("annotator1", "")).strip())
        print(f"   Loaded {len(ds)} examples from CSV")
        print(f"   ✅ GOLD MODE: {n_with_ann} samples with annotator labels")
        cl_dist = Counter(r["clarity_label"] for r in ds)
        print(f"   Clarity distribution (mapped): {dict(cl_dist)}")
    elif is_eval_mode:
        print(f"\n Loading evaluation set from CSV: {args.eval_csv}...")
        df = pd.read_csv(args.eval_csv)
        # Convert DataFrame to list of dicts for consistent interface
        ds = df.to_dict('records')
        if args.max_samples:
            ds = ds[:args.max_samples]
        print(f"   Loaded {len(ds)} examples from CSV")
        print(f"   🔍 EVALUATION MODE: Gold labels are empty, metrics will be skipped")
    else:
        print(f"\n Loading QEvasion dataset (split: {args.split})...")
        ds = load_dataset("ailsntua/QEvasion", split=args.split)
        if args.max_samples:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        print(f"   Loaded {len(ds)} examples")

    print(f"\n Starting {args.mode} classification...\n")

    all_results = []
    preds_evasion = []
    preds_clarity = []
    gold_multi = []
    total_api_calls_estimated = 0

    reasoning_file = None
    if args.save_reasoning:
        reasoning_file = open(f"{args.output_prefix}_reasoning.txt", "w", encoding="utf-8")

    for i, example in enumerate(ds):
        print(f"{'='*70}")
        print(f" SAMPLE {i+1}/{len(ds)}")
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

        gold_labels = extract_gold_labels(example) if (not is_eval_mode or is_eval_gold_mode) else []
        gold_multi.append(gold_labels)

        print(f"   Q: {question[:100]}{'...' if len(question) > 100 else ''}")
        print(f"   A: {answer[:150]}{'...' if len(answer) > 150 else ''}")

        # Classify
        evasion_label, clarity_label, confidence, metadata = classifier.classify(
            question, answer, full_question
        )

        preds_evasion.append(evasion_label)
        preds_clarity.append(clarity_label)

        grok_meta = metadata.get("grok", {})
        gemini_meta = metadata.get("gemini", {})
        ensemble_meta = metadata.get("ensemble", {})
        debate_meta = metadata.get("debate", {}) if isinstance(metadata.get("debate", {}), dict) else {}

        # Print result
        gold_clarity = example.get("clarity_label", "") if isinstance(example, dict) else ""
        if pd.isna(gold_clarity):
            gold_clarity = ""

        if is_eval_mode and not is_eval_gold_mode:
            is_evasion_correct = None
            is_clarity_correct = None
        else:
            is_evasion_correct = evasion_label in set(gold_labels) if gold_labels else False
            is_clarity_correct = clarity_label == gold_clarity

        if debate_meta:
            debate_meta["sample_id"] = i
            debate_meta["gold_clarity"] = None if is_eval_mode else gold_clarity

            if is_eval_mode and not is_eval_gold_mode:
                debate_meta["debate_helped"] = None
            else:
                round0_static = debate_meta.get("round0_static_clarity")
                changed_from_round0_static = round0_static is not None and clarity_label != round0_static
                debate_meta["debate_helped"] = (
                    bool(debate_meta.get("debate_triggered", False))
                    and changed_from_round0_static
                    and bool(is_clarity_correct)
                    and round0_static != gold_clarity
                )

        # Format vote counts for display
        def format_votes(votes_dict):
            if not votes_dict:
                return "N/A"
            sorted_votes = sorted(votes_dict.items(), key=lambda x: x[1], reverse=True)
            return ", ".join(f"{label}({count})" for label, count in sorted_votes)

        grok_votes_str = format_votes(grok_meta.get("vote_counts", {}))
        gemini_votes_str = format_votes(gemini_meta.get("vote_counts", {}))
        decision_reason = ensemble_meta.get("decision_reason", "UNKNOWN")
        api_calls_estimated = ensemble_meta.get("api_calls_estimated")
        if isinstance(api_calls_estimated, (int, float)):
            total_api_calls_estimated += int(api_calls_estimated)

        # --- Compute analysis fields ---
        text_feats = compute_text_features(question, answer, full_question)
        text_feats["question"] = question
        text_feats["answer"] = answer
        text_feats["full_question"] = full_question

        annotator = compute_annotator_analysis(gold_labels) if ((not is_eval_mode or is_eval_gold_mode) and gold_labels) else {}

        # Build round0 blocks from metadata
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

        grok_r0 = build_round0_block(grok_meta)
        gemini_r0 = build_round0_block(gemini_meta)

        # Evaluation flags
        if is_eval_mode and not is_eval_gold_mode:
            eval_block = {}
        else:
            clarity_safe = is_clarity_safe_evasion_error(evasion_label, gold_labels) if gold_labels else None
            eval_block = {
                "evasion_correct": is_evasion_correct,
                "clarity_correct": is_clarity_correct,
                "grok_evasion_correct": (grok_meta.get("evasion_label", "") in set(gold_labels)) if gold_labels else None,
                "gemini_evasion_correct": (gemini_meta.get("evasion_label", "") in set(gold_labels)) if gold_labels else None,
                "grok_clarity_correct": (grok_meta.get("clarity_label", "") == gold_clarity) if gold_clarity else None,
                "gemini_clarity_correct": (gemini_meta.get("clarity_label", "") == gold_clarity) if gold_clarity else None,
                "clarity_safe_evasion_error": clarity_safe,
            }

        # Debate block
        if args.mode == "debate" and debate_meta:
            debate_block = {
                "triggered": debate_meta.get("debate_triggered", False),
                "trigger_reason": debate_meta.get("debate_reason", ""),
                "round0_static_clarity": debate_meta.get("round0_static_clarity"),
                "round0_static_evasion": debate_meta.get("round0_static_evasion"),
                "round0_static_decision": debate_meta.get("round0_static_decision"),
                "rounds": debate_meta.get("rounds", []),
                "final_clarity": debate_meta.get("final_clarity"),
                "final_decision": debate_meta.get("final_decision"),
            }
            if not is_eval_mode or is_eval_gold_mode:
                eval_block["debate_helped"] = debate_meta.get("debate_helped")
        else:
            debate_block = {"triggered": False, "rounds": []}

        result = {
            "index": i,
            "mode": args.mode,
            "text_features": text_feats,
            "gold": {
                "evasion_labels": gold_labels,
                "clarity_label": gold_clarity,
                **annotator,
            } if (not is_eval_mode or is_eval_gold_mode) else {},
            "grok_round0": grok_r0,
            "gemini_round0": gemini_r0,
            "ensemble": {
                "final_evasion": evasion_label,
                "final_clarity": clarity_label,
                "confidence": confidence,
                "decision_reason": decision_reason,
                "final_clarity_votes_weighted": ensemble_meta.get("final_clarity_votes_weighted", {}),
                "final_vote_margin": ensemble_meta.get("final_vote_margin", 0.0),
                "round0_static_clarity_votes_weighted": ensemble_meta.get("round0_static_clarity_votes_weighted", {}),
                "grok_weight": 5,
                "gemini_weight": APEXUltimateClassifier.GEMINI_WEIGHT,
                "api_calls_estimated": api_calls_estimated,
            },
            "debate": debate_block,
            "evaluation": eval_block,
        }
        all_results.append(result)

        print(f"\n    RESULT:")
        print(f"      Grok votes:   {grok_votes_str}")
        print(
            f"      Grok final:   {grok_meta.get('evasion_label', '')} -> "
            f"{grok_meta.get('clarity_label', '')} (cons={grok_meta.get('consistency', 0.0):.2f})"
        )
        print(f"      Gemini votes: {gemini_votes_str}")
        print(
            f"      Gemini final: {gemini_meta.get('evasion_label', '')} -> "
            f"{gemini_meta.get('clarity_label', '')} (cons={gemini_meta.get('consistency', 0.0):.2f})"
        )
        print(f"      Decision: {decision_reason}")
        if api_calls_estimated is not None:
            print(f"      Estimated API calls (sample): {api_calls_estimated}")
            print(f"      Estimated API calls (running total): {total_api_calls_estimated}")
        if args.mode == "debate":
            print(f"      Debate triggered: {debate_meta.get('debate_triggered', False)}")
            if debate_meta.get("debate_triggered", False):
                print(f"      Debate rounds executed: {len(debate_meta.get('rounds', []))}")
                if "round0_static_clarity" in debate_meta:
                    print(
                        f"      Round0 static baseline: {debate_meta.get('round0_static_clarity')} "
                        f"({debate_meta.get('round0_static_decision')})"
                    )
        print("")

        if is_eval_mode and not is_eval_gold_mode:
            print(f"      EVASION:  Pred={evasion_label:<20} (Gold: N/A - Eval mode)")
            print(f"      CLARITY:  Pred={clarity_label:<20} (Gold: N/A - Eval mode)")
        else:
            print(f"      EVASION:  Pred={evasion_label:<20} Gold={gold_labels}  {'OK' if is_evasion_correct else 'NG'}")
            print(f"      CLARITY:  Pred={clarity_label:<20} Gold={gold_clarity:<15}  {'OK' if is_clarity_correct else 'NG'}")
            if args.mode == "debate" and debate_meta.get("debate_triggered", False):
                print(f"      DEBATE_HELPED: {debate_meta.get('debate_helped')}")

        # Compute running F1 scores only if not in eval mode
        if not is_eval_mode or is_eval_gold_mode:
            golds_clarity_so_far = [
                ds[j].get("clarity_label", "").strip() if isinstance(ds[j], dict) else ds[j]["clarity_label"]
                for j in range(i + 1)
            ]
            clarity_f1_so_far = f1_score(golds_clarity_so_far, preds_clarity, average="macro", zero_division=0)
            evasion_acc_so_far = compute_instance_accuracy(gold_multi, preds_evasion)
            evasion_f1_so_far, _ = compute_macro_f1(gold_multi, preds_evasion)

            print("")
            print(f"    RUNNING METRICS ({i+1} samples):")
            print(f"      Clarity Macro F1:  {clarity_f1_so_far:.4f}")
            print(f"      Evasion Macro F1:  {evasion_f1_so_far:.4f}  (Accuracy: {evasion_acc_so_far:.4f})")
        print()

        if reasoning_file:
            reasoning_file.write(f"\n{'='*80}\n")
            reasoning_file.write(f"SAMPLE {i}\n")
            reasoning_file.write(f"{'='*80}\n\n")
            reasoning_file.write(f"Mode: {args.mode}\n")
            reasoning_file.write(f"Question: {question}\n\n")
            reasoning_file.write(f"Answer: {answer}\n\n")
            reasoning_file.write(f"Gold evasion: {gold_labels}\n")
            reasoning_file.write(f"Gold clarity: {gold_clarity}\n\n")
            reasoning_file.write(f"--- GROK ---\n")
            reasoning_file.write(f"Votes: {grok_votes_str}\n")
            reasoning_file.write(f"Final: {grok_meta.get('evasion_label', '')} -> {grok_meta.get('clarity_label', '')}\n")
            reasoning_file.write(f"Consistency: {grok_meta.get('consistency', 0.0):.2f}\n\n")
            reasoning_file.write(f"--- GEMINI ---\n")
            reasoning_file.write(f"Votes: {gemini_votes_str}\n")
            reasoning_file.write(f"Final: {gemini_meta.get('evasion_label', '')} -> {gemini_meta.get('clarity_label', '')}\n")
            reasoning_file.write(f"Consistency: {gemini_meta.get('consistency', 0.0):.2f}\n\n")
            reasoning_file.write(f"--- ENSEMBLE ---\n")
            reasoning_file.write(f"Decision: {decision_reason}\n")
            reasoning_file.write(f"Estimated API calls: {api_calls_estimated}\n")
            reasoning_file.write(f"Final evasion: {evasion_label}\n")
            reasoning_file.write(f"Final clarity: {clarity_label}\n")
            reasoning_file.write(f"Evasion correct: {is_evasion_correct}\n")
            reasoning_file.write(f"Clarity correct: {is_clarity_correct}\n")
            if args.mode == "debate" and debate_meta:
                reasoning_file.write("\n--- DEBATE LOG ---\n")
                reasoning_file.write(json.dumps(debate_meta, ensure_ascii=False, indent=2))
                reasoning_file.write("\n")
            reasoning_file.write("\n")
            reasoning_file.flush()

    if reasoning_file:
        reasoning_file.close()

    # Final Metrics (skip if evaluation mode)

    if is_eval_mode and not is_eval_gold_mode:
        print("\n" + "="*70)
        print(" EVALUATION MODE - METRICS SKIPPED (no gold labels)")
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
        print(" FINAL METRICS")
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

        print("\n CLARITY METRICS (Task 1)")
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

    print("\n" + "="*70)
    print(" SAVING RESULTS")
    print("="*70)

    # Evasion pickle
    evasion_file = f"{args.output_prefix}_evasion.pkl"
    with open(evasion_file, "wb") as f:
        pickle.dump([list(range(len(preds_evasion))), preds_evasion], f)
    print(f"   Saved evasion labels to: {evasion_file}")

    # Clarity pickle
    clarity_file = f"{args.output_prefix}_clarity.pkl"
    with open(clarity_file, "wb") as f:
        pickle.dump([list(range(len(preds_clarity))), preds_clarity], f)
    print(f"   Saved clarity labels to: {clarity_file}")

    # Full results JSON
    results_json_file = f"{args.output_prefix}_full.json"
    with open(results_json_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"   Saved full results to: {results_json_file}")

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
            "mode": args.mode,
            "k_samples": args.k_samples,
            "max_debate_rounds": args.max_debate_rounds if args.mode == "debate" else 0,
            "debate_k": args.debate_k if args.mode == "debate" else 0,
            "grok_model": "grok-4-1-fast-reasoning",
            "gemini_model": "gemini-3-flash-preview",
            "gemini_thinking_level": args.thinking_level,
            "gemini_weight": APEXUltimateClassifier.GEMINI_WEIGHT,
            "strategy": (
                "iterative_debate_adaptive_stopping"
                if args.mode == "debate"
                else "weighted_voting_agree_first"
            ),
            "is_eval_mode": is_eval_mode
        },
        "samples_processed": len(ds),
        "api_calls_estimated_total": total_api_calls_estimated,
        "api_calls_estimated_avg_per_sample": (
            total_api_calls_estimated / len(ds) if len(ds) > 0 else 0
        ),
        "label_fallbacks": dict(FALLBACK_COUNTER),
    }

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"   Saved metrics to: {metrics_file}")

    print("\n" + "="*70)
    if args.mode == "debate":
        print(" DEBATE PROCESSING COMPLETE!")
    else:
        print(" ULTIMATE PROCESSING COMPLETE!")
    if is_eval_mode:
        print(f"   Processed {len(ds)} samples in EVALUATION mode")
        print(f"   Predictions saved to {clarity_file}")
    else:
        print(f"   Clarity Macro F1: {clarity_macro_f1:.4f}")
    print(f"   Estimated API calls total: {total_api_calls_estimated}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
