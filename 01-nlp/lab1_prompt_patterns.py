import json
import time
from typing import Any

from common import fallback_reasoning, fallback_sentiment_label, get_client, load_reviews, maybe_sleep


MODEL = "gpt-4o-mini"


def build_prompt(text: str, prompt_type: str) -> str:
    """Build the prompt used to compare the three classification patterns."""
    if prompt_type == "zero_shot":
        instructions = "Classify the review as POSITIVE, NEGATIVE, or NEUTRAL."
    elif prompt_type == "few_shot":
        instructions = (
            "Use these examples: 'I love this product' = POSITIVE; "
            "'It arrived broken' = NEGATIVE; 'It is an ordinary product' = NEUTRAL. "
            "Now classify the review as POSITIVE, NEGATIVE, or NEUTRAL."
        )
    else:
        instructions = (
            "Classify the review as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Briefly explain the wording that led to the classification, "
            "including any sarcasm or irony."
        )

    return (
        f"{instructions}\n\nReview: {text}\n\n"
        'Return only JSON with this shape: {"label": "...", "reasoning": "..."}'
    )


def classify_sentiment(text: str, prompt_type: str, client: Any = None) -> dict[str, Any]:
    """
    Classify one review with OpenAI, or use local logic when no key is configured.
    """
    if client is None:
        label = fallback_sentiment_label(text)
        reasoning = fallback_reasoning(text, prompt_type)
        return {
            "label": label,
            "reasoning": reasoning,
            "latency_ms": 12.0,
        }

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You classify product reviews accurately."},
            {"role": "user", "content": build_prompt(text, prompt_type)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=120,
    )
    content = response.choices[0].message.content or "{}"
    result = json.loads(content)
    label = str(result.get("label", "NEUTRAL")).upper()
    if label not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        raise ValueError(f"Unexpected sentiment label: {label}")

    return {
        "label": label,
        "reasoning": str(result.get("reasoning", "")),
        "latency_ms": (time.perf_counter() - started) * 1000,
    }


def run_lab1() -> None:
    """Run the three prompt pattern variants over the review dataset."""
    reviews = load_reviews()
    client = get_client()
    results: list[dict[str, Any]] = []

    print("=== Lab 1: Prompt pattern comparison ===")
    mode = "OpenAI API" if client is not None else "local fallback"
    print(f"Using {mode}.\n")

    for review in reviews:
        row = {
            "id": review["id"],
            "text": review["text"],
            "ground_truth": review["label"],
        }

        for pattern in ["zero_shot", "few_shot", "cot"]:
            result = classify_sentiment(review["text"], pattern, client)
            row[f"{pattern}_label"] = result["label"]
            row[f"{pattern}_latency_ms"] = result["latency_ms"]
            if pattern == "cot":
                row["cot_reasoning"] = result.get("reasoning", "")

        results.append(row)
        maybe_sleep(0.05)

    print(f"Classified {len(results)} reviews across 3 prompt patterns.\n")

    # Accuracy per pattern
    for pattern in ["zero_shot", "few_shot", "cot"]:
        col = f"{pattern}_label"
        acc = sum(1 for row in results if row[col] == row["ground_truth"]) / len(results)
        avg_lat = sum(row[f"{pattern}_latency_ms"] for row in results) / len(results)
        print(f"{pattern.upper():>12}: Accuracy={acc:.0%} | Avg latency={avg_lat:.0f}ms")

    print("\n--- Where patterns disagree ---")
    disagreement: list[dict[str, Any]] = [
        {
            "id": row["id"],
            "text": row["text"],
            "ground_truth": row["ground_truth"],
            "zero_shot": row["zero_shot_label"],
            "few_shot": row["few_shot_label"],
            "cot": row["cot_label"],
        }
        for row in results
        if row["zero_shot_label"] != row["few_shot_label"]
        or row["few_shot_label"] != row["cot_label"]
    ]

    for row in disagreement:
        print(row)

    print("\n--- Sarcasm spotlight (review #7) ---")
    sarcastic = next(row for row in results if row["id"] == 7)
    print(f"Review: {sarcastic['text']}")
    print(f"Ground truth: {sarcastic['ground_truth']}")
    print(f"Zero-shot: {sarcastic['zero_shot_label']}")
    print(f"Few-shot: {sarcastic['few_shot_label']}")
    print(f"CoT: {sarcastic['cot_label']}")
    print(f"\nCoT reasoning:\n{sarcastic['cot_reasoning']}")

    print("\n--- Learner conclusion ---")
    conclusions = {
        "When to use zero_shot": "Use it for quick and simple classification when you want a short answer.",
        "When to use few_shot": "Use it when a few examples help clarify the expected labels.",
        "When to use CoT": "Use it when you need better reasoning for tricky cases like sarcasm.",
        "Sarcasm handling": "The CoT style is better at catching irony and sarcasm than a short zero-shot prompt.",
    }
    for key, value in conclusions.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_lab1()
