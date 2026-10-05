import json
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from openai import OpenAI


load_dotenv()

# Ensure UTF-8 console output on Windows (arrows, warning glyphs, etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

DATA_PATH = Path(__file__).resolve().parent / "data" / "support_queries.json"

# When no OpenAI API key is configured, the script uses a simple heuristic classifier.

client = OpenAI()
USE_OPENAI = True``



def load_queries(path: Path) -> list[dict]:
    """Load a JSON file containing labeled support queries."""
    text = path.read_text(encoding="utf-8")
    queries = json.loads(text)
    return queries


def heuristic_score(text: str) -> float:
    """A simple heuristic score for escalation probability when API is unavailable."""
    escalator = ["charge", "refund", "accessed", "suspended", "fraud", "locked", "stolen", "dispute"]
    auto = ["how do i", "how to", "what are", "where can i", "can i", "change my", "download", "set up"]
    text_lower = text.lower()
    score = 0.5

    for word in escalator:
        if word in text_lower:
            score += 0.15
    for word in auto:
        if word in text_lower:
            score -= 0.12

    return float(np.clip(score, 0.0, 1.0))


def classify_query(text: str) -> dict:
    """Classify a support query and return an escalation score plus reasoning."""
    prompt = (
        "You are a customer support triage classifier.\n\n"
        "Given a support query, return a JSON object with:\n"
        "- \"score\": a float from 0.0 to 1.0 where:\n"
        "    0.0 = definitely auto-resolvable\n"
        "    0.5 = ambiguous\n"
        "    1.0 = definitely needs human escalation\n"
        "- \"reasoning\": one sentence explaining the score\n\n"
        f"Query: {text}\n\n"
        "Respond ONLY with valid JSON."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=100,
    )

    raw = response.choices[0].message.content.strip()
    try:
        result = json.loads(raw)
        return {
            "score": float(result.get("score", 0.5)),
            "reasoning": result.get("reasoning", "No reasoning returned."),
        }
    except json.JSONDecodeError:
        return {"score": heuristic_score(text), "reasoning": "Fallback heuristic due to invalid model output."}

    return {"score": heuristic_score(text), "reasoning": "Used local heuristic classifier."}


def route_query(score: float, threshold: float) -> str:
    """Route a query to ESCALATE when the score meets or exceeds the threshold."""
    return "ESCALATE" if score >= threshold else "AUTO"


def evaluate_batch(queries: list[dict], threshold: float) -> dict:
    """Run the classifier on a batch and compute basic metrics."""
    predictions = []
    for q in queries:
        result = classify_query(q["text"])
        predicted = route_query(result["score"], threshold)
        predictions.append(
            {
                "id": q["id"],
                "text": q["text"],
                "ground_truth": q["label"],
                "score": result["score"],
                "predicted": predicted,
                "correct": predicted == q["label"],
                "reasoning": result["reasoning"],
            }
        )
        time.sleep(0.1)

    df = pd.DataFrame(predictions)
    tp = len(df[(df["predicted"] == "ESCALATE") & (df["ground_truth"] == "ESCALATE")])
    fp = len(df[(df["predicted"] == "ESCALATE") & (df["ground_truth"] == "AUTO")])
    tn = len(df[(df["predicted"] == "AUTO") & (df["ground_truth"] == "AUTO")])
    fn = len(df[(df["predicted"] == "AUTO") & (df["ground_truth"] == "ESCALATE")])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    false_escalation_rate = fp / len(df) if len(df) > 0 else 0.0
    missed_escalation_rate = fn / len(df) if len(df) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_escalation_rate": false_escalation_rate,
        "missed_escalation_rate": missed_escalation_rate,
        "accuracy": df["correct"].mean() if len(df) > 0 else 0.0,
        "predictions": predictions,
    }


def update_threshold(
    current_threshold: float,
    batch_results: dict,
    target_precision: float = 0.85,
    learning_rate: float = 0.05,
    min_threshold: float = 0.30,
    max_threshold: float = 0.90,
) -> tuple[float, str]:
    """Adjust the routing threshold using a simple feedback rule."""
    precision = batch_results["precision"]
    recall = batch_results["recall"]

    if precision < target_precision:
        new_threshold = min(current_threshold + learning_rate, max_threshold)
        reason = f"Precision {precision:.0%} < target {target_precision:.0%} → raised threshold"
    elif precision > target_precision + 0.10 and recall < 0.90:
        new_threshold = max(current_threshold - learning_rate, min_threshold)
        reason = (
            f"Precision {precision:.0%} > target by more than 10 points and recall is low → lowered threshold"
        )
    else:
        new_threshold = current_threshold
        reason = f"Precision {precision:.0%} within target range → no threshold change"

    return new_threshold, reason


def print_metrics(name: str, results: dict) -> None:
    """Print a concise summary of batch metrics."""
    print(f"{name} metrics:")
    print(f"  Accuracy:               {results['accuracy']:.0%}")
    print(f"  Precision:              {results['precision']:.0%}")
    print(f"  Recall:                 {results['recall']:.0%}")
    print(f"  F1 score:               {results['f1']:.0%}")
    print(f"  False escalation rate:  {results['false_escalation_rate']:.0%}")
    print(f"  Missed escalation rate: {results['missed_escalation_rate']:.0%}")


def plot_threshold_effect(thresholds: list[float], accuracies: list[float]) -> None:
    """Plot accuracy vs threshold to visualize the decision boundary."""
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, accuracies, marker="o")
    plt.title("Adaptive threshold impact on accuracy")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main() -> None:
    queries = load_queries(DATA_PATH)
    print(f"Loaded {len(queries)} queries")
    print(
        "Label distribution:",
        pd.Series([q["label"] for q in queries]).value_counts().to_dict(),
    )
    print()

    batch_size = 17
    batch1 = queries[:batch_size]
    batch2 = queries[batch_size : batch_size * 2]
    batch3 = queries[batch_size * 2 : batch_size * 3]

    hardcoded_threshold = 0.5
    print(f"Running baseline batch 1 with threshold = {hardcoded_threshold}")
    baseline_results = evaluate_batch(batch1, hardcoded_threshold)
    print_metrics("Baseline", baseline_results)
    print()

    ambiguous_queries = [
        {"id": 51, "text": "I'm not sure if I should upgrade now or wait.", "label": "AUTO", "difficulty": "AMBIGUOUS"},
        {"id": 52, "text": "The payment went through but I didn't get access.", "label": "ESCALATE", "difficulty": "AMBIGUOUS"},
        {"id": 53, "text": "My subscription renews tomorrow — can I cancel today?", "label": "ESCALATE", "difficulty": "AMBIGUOUS"},
        {"id": 54, "text": "I think I chose the wrong plan, not sure which one.", "label": "AUTO", "difficulty": "AMBIGUOUS"},
        {"id": 55, "text": "There's a notification saying my account will be suspended.", "label": "ESCALATE", "difficulty": "AMBIGUOUS"},
    ]

    batch2_with_drift = batch2 + ambiguous_queries
    random.seed(42)
    random.shuffle(batch2_with_drift)

    print(f"Running drifted batch 2 with threshold = {hardcoded_threshold}")
    drift_results = evaluate_batch(batch2_with_drift, hardcoded_threshold)
    print_metrics("Drifted batch", drift_results)
    print()

    precision_drop = baseline_results["precision"] - drift_results["precision"]
    if precision_drop > 0.10:
        print(f"⚠️  DRIFT DETECTED: Precision dropped by {precision_drop:.0%}")
    else:
        print("No large precision drift detected.")
    print()

    new_threshold, reason = update_threshold(hardcoded_threshold, drift_results)
    print(f"Threshold update: {hardcoded_threshold:.2f} → {new_threshold:.2f}")
    print(f"Reason: {reason}")
    print()

    print(f"Running adapted batch 3 with threshold = {new_threshold}")
    adapted_results = evaluate_batch(batch3, new_threshold)
    print_metrics("Adapted batch", adapted_results)
    print()

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    accuracies = [evaluate_batch(batch1, t)["accuracy"] for t in thresholds]
    plot_threshold_effect(thresholds, accuracies)


if __name__ == "__main__":
    main()
