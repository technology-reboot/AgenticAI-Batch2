import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_MEETING_TRANSCRIPT = (
    "Good morning team. We need to review Q2 targets. "
    "Sales in the North region are down 12 percent versus last quarter. "
    "Action item: Priya to prepare a recovery plan by Friday. "
    "Also, the new CRM rollout is delayed by two weeks. "
    "Ramesh will send a revised timeline to all stakeholders by end of day Thursday. "
    "Finally, the marketing budget needs approval. "
    "Total ask is forty lakh rupees for the digital campaign. "
    "Anand to get CFO sign-off before next Monday."
)


def load_environment() -> Optional[str]:
    """Load environment variables from the project .env file if present."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("OPENAI_API_KEY")


def get_client():
    """Return an OpenAI client when an API key is available; otherwise None."""
    api_key = load_environment()
    if not api_key or api_key == "your_key_here":
        return None

    from openai import OpenAI

    return OpenAI(api_key=api_key)


def load_reviews() -> list[dict[str, Any]]:
    """Load the sample product reviews from the JSON data file."""
    reviews_path = DATA_DIR / "product_reviews.json"
    return json.loads(reviews_path.read_text(encoding="utf-8"))


def fallback_sentiment_label(text: str) -> str:
    """A small keyword-based fallback for training without an API key."""
    lowered = text.lower()

    # The sarcastic review should be treated as negative.
    if "premium quality" in lowered and "yeah right" in lowered:
        return "NEGATIVE"

    negative_words = ["terrible", "waste", "broken", "damaged", "joke", "bad", "poor"]
    positive_words = ["excellent", "incredible", "outstanding", "love", "recommend", "best", "exceeded"]

    if any(word in lowered for word in negative_words):
        return "NEGATIVE"
    if any(word in lowered for word in positive_words):
        return "POSITIVE"
    return "NEUTRAL"


def fallback_reasoning(text: str, prompt_type: str) -> str:
    """Create a simple reasoning string for the CoT fallback."""
    label = fallback_sentiment_label(text)
    return (
        f"[{prompt_type}] Detected a {label.lower()} tone based on the wording in the review."
    )


def maybe_sleep(seconds: float = 0.3) -> None:
    """Small delay to avoid hitting rate limits in the real API demo."""
    time.sleep(seconds)
