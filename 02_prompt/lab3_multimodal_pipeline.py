import base64
import json
from pathlib import Path
from typing import Any

from common import DATA_DIR, SAMPLE_MEETING_TRANSCRIPT, get_client, maybe_sleep

MODEL = "gpt-4o-mini"

FALLBACK_ACTION_ITEMS = {
    "meeting_summary": "Q2 review meeting covering sales performance, CRM rollout delay, and marketing budget approval",
    "action_items": [
        {"owner": "Priya", "task": "Prepare a recovery plan for North region sales", "deadline": "Friday"},
        {"owner": "Ramesh", "task": "Send revised CRM timeline to stakeholders", "deadline": "Thursday"},
        {"owner": "Anand", "task": "Get CFO sign-off on the digital campaign budget", "deadline": "Monday"},
    ],
    "decisions": ["CRM rollout delayed by 2 weeks", "40 lakh digital campaign budget under review"],
}

FALLBACK_CHART_DATA = {
    "chart_type": "Bar chart",
    "regions": ["North", "South", "East", "West"],
    "q1_values": {"North": 42, "South": 38, "East": 29, "West": 33},
    "q2_values": {"North": 37, "South": 41, "East": 31, "West": 35},
    "top_performer_q2": "South",
    "biggest_decline": "North",
    "insight": "The overall trend is slightly positive with South and East improving while North declined.",
}


def encode_image(path: Path) -> str:
    """Encode an image file to base64 for the vision API call."""
    with path.open("rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")


def transcribe_audio(path: Path, client: Any) -> str:
    """Transcribe the meeting audio with Whisper, or fall back to a sample transcript."""
    if client is None:
        return SAMPLE_MEETING_TRANSCRIPT

    with path.open("rb") as file_handle:
        response = client.audio.transcriptions.create(model="whisper-1", file=file_handle)
    return response.text


def extract_action_items(transcript: str, client: Any) -> dict[str, Any]:
    """Ask the model to pull structured action items out of the transcript."""
    if client is None:
        return FALLBACK_ACTION_ITEMS

    prompt = (
        "Read this meeting transcript and return a JSON object with keys: "
        "'meeting_summary' (one sentence), 'action_items' (a list of objects each "
        "with 'owner', 'task', 'deadline'), and 'decisions' (a list of strings).\n\n"
        f"Transcript:\n{transcript}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=500,
    )
    return json.loads(response.choices[0].message.content or "{}")


def analyze_chart(path: Path, client: Any) -> dict[str, Any]:
    """Ask a vision-capable model to read the bar chart image."""
    if client is None:
        return FALLBACK_CHART_DATA

    image_b64 = encode_image(path)
    prompt = (
        "Read this bar chart and return a JSON object with keys: chart_type, regions, "
        "q1_values (object mapping each region to its Q1 value), "
        "q2_values (object mapping each region to its Q2 value), "
        "top_performer_q2, biggest_decline, and insight (one sentence)."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    return json.loads(response.choices[0].message.content or "{}")


def run_lab3() -> None:
    """Run the three-step multimodal flow: transcribe, extract, analyse."""
    print("=== Lab 3: Multimodal pipeline ===")

    audio_path = DATA_DIR / "meeting_audio.wav"
    chart_path = DATA_DIR / "bar_chart.png"

    if not audio_path.exists() or not chart_path.exists():
        print("The data files are missing. Run 'python setup/generate_lab_data.py' first.")
        return

    client = get_client()
    mode = "OpenAI API" if client is not None else "local fallback"
    print(f"Using {mode}.\n")

    print("Step 1: Transcribe audio")
    transcript = transcribe_audio(audio_path, client)
    print(transcript)
    maybe_sleep(0.3)

    print("\nStep 2: Extract action items")
    action_items_json = extract_action_items(transcript, client)
    print(json.dumps(action_items_json, indent=2))
    maybe_sleep(0.3)

    print("\nStep 3: Analyse the chart image")
    chart_data = analyze_chart(chart_path, client)
    print(json.dumps(chart_data, indent=2))
    maybe_sleep(0.3)

    print("\nStep 4: Combine into meeting summary")
    meeting_summary = {
        "transcript": transcript,
        "action_items": action_items_json.get("action_items", []),
        "meeting_summary": action_items_json.get("meeting_summary", ""),
        "decisions": action_items_json.get("decisions", []),
        "chart_analysis": chart_data,
    }
    print(json.dumps(meeting_summary, indent=2))

    print("\n--- Hallucination check ---")
    print("Compare the action items and figures above against the transcript and")
    print("chart image printed in steps 1 and 3 - confirm nothing was invented or")
    print("dropped, and that names, deadlines, and numbers match exactly.")


if __name__ == "__main__":
    run_lab3()
