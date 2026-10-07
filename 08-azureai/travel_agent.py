import json
import os

import requests

from pathlib import Path
from typing import Any, Dict, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
PROJECT_ROOT = Path(__file__).resolve().parent

def load_environment() -> Dict[str, str]:
    """Load environment variables from .env and return them as a dictionary."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {key: os.getenv(key, "") for key in [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
        "OPENWEATHER_API_KEY",
    ]}

def get_clients() -> Tuple[OpenAI, AsyncOpenAI, str, str]:
    """Create Azure OpenAI clients (v1 API) for the lab scripts."""
    env = load_environment()
    endpoint = env["AZURE_OPENAI_ENDPOINT"].strip().rstrip("/")
    api_key = env["AZURE_OPENAI_API_KEY"].strip()

    if not endpoint or not api_key:
        raise RuntimeError(
            "Azure credentials are missing. Please copy .env.example to .env "
            "and fill in your Azure OpenAI values."
        )

    deployment = env["AZURE_OPENAI_DEPLOYMENT"] or "gpt-4.1"

    # The v1 endpoint is OpenAI-compatible: the SDK appends /chat/completions and
    # the deployment name goes in the request body, so no api_version is needed.
    base_url = f"{endpoint}/openai/v1/" if not endpoint.endswith("/openai/v1") else f"{endpoint}/"
    sync_client = OpenAI(base_url=base_url, api_key=api_key)
    async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    return sync_client, async_client, deployment, ""

WEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

TRAVEL_INTENT = {"destination": "Rajasthan", "duration_days": 5, "travel_month": "February", "group_type": "family", "preferences": ["heritage sites", "local food markets"], "accommodation_preference": "heritage hotels or havelis", "budget_inr": 200000}

INSTRUCTIONS = "You are a travel planning agent. Always call get_weather first for the destination and travel month, then build a day-by-day itinerary that aligns outdoor activities with the good weather windows. Respond with valid JSON only, using keys: destination, duration_days, weather_summary, packing_essentials, itinerary, estimated_cost_inr, travel_tips."

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the near-term weather forecast for a destination. Returns a JSON string with average, min and max temperature plus the expected conditions so the itinerary can be aligned to good weather windows and packing advice.",
        "parameters": {
            "type": "object",
            "properties": {
                "destination": {"type": "string", "description": "City or region to look up."},
                "travel_month": {"type": "string", "description": "Month of travel, e.g. February."},
            },
            "required": ["destination", "travel_month"],
        },
    },
}

def get_weather(destination: str, travel_month: str) -> str:
    """Look up the near-term weather forecast for a destination. Returns a JSON string with average, min and max temperature plus the expected conditions so the itinerary can be aligned to good weather windows and packing advice."""
    try:
        geo = requests.get("https://api.openweathermap.org/geo/1.0/direct", params={"q": destination, "limit": 1, "appid": WEATHER_API_KEY}, timeout=10).json()
        if not geo:
            return json.dumps({"destination": destination, "error": "location not found"})
        forecast = requests.get("https://api.openweathermap.org/data/2.5/forecast", params={"lat": geo[0]["lat"], "lon": geo[0]["lon"], "cnt": 8, "appid": WEATHER_API_KEY, "units": "metric"}, timeout=10).json()
        temps = [entry["main"]["temp"] for entry in forecast.get("list", [])]
        conditions = sorted({entry["weather"][0]["description"] for entry in forecast.get("list", [])})
        return json.dumps({"destination": destination, "travel_month": travel_month, "avg_temp_celsius": round(sum(temps) / len(temps), 1) if temps else None, "min_temp_celsius": min(temps) if temps else None, "max_temp_celsius": max(temps) if temps else None, "conditions": conditions, "source": "OpenWeather 5-day / 3-hour forecast"})
    except Exception as exc:  # keep the tool resilient so the agent can still answer
        return json.dumps({"destination": destination, "error": str(exc)})

def run_agent(client, deployment: str, travel_intent: dict) -> dict:
    """Run the trip planner agent loop: let the model call get_weather, then return its JSON plan."""
    messages = [
        {"role": "system", "content": INSTRUCTIONS},
        {"role": "user", "content": json.dumps(travel_intent)},
    ]

    for _ in range(5):  # a few turns is plenty for one tool call plus the answer
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=[WEATHER_TOOL],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return json.loads(message.content)

        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {"name": call.function.name, "arguments": call.function.arguments}}
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"Tool call: get_weather({args})")
            result = get_weather(args.get("destination", ""), args.get("travel_month", ""))
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    raise RuntimeError("Agent did not produce a final answer within the turn limit.")

def pretty_print_json(data: Dict[str, Any]) -> None:
    """Print JSON in a readable indented format."""
    print(json.dumps(data, indent=2))


def run_lab() -> None:
    """Ask the agent to plan the demo trip and print the combined answer."""
    print("=== Lab 2: Trip Planner + Weather agent ===")

    client, _, deployment, _ = get_clients()
    print("Azure OpenAI client ready.\n")

    plan = run_agent(client, deployment, TRAVEL_INTENT)
    print("\nAgent response:")
    pretty_print_json(plan)


if __name__ == "__main__":
    run_lab()