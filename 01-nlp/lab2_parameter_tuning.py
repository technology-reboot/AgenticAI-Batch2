import json
import time
from typing import Any, Optional

from common import get_client, maybe_sleep

MODEL = "gpt-4o-mini"


def generate(
    prompt: str,
    temperature: float,
    max_tokens: int = 120,
    stop: Optional[list[str]] = None,
    client: Any = None,
) -> dict[str, Any]:
    """
    Call the OpenAI API with the given sampling parameters, or use a local
    fallback when no key is configured.
    """
    if client is None:
        text = f"[local fallback] Temperature={temperature} | Prompt={prompt[:40]}..."
        return {"text": text, "latency_ms": 12.0, "temperature": temperature}

    started = time.perf_counter()
    kwargs: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        kwargs["stop"] = stop

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""

    return {
        "text": text,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "temperature": temperature,
    }


def strip_code_fence(text: str) -> str:
    """Strip a ```json ... ``` (or plain ``` ... ```) fence some models add
    around JSON output even when asked not to."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def run_lab2() -> None:
    """Explore how temperature affects response diversity and structure."""
    print("=== Lab 2: Parameter tuning explorer ===")
    client = get_client()
    mode = "OpenAI API" if client is not None else "local fallback"
    print(f"Using {mode}.\n")

    creative_prompt = "Write a one-sentence product description for a pair of wireless earbuds."
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    n_runs = 3

    print("=== Creative task: temperature spectrum ===")
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        responses = []
        for _ in range(n_runs):
            result = generate(creative_prompt, temperature=temp, max_tokens=60, client=client)
            responses.append(result["text"].strip())
            maybe_sleep(0.3)

        for i, response in enumerate(responses, 1):
            print(f"  Run {i}: {response}")

        diversity = len(set(responses)) / len(responses)
        print(f"  Diversity: {diversity:.0%} ({len(set(responses))} unique / {n_runs} runs)")

    print("\n=== Structured extraction: why temperature 0 matters ===")
    extraction_prompt = (
        "Extract the following fields from this invoice text as JSON with keys "
        "vendor_name, invoice_date, total_amount, line_items:\n\n"
        "Invoice from Acme Supplies, dated 2026-01-15, total $482.50. "
        "Items: 10x USB cables, 2x HDMI adapters.\n\n"
        "Return only the JSON object, with no other text."
    )
    parse_failures = {0.0: 0, 0.7: 0, 1.2: 0}

    for temp in [0.0, 0.7, 1.2]:
        print(f"\n--- Temperature = {temp} (3 runs) ---")
        for run in range(3):
            result = generate(extraction_prompt, temperature=temp, max_tokens=200, client=client)
            raw = strip_code_fence(result["text"])
            try:
                parsed = json.loads(raw)
                print(f"  Run {run + 1}: [OK] Valid JSON - vendor: {parsed.get('vendor_name', '?')}")
            except json.JSONDecodeError:
                parse_failures[temp] += 1
                print(f"  Run {run + 1}: [FAIL] JSON parse FAILED - output: {raw[:80]}...")
            maybe_sleep(0.3)

    print("\n=== Parse failure rate ===")
    for temp, failures in parse_failures.items():
        print(f"  Temperature {temp}: {failures}/3 failures ({failures / 3:.0%} failure rate)")

    print("\n=== Stop sequence demo ===")
    multi_section_prompt = (
        "Write a brief product review for wireless earbuds using exactly this format:\n"
        "Pros:\n<your text>\nCons:\n<your text>\nVerdict:\n<your text>"
    )
    result_full = generate(multi_section_prompt, temperature=0.7, max_tokens=200, client=client)
    print("Without stop sequence:")
    print(result_full["text"])

    print("\nWith stop=['Cons:']:")
    result_stopped = generate(
        multi_section_prompt, temperature=0.7, max_tokens=200, stop=["Cons:"], client=client
    )
    print(result_stopped["text"])

    print("\n--- Key conclusions ---")
    print("- Temperature 0.0 is best for deterministic classification and JSON extraction.")
    print("- Higher temperatures create more variety, but can hurt structure and reliability.")
    print("- Stop sequences help restrict output length and keep responses focused.")


if __name__ == "__main__":
    run_lab2()
