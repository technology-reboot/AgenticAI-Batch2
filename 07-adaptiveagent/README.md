# Session 05 – Adaptive Triage Feedback Loop

This lab implements a simple simulated RL-style feedback loop for a customer support triage agent. The agent classifies queries as either `AUTO` or `ESCALATE`, then adapts a routing threshold after observing batch results.

## Files

- `lab1_adaptive_feedback_loop.py`: Main training script.
- `data/support_queries.json`: Sample support queries with ground truth labels.
- `requirements.txt`: Python dependencies.
- `.env.example`: Example environment file for `OPENAI_API_KEY`.

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example` and add your OpenAI API key:

```powershell
copy .env.example .env
```

Then edit `.env` and set:

```text
OPENAI_API_KEY=your_api_key_here
```

## Run the code

From the `labs/session_05_rlhf_adaptive` folder, run:

```powershell
python lab1_adaptive_feedback_loop.py
```

If `OPENAI_API_KEY` is not set, the script will run with a simple heuristic classifier so you can still explore the feedback loop.

## What the script does

1. Loads labeled support queries.
2. Runs a baseline triage batch with a hardcoded threshold.
3. Introduces a drifted batch with ambiguous queries.
4. Evaluates metrics and updates the routing threshold.
5. Re-runs a final batch using the adapted threshold.
