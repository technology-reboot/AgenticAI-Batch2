# Session 02 Prompt Engineering Lab

This folder contains a simple training-friendly version of the Session 2 lab exercises.
The examples are written as plain Python scripts so they are easy to read and run.

## Project structure

- `lab1_prompt_patterns.py` — compare zero-shot, few-shot, and CoT prompts
- `lab2_parameter_tuning.py` — explore temperature, JSON output stability, and stop sequences
- `lab3_multimodal_pipeline.py` — combine audio transcription, text extraction, and chart analysis
- `setup/generate_lab_data.py` — create sample audio and chart files used by the labs
- `data/` — sample reviews, generated audio, and chart image

## Setup

1. Open a terminal in this folder.
2. Create and activate a virtual environment if you want.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example environment file and add your OpenAI key if you want to use the real API:

```bash
copy .env.example .env
```

Then edit `.env` and replace `your_key_here` with your real key.

## Generate the lab data files

```bash
python setup/generate_lab_data.py
```

This creates:
- `data/meeting_audio.wav` — real speech synthesized with the OpenAI TTS API when a key is configured, otherwise a synthetic tone
- `data/bar_chart.png`

## Run the labs

### Lab 1: Prompt patterns

```bash
python lab1_prompt_patterns.py
```

### Lab 2: Parameter tuning

```bash
python lab2_parameter_tuning.py
```

### Lab 3: Multimodal pipeline

```bash
python lab3_multimodal_pipeline.py
```

## Notes

- All three labs call the real OpenAI API (chat completions, Whisper transcription, TTS, and vision) when `OPENAI_API_KEY` is set in `.env`.
- If no OpenAI key is available, the scripts fall back to simple local examples so the training flow still works.
- The code is intentionally simple and commented for classroom use.
