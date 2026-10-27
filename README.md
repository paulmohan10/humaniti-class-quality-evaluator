---
title: Humaniti AI Class Quality Evaluator
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Humaniti AI — Class Quality Evaluator

A Gradio prototype for the Humaniti AI / 8x8 recruitment challenge. It accepts an audio/video class recording, transcribes speech with Hugging Face hosted inference, optionally inspects sampled video frames with a vision-language model, and produces a provisional Poor / Average / Good / Excellent label with evidence and dimension scores.

## Important: privacy and limitations

The challenge brief says recordings must remain confidential and should only be processed using services specifically permitted by the organizer, especially when children are identifiable.

**Hosted inference is disabled by default.** The app will not assess recordings unless:
1. The organizer explicitly permits the configured hosted inference service for these recordings.
2. The deployer sets `ALLOW_HOSTED_INFERENCE=true`.
3. The user confirms the permission and authorization checkboxes in the UI.

When enabled, extracted audio is sent to the configured Hugging Face inference provider for ASR, and sampled frames are sent for vision analysis. Check the applicable provider terms and organizer policy before enabling this. Do not use a public Space for confidential recordings. Prefer a private Space and restrict access; a public Space exposes the app to anyone.

The challenge PDF provides a task statement but no labeled examples, gold scores, or formal rubric. Therefore the included scoring is a **baseline heuristic**, not a validated measure of teaching quality. Do not present confidence as calibrated. Evaluate and tune against organizer-provided labels before relying on the grades.

## Model choices

- ASR default: `openai/whisper-large-v3`. Hugging Face's Automatic Speech Recognition documentation lists it as a recommended ASR model and documents supported providers. See https://huggingface.co/docs/inference-providers/tasks/automatic-speech-recognition
- Vision default: `Qwen/Qwen2.5-VL-7B-Instruct`, an Apache-2.0 vision-language model with published video/image benchmark results. Availability varies by inference provider. See https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- The code exposes `HF_ASR_MODEL` and `HF_VISION_MODEL` environment variables so you can change the model after checking current provider availability. If a model is not available to your token/provider, inference will fail visibly rather than silently inventing results.

The grade uses transcript cues and simple acoustic statistics. It does not identify speakers or infer student emotions. It treats image observations as limited evidence about learning materials only. The baseline score should be improved using a rubric reviewed by educators and validated on labeled recordings.

## Local setup

Requires Python 3.10+ and `ffmpeg`.

### 1. Install ffmpeg

- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: install ffmpeg and ensure `ffmpeg` is on PATH.

### 2. Create environment and install dependencies

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Hugging Face access

Create a Hugging Face access token with the minimum required permissions at https://huggingface.co/settings/tokens. Keep it secret.

```bash
# macOS/Linux
export HF_TOKEN="hf_..."
export ALLOW_HOSTED_INFERENCE="true"
# Windows PowerShell
$env:HF_TOKEN="hf_..."
$env:ALLOW_HOSTED_INFERENCE="true"
```

Only set `ALLOW_HOSTED_INFERENCE=true` after obtaining organizer permission and checking the inference provider's data terms. If permission is not available, leave it false; this prototype deliberately blocks processing.

Optional configuration:

```bash
export HF_ASR_MODEL="openai/whisper-large-v3"
export HF_VISION_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export MAX_UPLOAD_MB="500"
export MAX_VIDEO_FRAMES="6"
```

### 4. Run

```bash
python app.py
```

Open http://localhost:7860. Upload a recording, confirm both permission checkboxes, and click **Assess class quality**.

## Deploy to Hugging Face Spaces

1. Create a **private** Gradio Space at https://huggingface.co/new-space. Keep the Space private because the input may contain children or confidential class content. Verify access controls and any plan requirements.
2. Push `app.py`, `requirements.txt`, and `README.md` to the Space repository. Add `ffmpeg` to the runtime; for a Gradio Space that does not have it installed, use a Docker Space with the following `Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]
```

For the Docker option, create a Docker Space and commit `Dockerfile` as well.
3. In Space Settings → Variables and secrets, add `HF_TOKEN` as a **Secret**. Add `ALLOW_HOSTED_INFERENCE=true` as a **Variable only after organizer permission is confirmed**. Never commit the token to Git.
4. Choose compute suitable for the application and provider quotas. Large video files and repeated frame analysis may take time.
5. Test with a non-sensitive sample first. Confirm upload, audio extraction, transcription, frame analysis, report rendering, and failure messages.
6. Share the app only with authorized evaluators. Do not upload challenge recordings to a public dataset or include them in the repository, logs, screenshots, or README.

Official Spaces docs: https://huggingface.co/docs/hub/en/spaces-overview

## What the report contains

- `quality_label`: Poor / Average / Good / Excellent.
- `prototype_score_out_of_5`: heuristic score, not a calibrated probability.
- `dimension_scores_out_of_5`: clarity, interaction signals, rough pacing, audio continuity, and visual support if video is present.
- `evidence`: specific transcript/audio/frame observations and limitations.
- `visual_observations`: timestamped notes on learning materials in sampled frames.
- `confidence`: explicitly warns that the baseline is not calibrated.

## Next improvements before submission

1. Have educators define observable rubric anchors for each of the four labels.
2. Score a held-out, labeled subset; report confusion matrix, macro-F1, and per-class recall.
3. Replace keyword-count heuristics with structured model assessment grounded in transcript snippets and timestamps.
4. Add speech timestamps/segments and connect every explanation to a timestamp.
5. Add file cleanup and retention controls, access restrictions, request limits, and privacy review.
6. Check every model/provider's current availability, latency, price, license, and data handling before deployment.

## Repository contents

- `app.py` — Gradio application and assessment pipeline
- `requirements.txt` — Python dependencies
- `README.md` — setup, privacy, model choices, deployment
- `Dockerfile` — optional Docker Space runtime with ffmpeg
