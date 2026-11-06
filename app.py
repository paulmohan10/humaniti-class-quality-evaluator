import os, shutil, subprocess, tempfile
from pathlib import Path
import gradio as gr
import numpy as np
import librosa
import cv2
#import spaces
from PIL import Image
from huggingface_hub import InferenceClient

ASR_MODEL = os.getenv("HF_ASR_MODEL", "openai/whisper-large-v3")
VISION_MODEL = os.getenv("HF_VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
HOSTED_ENABLED = os.getenv("ALLOW_HOSTED_INFERENCE", "false").lower() == "true"

def ffmpeg(args):
    p = subprocess.run(args, capture_output=True, text=True, timeout=180)
    if p.returncode:
        raise RuntimeError((p.stderr or "ffmpeg failed")[-1200:])

def prepare(path):
    if Path(path).stat().st_size > int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024:
        raise ValueError("File exceeds upload limit.")
    work = tempfile.mkdtemp(prefix="humaniti_")
    audio = str(Path(work) / "audio.wav")
    ffmpeg(["ffmpeg","-y","-i",path,"-vn","-ac","1","-ar","16000","-t","3600",audio])
    frames = []
    cap = cv2.VideoCapture(path)
    count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    duration = count / fps if count > 0 and fps > 0 else 0
    n = int(os.getenv("MAX_VIDEO_FRAMES", "6"))
    if cap.isOpened() and duration:
        for i in range(n):
            cap.set(cv2.CAP_PROP_POS_MSEC, duration * (i + .5) / n * 1000)
            ok, frame = cap.read()
            if ok:
                h,w = frame.shape[:2]
                scale = min(1, 768/max(h,w))
                if scale < 1: frame = cv2.resize(frame,(int(w*scale),int(h*scale)))
                frames.append((round(duration*(i+.5)/n,1), Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
    cap.release()
    return audio, frames, work

def client():
    token = os.getenv("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN missing; add it as a Space Secret.")
    return InferenceClient(token=token, timeout=120)

def transcribe(audio):
    import soundfile as sf
    import numpy as np
    import tempfile
    import os

    chunk_seconds = 300
    sample_rate = 16000

    audio_data, sr = sf.read(audio)

    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    chunk_samples = chunk_seconds * sr
    transcripts = []

    for start in range(0, len(audio_data), chunk_samples):
        chunk = audio_data[start:start + chunk_samples]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            chunk_path = f.name

        try:
            sf.write(chunk_path, chunk, sr)

            result = client().automatic_speech_recognition(
                audio=chunk_path,
                model=ASR_MODEL
            )

            text = (
                result.get("text", "")
                if isinstance(result, dict)
                else getattr(result, "text", result)
            )

            transcripts.append(str(text).strip())

        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return " ".join(transcripts)

def inspect_frames(frames):
    import base64, io
    results=[]
    vision_client = InferenceClient(
        token=os.getenv("HF_TOKEN"),
        timeout=120,
        provider="ovhcloud"
    )
    for ts, image in frames:
        b=io.BytesIO(); image.save(b,format="JPEG",quality=82)
        url="data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
        response=vision_client.chat_completion(model=VISION_MODEL,messages=[{"role":"user","content":[
          {"type":"text","text":"Describe only visible teaching materials: board, slides, text, diagrams, legibility and relevance. Do not identify people or infer age, identity, emotions, or engagement from a still image. If unsure, say so. Under 60 words."},
          {"type":"image_url","image_url":{"url":url}}]}],max_tokens=120,temperature=0.1)
        results.append({"timestamp_seconds":ts,"observation":str(response.choices[0].message.content).strip()})
    return results




def split_transcript(transcript, max_chars=10000):
    words = transcript.split()
    chunks = []
    current_chunk = []

    current_length = 0

    for word in words:
        word_length = len(word) + 1

        if current_length + word_length > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def analyze_teaching_evidence(transcript, visual):
    import json
    import time

    weights = {
        "Introduction and rapport building": 10,
        "Instruction clarity and explanation": 20,
        "Student engagement and individual attention": 20,
        "Communication energy and pleasantness": 15,
        "Teaching materials and visual support": 10,
        "Lesson structure and conclusion": 10,
        "Homework and learning reinforcement": 15
    }

    anchors = """
Use these scoring anchors:
1. Introduction and rapport: 1 = absent; 3 = basic; 5 = strong, inclusive rapport.
2. Instruction clarity: 1 = difficult to follow; 3 = understandable with gaps; 5 = consistently clear with examples and clarification.
3. Student engagement: 1 = little interaction; 3 = some interaction; 5 = regular, inclusive interaction and individual attention.
4. Communication energy and pleasantness: 1 = observable communication problems; 3 = adequate; 5 = consistently pleasant, energetic and supportive. Do not infer pleasantness from transcript text alone.
5. Materials: 1 = absent or unsuitable; 3 = basic relevant materials; 5 = clearly relevant and effectively integrated.
6. Structure: 1 = incoherent; 3 = basic structure; 5 = well-organised with effective transitions and conclusion.
7. Homework: 1 = no observable reinforcement; 3 = some reinforcement; 5 = specific homework and effective reinforcement.
"""

    visual_text = json.dumps(visual, ensure_ascii=False)

    def call_qwen(prompt, max_tokens=1200):
        last_error = None

        for attempt in range(3):
            try:
                response = client().chat_completion(
                    model=os.getenv(
                        "HF_EVALUATION_MODEL",
                        "Qwen/Qwen2.5-VL-72B-Instruct"
                    ),
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    max_tokens=max_tokens,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()

            except Exception as exc:
                last_error = exc
                if attempt < 2:
                    time.sleep(3 * (attempt + 1))

        raise RuntimeError(f"Qwen request failed after retries: {last_error}")

    chunks = split_transcript(transcript, max_chars=8000)

    # Short recordings use one scoring request.
    if len(chunks) == 1:
        evidence_text = transcript

    # Long recordings: summarize evidence from each section first.
    else:
        summaries = []

        for index, chunk in enumerate(chunks, start=1):
            prompt = f"""
Extract concise, factual teaching evidence from transcript section
{index} of {len(chunks)}.

Return only JSON with these keys:
- introduction_and_rapport
- instruction_clarity
- student_interaction
- communication_observations
- lesson_structure
- homework_and_reinforcement
- other_relevant_evidence

Use short bullet-like statements in each value.
If evidence is absent, use an empty string.
Do not score the teacher.
Do not invent events or infer student engagement without evidence.
The transcript may contain Malayalam and English. Preserve meaning.

TRANSCRIPT SECTION:
{chunk}
"""
            summaries.append(call_qwen(prompt, max_tokens=600))

        evidence_text = "\n".join(
            f"SECTION {i} EVIDENCE:\n{summary}"
            for i, summary in enumerate(summaries, start=1)
        )

    scoring_prompt = f"""
Evaluate the teaching session using only the supplied evidence.

Return valid JSON only, with this structure:
{{
  "dimensions": [
    {{
      "name": "Introduction and rapport building",
      "score": 1,
      "evidence": "Specific supporting evidence",
      "reason": "Brief explanation"
    }}
  ],
  "overall_observations": ["observation"]
}}

Include all seven dimensions, using these exact names:
{json.dumps(list(weights.keys()))}

Score each dimension from 1 to 5.
Use null when there is insufficient evidence.
Do not invent evidence.
Do not assume questions automatically prove student engagement.
Do not infer emotions from still images.
Do not penalize Malayalam or bilingual teaching.
Distinguish teacher speech from embedded video narration.
Do not infer homework from future puzzles or revision unless it is explicitly assigned.

For communication energy and pleasantness, use only genuine audio-derived
observations if supplied. Do not infer vocal tone or pleasantness from text.

{anchors}

RUBRIC WEIGHTS:
{json.dumps(weights)}

VISUAL OBSERVATIONS:
{visual_text}

TEACHING EVIDENCE:
{evidence_text}
"""

    result = call_qwen(scoring_prompt, max_tokens=1800)

    # Extract JSON if the model wraps it in Markdown fences.
    result = result.replace("```json", "").replace("```", "").strip()

    start = result.find("{")
    end = result.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Qwen did not return valid JSON.")


    parsed = json.loads(result[start:end + 1])
    if isinstance(parsed, list):
        parsed = {"dimensions": parsed}
    if isinstance(parsed.get("dimensions"), list):
        parsed["dimensions"] = {
        item["name"]: item
        for item in parsed["dimensions"]
        if isinstance(item, dict) and "name" in item
        }
    return parsed


def grade(transcript, features, visual, analysis):
    rubric_weights = {
        "Introduction and rapport building": 10,
        "Instruction clarity and explanation": 20,
        "Student engagement and individual attention": 20,
        "Communication energy and pleasantness": 15,
        "Teaching materials and visual support": 10,
        "Lesson structure and conclusion": 10,
        "Homework and learning reinforcement": 15
    }

    dimensions = analysis.get("dimensions", {})
    dimension_scores = {}
    evidence = []
    weighted_total = 0.0
    available_weight = 0

    for name, weight in rubric_weights.items():
        item = dimensions.get(name, {})
        score = item.get("score")

        if isinstance(score, (int, float)) and not isinstance(score, bool) and 1 <= score <= 5:
            dimension_scores[name] = round(float(score), 1)
            weighted_total += float(score) * weight
            available_weight += weight
        else:
            dimension_scores[name] = None

        detail = item.get("evidence", "No reliable evidence supplied.")
        limitations = item.get("limitations", "Not specified.")

        evidence.append({
            "dimension": name,
            "score": dimension_scores[name],
            "weight_percent": weight,
            "evidence": detail,
            "limitations": limitations
        })

    coverage = available_weight / 100

    if available_weight == 0:
        score = None
        label = "Insufficient evidence"
    else:
        score = round(weighted_total / available_weight, 2)

        if coverage < 0.70:
            label = "Insufficient evidence"
        elif score < 2.5:
            label = "Poor"
        elif score < 3.25:
            label = "Average"
        elif score < 4.1:
            label = "Good"
        else:
            label = "Excellent"

    return {
        "quality_label": label,
        "prototype_score_out_of_5": score,
        "confidence": {
            "evidence_coverage_percent": round(coverage * 100),
            "note": "Coverage measures available rubric evidence, not statistical confidence."
        },
        "dimension_scores_out_of_5": dimension_scores,
        "evidence": evidence,
        "audio_features": features,
        "visual_observations": visual,
        "model_configuration": {
            "asr_model": ASR_MODEL,
            "vision_model": VISION_MODEL if visual else None
        },
        "caveat": "AI-assisted prototype. Validate against independent human-labelled recordings before operational use."
    }

#@spaces.GPU

def evaluate(upload, permitted, authorized):
    if not upload: raise gr.Error("Upload a recording first.")
    if not HOSTED_ENABLED: raise gr.Error("Hosted inference is disabled. Obtain organizer permission, then set ALLOW_HOSTED_INFERENCE=true.")
    if not (permitted and authorized): raise gr.Error("Confirm organizer permission and authorization before processing.")
    work=None
    try:
        audio,frames,work=prepare(upload)
        y,sr=librosa.load(audio,sr=16000,mono=True,duration=3600)
        if not len(y): raise ValueError("No audio extracted.")
        rms=librosa.feature.rms(y=y)[0]
        quiet=max(float(np.percentile(rms,20))*.6,.003)
        features={"duration_seconds":round(len(y)/sr,1),"rms_mean":round(float(np.mean(rms)),4),
                  "quiet_frame_ratio":round(float(np.mean(rms<quiet)),3),
                  "note":"Descriptive acoustic statistics; not direct teaching-quality measures."}
        transcript=transcribe(audio)

        visual = inspect_frames(frames) if frames else []
        #extracted_evidence = extract_teaching_evidence(transcript)
        #analysis = analyze_teaching_evidence(extracted_evidence, visual)

        analysis = analyze_teaching_evidence(transcript, visual)
        #return grade(transcript, features, visual, analysis), transcript

        #report = grade(transcript, features, visual, analysis)
        #report["extracted_evidence"] = extracted_evidence
        #return report, transcript

        return grade(transcript, features, visual, analysis), transcript




        
    except Exception as e:
        raise gr.Error(f"Assessment failed: {type(e).__name__}: {e}")
    finally:
        if work: shutil.rmtree(work,ignore_errors=True)

with gr.Blocks(title="Humaniti AI Class Quality Evaluator",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Humaniti AI · Class Quality Evaluator\nUpload a class recording for a provisional evidence-backed assessment. **Hosted inference sends extracted audio and sampled frames to the configured provider. Do not proceed without organizer permission.**")
    upload=gr.File(label="Class recording",file_types=[".mp4",".mov",".mkv",".webm",".mp3",".wav",".m4a",".ogg"],type="filepath")
    permitted=gr.Checkbox(label="The organizer explicitly permits hosted inference for this recording.",value=False)
    authorized=gr.Checkbox(label="I am authorized to submit this recording and understand audio/frames are sent to the model provider.",value=False)
    run=gr.Button("Assess class quality",variant="primary")
    report=gr.JSON(label="Assessment report")
    transcript=gr.Textbox(label="Recognized transcript",lines=8)
    run.click(evaluate,[upload,permitted,authorized],[report,transcript])
    gr.Markdown("### Limitations\nThe rubric is heuristic and unvalidated. Accent, identity, appearance, loudness, and inferred emotions are not quality criteria. Video analysis uses sampled frames only.")
if __name__=="__main__":
    demo.queue(max_size=4).launch(server_name="0.0.0.0",server_port=int(os.getenv("PORT","7860")))
