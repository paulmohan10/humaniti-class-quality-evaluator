import os, shutil, subprocess, tempfile
from pathlib import Path
import gradio as gr
import numpy as np
import librosa
import cv2
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
    result = client().automatic_speech_recognition(audio=audio, model=ASR_MODEL)
    return str(result.get("text","") if isinstance(result,dict) else getattr(result,"text",result)).strip()

def inspect_frames(frames):
    import base64, io
    results=[]
    for ts, image in frames:
        b=io.BytesIO(); image.save(b,format="JPEG",quality=82)
        url="data:image/jpeg;base64,"+base64.b64encode(b.getvalue()).decode()
        response=client().chat_completion(model=VISION_MODEL,messages=[{"role":"user","content":[
          {"type":"text","text":"Describe only visible teaching materials: board, slides, text, diagrams, legibility and relevance. Do not identify people or infer age, identity, emotions, or engagement from a still image. If unsure, say so. Under 60 words."},
          {"type":"image_url","image_url":{"url":url}}]}],max_tokens=120,temperature=0.1)
        results.append({"timestamp_seconds":ts,"observation":str(response.choices[0].message.content).strip()})
    return results

def grade(transcript, features, visual):
    text=transcript.lower(); words=text.split(); wc=len(words)
    mins=max(features["duration_seconds"]/60,.1); wpm=wc/mins
    q=transcript.count("?")
    interaction_cues=["what do you think","can you tell me","your turn","let's try","any questions","why do you think","show me","can you explain"]
    explanation_cues=["because","for example","let's look","this means","step by step","first","next","remember","let me explain"]
    ih=sum(text.count(x) for x in interaction_cues); eh=sum(text.count(x) for x in explanation_cues)
    dims={"Instruction clarity":round(min(5,2.4+min(eh,5)*.35+(.3 if wc>80 else 0)),1),
          "Interaction signals":round(min(5,1.8+min(q,5)*.35+min(ih,4)*.35),1),
          "Pacing (rough speech-rate proxy)":round(2.6 if wc>30 and wpm<45 else 3.8 if 70<=wpm<=180 else 2.5 if wpm>230 else 3,1),
          "Audio continuity (rough proxy)":round(2.7 if features["quiet_frame_ratio"]>.55 else 3.5 if features["quiet_frame_ratio"]<.15 else 3.2,1)}
    weights={"Instruction clarity":.35,"Interaction signals":.30,"Pacing (rough speech-rate proxy)":.15,"Audio continuity (rough proxy)":.08}
    if visual:
        useful=sum(any(k in v["observation"].lower() for k in ["slide","board","diagram","text","equation"]) for v in visual)
        dims["Visual learning support (sampled frames)"]=round(min(5,2.5+.5*useful),1); weights["Visual learning support (sampled frames)"]=.12
    else:
        weights={k:v/.88 for k,v in weights.items()}
    score=sum(dims[k]*weights[k] for k in weights)
    label="Poor" if score<2.5 else "Average" if score<3.25 else "Good" if score<4.1 else "Excellent"
    evidence=[f"ASR returned {wc} words in {features['duration_seconds']:.0f} seconds (~{wpm:.0f} words/min).",
      f"Transcript contains {q} question marks; this is a weak interaction signal, not proof of learner participation.",
      f"Found {eh} predefined explanation/example cue phrases; these may miss equivalent wording.",
      "Audio energy is descriptive context only, not a direct measure of teaching quality."]
    if visual: evidence.append(f"Inspected {len(visual)} sampled video frames; these do not represent the entire lesson.")
    if not wc: evidence.append("Transcript is empty; content-based grading is unreliable.")
    return {"quality_label":label,"prototype_score_out_of_5":round(score,2),
      "confidence":"Low — prototype heuristics, not calibrated on labeled challenge examples",
      "dimension_scores_out_of_5":dims,"evidence":evidence,"audio_features":features,
      "visual_observations":visual,"model_configuration":{"asr_model":ASR_MODEL,"vision_model":VISION_MODEL if visual else None},
      "caveat":"Prototype only. Validate and calibrate against organizer-labeled recordings before relying on labels."}

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
        visual=inspect_frames(frames) if frames else []
        return grade(transcript,features,visual),transcript
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
