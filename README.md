# Humaniti AI – Class Quality Evaluator

An AI-powered prototype for reviewing recorded classroom sessions using speech transcription, visual observations and evidence-based assessment.

## Live Application

[Try Humaniti AI – Class Quality Evaluator](https://huggingface.co/spaces/pmchittilappilly/humaniti-class-quality-evaluator)

## How to Use

1. Open the application using the link above.
2. Upload a classroom recording.
3. Confirm organiser permission and your authorisation to submit the recording.
4. Click **Assess class quality**.
5. Review the assessment report and transcript.

Only upload recordings when you have the necessary permissions. Extracted audio and sampled video frames are sent to the configured hosted inference provider.

## Assessment Criteria

| Dimension | Weight |
|---|---:|
| Introduction and rapport building | 10% |
| Instruction clarity and explanation | 20% |
| Student engagement and individual attention | 20% |
| Communication energy and pleasantness | 15% |
| Teaching materials and visual support | 10% |
| Lesson structure and conclusion | 10% |
| Homework and learning reinforcement | 15% |

## Outputs

- Overall classification: **Poor, Average, Good or Excellent**
- Weighted quality score and evidence coverage
- Individual dimension scores
- Supporting evidence and explanations
- Overall observations
- Visual observations from sampled frames
- Generated classroom transcript
- Basic audio measurements

If evidence coverage is below 70%, the result is classified as **Insufficient evidence**.

## Technology Stack

Python, Gradio, Hugging Face Spaces, Whisper Large V3, Qwen models, FFmpeg, OpenCV and Librosa.

## Current Status and Future Improvements

This is a working prototype, not yet a validated assessment system. Current results do not consistently match the expected human-labelled reference groups.

Further improvements require:
- More labelled recordings across all four quality categories.
- Overall human-assigned labels and dimension-level scores.
- Improved acoustic analysis and scoring consistency.
- Testing against independent recordings.
- Better reliability when processing longer videos.

Further rubric tuning and validation are essential.

## Feedback

If you encounter errors or issues, please share the error message and relevant assessment output.

---

**Project:** Humaniti AI – Class Quality Evaluator

**Live Demo:** https://huggingface.co/spaces/pmchittilappilly/humaniti-class-quality-evaluator

**GitHub Repository:** https://github.com/paulmohan10/humaniti-class-quality-evaluator
