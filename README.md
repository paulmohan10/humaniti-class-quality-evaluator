# Humaniti AI – Class Quality Evaluator

## Project Overview

Humaniti AI is an AI-powered prototype designed to evaluate classroom teaching quality using speech transcription, visual observations and evidence-based assessment.

The system analyses recorded classroom sessions and generates an assessment across seven teaching-quality dimensions.

## 1. How It Works

The application follows three main stages:

1. **Audio Processing:** Extracts classroom audio and uses Whisper Large V3 to generate speech transcripts.
2. **Visual Analysis:** Uses Qwen vision models to examine sampled video frames and identify visible teaching materials.
3. **AI Evaluation:** Uses Qwen language models to assess teaching evidence against predefined scoring criteria.

The system is designed to support English and Malayalam bilingual classroom recordings.

## 2. Assessment Dimensions

| Dimension | Weight |
|---|---:|
| Introduction and rapport building | 10% |
| Instruction clarity and explanation | 20% |
| Student engagement and individual attention | 20% |
| Communication energy and pleasantness | 15% |
| Teaching materials and visual support | 10% |
| Lesson structure and conclusion | 10% |
| Homework and learning reinforcement | 15% |
| **Total** | **100%** |

## 3. Generated Outputs

The evaluator provides:

- Overall quality classification: Poor, Average, Good or Excellent.
- Weighted assessment score.
- Evidence coverage.
- Individual dimension-level scores.
- Supporting evidence and explanations.
- Overall teaching observations.
- Visual observations from sampled frames.
- Generated classroom transcript.
- Basic audio measurements.

When evidence coverage is insufficient, the system can return an Insufficient Evidence classification.

## 4. Technology Stack

- Python
- Gradio
- Hugging Face Spaces and InferenceClient
- Whisper Large V3
- Qwen Vision and Language Models
- FFmpeg
- OpenCV
- Librosa
- NumPy and Pillow

## 5. Current Status and Limitations

This project is currently a working prototype and has not yet achieved consistent agreement with the human-labelled reference categories.

Current limitations include:
- Limited labelled reference recordings.
- Need for better differentiation between quality categories.
- Difficulty evaluating student engagement from speech alone.
- Limitations in assessing communication energy and pleasantness without dedicated acoustic analysis.
- Potential loss of context during long-transcript summarisation.
- Occasional hosted inference timeouts.

## 6. Future Improvements

Planned improvements include:

- Expanding the labelled dataset across Poor, Average, Good and Excellent categories.
- Collecting both overall human-assigned labels and dimension-level scores.
- Improving scoring anchors and evaluation prompts.
- Developing more reliable acoustic analysis.
- Improving long-recording processing.
- Validating results against independent, unseen recordings.

A larger and balanced labelled dataset will be essential for improving classification consistency and reliability.

## Project Information

**Project:** Humaniti AI – Class Quality Evaluator  
**Repository:** Humaniti Class Quality Evaluator

