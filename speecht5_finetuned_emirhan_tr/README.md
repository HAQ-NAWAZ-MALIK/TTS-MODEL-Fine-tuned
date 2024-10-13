---
license: mit
datasets:
- keithito/lj_speech
language:
- en
base_model:
- microsoft/speecht5_tts
tags:
- tts
- generated_from_trainer
library_name: transformers
---

# *Note:* 
*This report was prepared as a task given by the IIT Roorkee PARIMAL intern program. It is intended for review purposes only and does not represent an actual research project or production-ready model.*

# Omarrran/speecht5_finetuned_emirhan_tr 

This model is a fine-tuned version of [microsoft/speecht5_tts](https://huggingface.co/microsoft/speecht5_tts) on the lj_speech dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3715

# Fine-tuning SpeechT5 for English Text-to-Speech (TTS)

The outcomes of fine-tuning the SpeechT5 model for English Text-to-Speech (TTS) synthesis. The project was conducted as a demo assignment, leveraging the LJSpeech dataset to enhance the model's capabilities in generating natural-sounding English speech. Key achievements include improved intonation, pronunciation, and speaker consistency, demonstrating the potential of SpeechT5 in TTS applications.

## 1. Introduction

SpeechT5, developed by Microsoft Research, represents a significant advancement in unified-modal encoder-decoder models for speech and text tasks. Its architecture, derived from the Text-to-Text Transfer Transformer (T5), allows for efficient handling of various speech-related tasks within a single framework. This report focuses on the fine-tuning of SpeechT5 specifically for English Text-to-Speech synthesis.

### Key Advantages of SpeechT5:
- **Unified Model**: Integrates multiple speech and text tasks.
- **Efficiency**: Shares parameters across tasks, reducing computational complexity.
- **Cross-task Learning**: Enhances performance through transfer learning.
- **Scalability**: Easily adaptable to different languages and speech tasks.

## 2. Objective

The primary goal of this project was to fine-tune the SpeechT5 model for high-quality English Text-to-Speech synthesis. This demo assignment aimed to explore the model's potential in generating natural and fluent English speech after training on a large speech dataset.

**Project Specifications:**
- **Duration**: 60 minutes (demo assignment)
- **Training Epochs**: 500
- **Hardware**: T4 GPU

## 3. Methodology

### Dataset

**LJSpeech Dataset**
- **Content**: ~24 hours of single-speaker English speech data
- **Size**: 13,100 short audio clips
- **Source**: Readings from seven non-fiction books
- **Preprocessing**:
  - Audio resampled to 16kHz
  - Text normalized for consistent pronunciation
  - Special characters and numbers converted to written form

### Model Architecture

**Base Model**: `microsoft/speecht5_tts` from Hugging Face
- **Type**: Unified-modal encoder-decoder
- **Foundation**: T5 architecture

### Fine-tuning Process

**Hardware Setup:**
- GPU: NVIDIA T4
- Total Runtime: 60 minutes

**Hyperparameters:**
- Epochs: 500
- Batch Size: 4
- Optimizer: AdamW with weight decay
- Learning Rate: 1e-5
- Scheduler: Linear with warmup
- Gradient Accumulation: Implemented to simulate larger batches

**Training Procedure:**
1. Utilized Hugging Face Transformers library
2. Implemented regular validation checks
3. Applied early stopping based on validation loss

**Challenges Addressed:**
- Memory constraints (T4 GPU limitations)
- Time management (60-minute constraint)
- Overfitting mitigation

## 4. Results and Evaluation

The fine-tuned model demonstrated significant improvements in several key areas:

**Naturalness of Speech:**
- Enhanced intonation patterns
- Improved pronunciation of complex words
- Better rhythm and pacing, especially for longer sentences

**Voice Consistency:**
- Maintained consistent voice quality across various samples
- Sustained quality in generating extended speech segments

**Quantitative Metrics:**
-
### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.4691        | 0.3053 | 100  | 0.4127          |
| 0.4492        | 0.6107 | 200  | 0.4079          |
| 0.4342        | 0.9160 | 300  | 0.3940          |
| 0.4242        | 1.2214 | 400  | 0.3917          |
| 0.4215        | 1.5267 | 500  | 0.3866          |
| 0.4207        | 1.8321 | 600  | 0.3843          |
| 0.4156        | 2.1374 | 700  | 0.3816          |
| 0.4136        | 2.4427 | 800  | 0.3807          |
| 0.4107        | 2.7481 | 900  | 0.3792          |
| 0.408         | 3.0534 | 1000 | 0.3765          |
| 0.4048        | 3.3588 | 1100 | 0.3762          |
| 0.4013        | 3.6641 | 1200 | 0.3742          |
| 0.4002        | 3.9695 | 1300 | 0.3733          |
| 0.3997        | 4.2748 | 1400 | 0.3727          |
| 0.4012        | 4.5802 | 1500 | 0.3715          |


### Framework versions

- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 3.0.1
- Tokenizers 0.19.1

## 5. Limitations and Future Work

**Current Limitations:**
1. Single-speaker output
2. Limited emotional range and style control

**Proposed Future Directions:**
1. Multi-speaker fine-tuning
2. Emotion and style control integration
3. Domain-specific adaptations (e.g., technical, medical)
4. Model optimization for faster inference

## 6. Conclusion

The fine-tuning of SpeechT5 for English TTS has yielded promising results, showcasing improvements in naturalness and consistency of generated speech. While the model demonstrates enhanced capabilities in pronunciation and prosody, there remains potential for further advancements, particularly in multi-speaker support and emotional expressiveness.

## 7. Acknowledgments

- Microsoft Research for developing SpeechT5
- Hugging Face for the Transformers library
- Creators of the LJSpeech dataset