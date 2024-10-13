
# SpeechT5 Fine-Tuned TTS Model Report Card

## Model Overview
- **Base Model**: Microsoft SpeechT5 (microsoft/speecht5_tts)
- **Fine-Tuned Model**: [Omarrran/speecht5_finetuned_emirhan_tr](https://huggingface.co/Omarrran/speecht5_finetuned_emirhan_tr)
- **Task**: Text-to-Speech (TTS)
- **Language**: English
- **Dataset**: LJ Speech dataset

## Training Details
- **Training Data**: LJ Speech dataset (train split)
- **Validation Data**: LJ Speech dataset (test split, 20% of total data)
- **Fine-tuning Steps**: 1500
- **Batch Size**: 4 (per device)
- **Gradient Accumulation Steps**: 8
- **Learning Rate**: 1e-4
- **Warm-up Steps**: 100

## Model Performance
- **Evaluation Strategy**: Steps
- **Evaluation Frequency**: Every 100 steps
- **Metric**: Not specified (uses `greater_is_better=False`)

## Model Capabilities
- Generates speech from input text
- Supports speaker embeddings for voice customization
- Handles technical terms and abbreviations (e.g., API, CUDA, GPU)
- Converts numbers to word form for natural speech

## Limitations
- Limited to English language
- Voice quality may vary depending on input text and speaker embeddings
- Performance on out-of-domain text or accents not evaluated

## Ethical Considerations
- Potential for misuse in creating deepfake audio
- Bias in voice generation based on training data demographics

## Usage
The model can be used with the Hugging Face Transformers library:

```python
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan

model = SpeechT5ForTextToSpeech.from_pretrained("Omarrran/speecht5_finetuned_emirhan_tr")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate speech
# (See example in the notebook for full usage details)
```

## Demo
A live demo of the model is available on Hugging Face Spaces:
[TTS Model Demo](https://huggingface.co/spaces/Omarrran/tts_model_demo)

## Citation
If you use this model, please cite:
```
@misc{speecht5_finetuned_emirhan_tr,
  author = {HAQ NAWAZ MALIK},
  title = {Fine-tuned SpeechT5 for Text-to-Speech},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://huggingface.co/Omarrran/speecht5_finetuned_emirhan_tr}},
}
```

## Acknowledgements
- Base SpeechT5 model by Microsoft
- LJ Speech dataset
- PARIMAL intern program at IIT Roorkee
