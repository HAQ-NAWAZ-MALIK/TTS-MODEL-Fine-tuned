
# SpeechT5 Fine-Tuned English TTS Model Report Card

## Model Overview
- **Base Model**: Microsoft SpeechT5 (microsoft/speecht5_tts)
- **Fine-Tuned Model**: [Omarrran/speecht5_finetuned_emirhan_tr](https://huggingface.co/Omarrran/english_speecht5_finetuned/)
- **Task**: Text-to-Speech (TTS)
- **Language**: English
- **Dataset**: LJ Speech dataset
![image](https://github.com/user-attachments/assets/379e05e0-2add-4a79-9c90-19bf420d71cd)



![image](https://github.com/user-attachments/assets/57740df8-230c-474f-9a10-51ab4f780fb4)

## Training Details
- **Training Data**: LJ Speech dataset (train split)
- **Validation Data**: LJ Speech dataset (test split, 20% of total data)
- **Fine-tuning Steps**: 1500
- **Batch Size**: 4 (per device)
- **Gradient Accumulation Steps**: 8
- **Learning Rate**: 1e-4
- **Warm-up Steps**: 100
# Metrics Explanation
![image](https://github.com/user-attachments/assets/7fb696d6-c2ed-4c3e-b716-93c46a6faa45)

| Metric | Trend | Explanation |
|--------|-------|-------------|
| eval/loss | Decreasing | Measures the model's error on the evaluation dataset. Decreasing trend indicates improving model performance. |
| eval/runtime | Fluctuating, slightly decreasing | Time taken for evaluation. Minor fluctuations are normal, slight decrease may indicate optimization. |
| eval/samples_per_second | Increasing | Number of samples processed per second during evaluation. Increase suggests improved processing efficiency. |
| eval/steps_per_second | Increasing | Number of steps completed per second during evaluation. Increase indicates faster evaluation process. |
| train/epoch | Linearly increasing | Number of times the entire dataset has been processed. Linear increase is expected. |
| train/grad_norm | Decreasing with fluctuations | Magnitude of gradients. Decreasing trend with some fluctuations is normal, indicating stabilizing training. |
| train/learning_rate | sliglty inreasing | Rate at which the model updates its parameters. Decrease over time is typical in many learning rate schedules. |
| train/loss | Decreasing | Measures the model's error on the training dataset. Decreasing trend indicates the model is learning. |




## Key Differences and Improvements:
1. Dataset: the above model is fine-tuned on the LJSpeech dataset, which improves its performance on English TTS tasks.
2. Speaker Embeddings: incorporated speaker embeddings, which helps in maintaining speaker characteristics.
3. Text Preprocessing: This model includes advanced text preprocessing, including number-to-word conversion and technical term handling.
4. Training Optimizations: Used FP16 training and gradient checkpointing, which allows for more efficient training on GPUs.
5. Regular Evaluation: Training process includes regular evaluation, which helps in monitoring the model's performance during training.


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

model = SpeechT5ForTextToSpeech.from_pretrained("Omarrran/english_speecht5_finetuned")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Generate speech
# (See example in the notebook for full usage details)
```


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
