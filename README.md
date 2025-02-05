#  Fine-Tuning SpeechT5 for English Speech-to-Text with a Focus on Technical Terminology

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/Aumkeshchy2003/English-tts)

Transform technical speech into accurate text with our fine-tuned SpeechT5 model. This repository provides a comprehensive guide for training a Speech-to-Text model using Hugging Face's transformers library, combined with audio datasets and SpeechBrain for speaker embeddings.

##  Features

- Custom text normalization pipeline
- Speaker embedding generation using SpeechBrain
- Vocabulary extraction and management
- Optimized for technical terminology
- Complete data preparation workflow

##  Table of Contents

- [Installation](#installation)
- [Setup](#setup)
- [Dataset Loading](#dataset-loading)
- [Text Normalization](#text-normalization)
- [Vocabulary Extraction](#vocabulary-extraction)
- [Speaker Embedding Creation](#speaker-embedding-creation)
- [Data Preparation](#data-preparation)

##  Installation

Install the required Python packages:

```bash
pip install transformers datasets soundfile accelerate speechbrain==0.5.16
```

##  Setup

Authenticate your Hugging Face account to access models and datasets:

```python
from huggingface_hub import notebook_login
notebook_login()  # Login with your Hugging Face token
```

##  Dataset Loading

Load the English technical text-to-speech dataset:

```python
from datasets import load_dataset, Audio
dataset = load_dataset("Yassmen/TTS_English_Technical_data", split="train")

# Resample audio to 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

##  Text Normalization

Implement text normalization for improved model performance:

```python
import re

def normalize_text(text):
    text = text.lower()  # Lowercase conversion
    text = re.sub(r'[^\w\s\']', '', text)  # Remove punctuation
    return ' '.join(text.split())  # Remove extra whitespace

def add_normalized_text(example):
    example['normalized_text'] = normalize_text(example['transcription'])
    return example

dataset = dataset.map(add_normalized_text)
```

##  Vocabulary Extraction

Extract and compare dataset vocabulary with tokenizer vocabulary:

```python
def extract_all_chars(batch):
    all_text = " ".join(batch["normalized_text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = dataset.map(
    extract_all_chars,
    batched=True,
    keep_in_memory=True,
    remove_columns=dataset.column_names
)

# Compare vocabularies
dataset_vocab = set(vocabs["vocab"][0])
tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
print(dataset_vocab - tokenizer_vocab)  # Displays characters not in tokenizer's vocabulary
```

##  Speaker Embedding Creation

Generate speaker embeddings using SpeechBrain's x-vector model:

```python
from speechbrain.pretrained import EncoderClassifier

speaker_model = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        return speaker_embeddings.squeeze().cpu().numpy()
```

##  Data Preparation

Process the dataset for training:

```python
def prepare_dataset(example):
    audio = example["audio"]
    
    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )
    
    example["labels"] = example["labels"][0]  # Strip off batch dimension
    example["speaker_embeddings"] = create_speaker_embedding(audio["array"])
    
    return example

processed_example = prepare_dataset(dataset[0])
```

##  Conclusion

This repository provides a complete workflow for:

- Loading and preprocessing audio datasets
- Normalizing transcription text
- Creating and managing dataset vocabulary
- Generating speaker embeddings via SpeechBrain
- Preparing datasets for model training

---

💡 For more information about the model or to try it out, visit our [Hugging Face Space](https://huggingface.co/spaces/Aumkeshchy2003/English-tts).
