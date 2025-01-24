# Fine-Tuning SpeechT5 for English Speech-to-Text with a Focus on Technical Terminology
Link to huggingface space:

                         Aumkeshchy2003/English-tts

This repository demonstrates how to train a Speech-to-Text model using Hugging Face's transformers library with audio datasets and SpeechBrain for speaker embeddings. The code processes audio and text data, performs text normalization, and creates speaker embeddings for a technical text-to-speech model.

**Table of Contents**

    Installation
    Setup
    Dataset Loading
    Text Normalization
    Vocabulary Extraction
    Speaker Embedding Creation
    Data Preparation

**Installation**

To run this code, make sure to install the required Python packages:

       pip install transformers datasets soundfile accelerate speechbrain==0.5.16

**Setup**

Before proceeding, you'll need to authenticate your Hugging Face account to download models and datasets.

      from huggingface_hub import notebook_login
      notebook_login()  # Login with your Hugging Face token
 
This step enables access to Hugging Face models and datasets via your account.

**Dataset Loading**

We load the English technical text-to-speech dataset (TTS_English_Technical_data) from Hugging Face:


      from datasets import load_dataset, Audio
      dataset = load_dataset("Yassmen/TTS_English_Technical_data", split="train")

This dataset includes audio and transcription features for training speech-to-text models. The audio data is resampled to 16kHz:

      dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
     
**Text Normalization**

Text normalization converts text to lowercase and removes unnecessary characters, ensuring better model performance. Here's a function for normalization:

      import re
      def normalize_text(text):
      text = text.lower()  # Lowercase conversion
      text = re.sub(r'[^\w\s\']', '', text)  # Remove punctuation
      return ' '.join(text.split())  # Remove extra whitespace

We then add a normalized_text column to the dataset:

       def add_normalized_text(example):
         example['normalized_text'] = normalize_text(example['transcription'])
         return example

       dataset = dataset.map(add_normalized_text)

**Vocabulary Extraction**

To match the audio transcriptions with the tokenizer's vocabulary, we extract the unique characters in the dataset and compare them with the tokenizer's vocabulary:


       def extract_all_chars(batch):
         all_text = " ".join(batch["normalized_text"])
         vocab = list(set(all_text))
         return {"vocab": [vocab], "all_text": [all_text]}

      vocabs = dataset.map(extract_all_chars, batched=True, keep_in_memory=True, 
      remove_columns=dataset.column_names)

Next, we check for differences between the dataset vocabulary and the tokenizer vocabulary:

      dataset_vocab = set(vocabs["vocab"][0])
      tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
      print(dataset_vocab - tokenizer_vocab)  # Displays characters not in tokenizer's vocabulary
 
**Speaker Embedding Creation**

Speaker embeddings are crucial for speaker recognition and clustering tasks. We use the SpeechBrain x-vector model to generate speaker embeddings:

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

**Data Preparation**

Finally, we process the dataset by tokenizing text and extracting the corresponding audio and speaker embeddings:

      def prepare_dataset(example):
        audio = example["audio"]

      example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
      )

       example["labels"] = example["labels"][0]  # Strip off batch dimension
       example["speaker_embeddings"] = create_speaker_embedding(audio["array"])  # Speaker embeddings

       return example

The dataset is processed and ready for model training:

       processed_example = prepare_dataset(dataset[0])

This completes the steps required to prepare the data for training a Speech-to-Text model.


**Conclusion**

This repository outlines how to:

    Load and preprocess an audio dataset.
    Normalize transcription text.
    Create vocabulary for the dataset.
    Generate speaker embeddings using SpeechBrain.
    Prepare the dataset for training.
