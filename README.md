# Text-generator-for-learning
A  text generation LLM training project prepared for learning.

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A transformer-based text generation system for news-style content, implemented in PyTorch. This implementation features dynamic text processing, autoregressive decoding, and efficient training routines.

## Features

- **Transformer Architecture**: Custom implementation with positional encoding and masked self-attention
- **Dynamic Text Processing**: 
  - Automatic padding/truncation 
  - SOS/EOS token handling
- **Efficient Training**:
  - Gradient clipping
  - Learning rate scheduling
  - NaN loss detection
  - Model checkpointing
- **Temperature Sampling**: Controlled randomness for text generation
- **OOV Handling**: Robust unknown word handling with `<unk>` tokens

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchtext 0.15+
- tqdm

```bash
pip install torch torchtext tqdm
```

## Dataset Preparation
- 1.Create a news_data.txt file with the following format:
  ```
  Breaking news: Major tech company announces breakthrough...
  Sports update: Championship game ends with historic upset...
  Political development: New legislation passes with bipartisan support...
  Technology update: AI system achieves human-level performance...
  ```
- 2.Data Format Requirements:
  - One complete news item per line
  - Minimum 10,000 samples recommended
  - Include at least 5 categories (e.g., sports, tech, politics)
  - UTF-8 encoding
  
## Training
- Configure parameters in main():
  ```
  # Training Parameters
  BATCH_SIZE = 8       # Number of samples per batch
  SEQ_LENGTH = 128     # Maximum token sequence length 
  EPOCHS = 10          # Total training iterations
  LEARNING_RATE = 1e-4 # Initial learning rate
  WARMUP_STEPS = 2000  # Warmup steps for learning rate

  # Model Architecture
  EMBEDDING_DIM = 512
  NHEAD = 8
  NUM_LAYERS = 6
  ```
- Start training:
  ```
  python news_generator.py
  ```
- Checkpoints save to:
  ```
  best_model_epochX.pt
  ```

## Text Generation
- Generate with temperature control:
  ```
  test_output = generate_text(
    prompt="Technology breakthrough",
    model=model,
    vocab=train_dataset.vocab,
    tokenizer=train_dataset.tokenizer,
    temperature=0.7  # [0.1-1.0] Lower = more deterministic
  )
  ```

## Example Output
- Input Prompt:
  ```
  "Technology breakthrough"
  ```
- Generated Text:
  ```
  technology breakthrough in quantum computing achieved by researchers at stanford 
  university could revolutionize data encryption methods the team demonstrated a new approach to...
  ```

## Model Architecture
```
NewsGenerator(
  (embedding): Embedding(32000, 512)
  (transformer): Transformer(
    (encoder): TransformerEncoder(...)
    (decoder): TransformerDecoder(...)
  )
  (fc_out): Linear(in_features=512, out_features=32000, bias=True)
)
```
## License
MIT License - See LICENSE for details


