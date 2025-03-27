import torch
import torch.nn as nn
import time
import math
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tqdm import tqdm

# ======================
# 1. Data Preprocessing
# ======================
class NewsDataset(Dataset):
    def __init__(self, texts, seq_length=512):
        self.tokenizer = get_tokenizer('basic_english')
        self.seq_length = seq_length
        
        # Build vocabulary from tokenized texts
        self.vocab = build_vocab_from_iterator(
            (self.tokenizer(text) for text in texts),
            specials=['<unk>', '<pad>', '<sos>', '<eos>']
        )
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        # Preprocess texts with padding/truncation
        self.texts = [self._process_text(text) for text in texts]
    
    def _process_text(self, text):
        """Process individual text into tensor with padding/truncation"""
        tokens = self.tokenizer(text)
        indices = (
            [self.vocab['<sos>']] 
            + [self.vocab[t] for t in tokens] 
            + [self.vocab['<eos>']]
        )
        
        # Truncate or pad sequence
        if len(indices) > self.seq_length:
            return indices[:self.seq_length]
        return indices + [self.vocab['<pad>']] * (self.seq_length - len(indices))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.texts[idx]
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(indices[:-1], dtype=torch.long)  # Autoregressive target
        )

# ======================
# 2. Transformer Model
# ======================
class NewsGenerator(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, d_model))
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        
        output = self.transformer(
            src,
            tgt,
            src_mask=self.generate_square_subsequent_mask(src.size(1)),
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(1))
        )
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for autoregressive decoding"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.embedding.weight.device)

# ======================
# 3. Text Generation
# ======================
def generate_text(prompt, model, vocab, tokenizer, max_length=50, temperature=0.7):
    """Generate text using trained model with temperature sampling"""
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab['<sos>']] + [vocab[token] for token in tokens]
    
    with torch.no_grad():
        for _ in range(max_length):
            src = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
            output = model(src, src[:, :-1])
            
            next_token_logits = output[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1).item()
            
            if next_token == vocab['<eos>']:
                break
            indices.append(next_token)
    
    # Convert indices to text
    itos = vocab.get_itos()
    return ' '.join([itos[idx] for idx in indices if idx not in {vocab['<sos>'], vocab['<eos>']}])

# ======================
# 4. Training Utilities
# ======================
def make_collate_fn(dataset):
    """Create dynamic padding function with dataset-specific padding index"""
    pad_idx = dataset.vocab['<pad>']
    def collate_fn(batch):
        src_list, tgt_list = zip(*batch)
        src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=pad_idx)
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=pad_idx)
        return src_padded.to(device), tgt_padded.to(device)
    return collate_fn

# ======================
# 5. Training Progress
# ======================
def train_model():
    """Main training loop with progress tracking and model saving"""
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_batches = len(train_loader)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (src, tgt) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with shifted target
            output = model(src, tgt[:, :-1])
            output = output.reshape(-1, vocab_size)
            tgt = tgt[:, 1:].reshape(-1)

            loss = criterion(output, tgt)
            loss.backward()

            # Gradient handling
            if math.isnan(loss.item()):
                print("NaN loss detected, skipping update")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item() * src.size(0)

        # Epoch statistics
        avg_loss = epoch_loss / len(train_dataset)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")

        # Learning rate scheduling
        if (epoch + 1) % 3 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1
                print(f"Learning rate reduced to {param_group['lr']:.6f}")

        # Model checkpointing
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"best_model_epoch{epoch+1}.pt")

    print(f"Training complete! Best loss: {best_loss:.4f}")

# ======================
# 6. Main Execution
# ======================
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 8
    SEQ_LENGTH = 128
    EPOCHS = 10
    LEARNING_RATE = 1e-4

    # Data loading
    with open("news_data.txt", "r", encoding="utf-8") as f:
        news_texts = [line.strip() for line in f if line.strip()]
    
    # Initialize components
    train_dataset = NewsDataset(news_texts)
    collate_fn = make_collate_fn(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    # Model setup
    vocab_size = len(train_dataset.vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NewsGenerator(vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab['<pad>'])
    
    # Start training
    train_model()
    
    # Generation demo
    print("\nGeneration demo:")
    sample_token = "supercalifragilistic"
    print(f"OOV test: '{sample_token}' -> {train_dataset.vocab[sample_token]} "
          f"(<unk> index: {train_dataset.vocab['<unk>']})")

    test_output = generate_text(
        prompt="Technology breakthrough",
        model=model,
        vocab=train_dataset.vocab,
        tokenizer=train_dataset.tokenizer
    )
    print(f"\nGenerated text: {test_output}")
