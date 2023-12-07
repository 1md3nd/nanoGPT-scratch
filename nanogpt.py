import time
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64  # size of parallel batches of block_size (batch_dimension)
block_size = 128  # size of the chunk of data we process (time_dimension)
max_iter = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iter = 200
n_embd = 128
n_layer = 4
head_size = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Read and preprocess text data
with open('Rabindranath.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mappings between characters and integers
str2int = {ch: i for i, ch in enumerate(chars)}
int2str = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [str2int[c] for c in s]
decode = lambda l: ''.join([int2str[n] for n in l])

# Convert text to tensor
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # to get a random chunk of data for each training or validation
    data_split = train_data if split == 'train' else val_data
    idx = torch.randint(len(data_split) - block_size - 1, (batch_size,))
    x = torch.stack([data_split[i: block_size + i] for i in idx])
    y = torch.stack([data_split[i + 1: block_size + i + 1] for i in idx])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    losses = {}
    model.eval()
    for split in ['train', 'val']:
        loss = torch.zeros(eval_iter)
        for k in range(eval_iter):
            Xe, Ye = get_batch(split)
            _, lss = model(Xe, Ye)
            loss[k] = lss.item()
        losses[split] = loss.mean()
    model.train()
    return losses


class Head(nn.Module):
    """Self-attention head module."""
    def __init__(self, n_embd, head_size, block_size, dropout) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Attention calculation
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, num_heads, n_embd, block_size, dropout) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, n_embd // num_heads, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from different attention heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Feedforward module."""
    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    """Transformer block module."""
    def __init__(self, n_embd, n_head, block_size, dropout) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, n_embd, block_size, dropout)
        self.network = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Self-attention and feedforward steps
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.network(self.ln2(x))
        return x

class BigramModel(nn.Module):
    """Bigram language model with optimizations."""
    def __init__(self, vocab_size,):
        super().__init__()

        # Embedding layers
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=head_size) for _ in range(n_layer)])

        # Layer normalization for the final output
        self.ln_f = nn.LayerNorm(n_embd)

        # Linear layer for generating logits
        self.lm_head = nn.Linear(n_embd, vocab_size)


    def forward(self, idx, target=None):
        # Input shape: (BATCH_SIZE, SEQUENCE_LENGTH)
        B, T = idx.shape

        # Token and positional embeddings
        token_embd = self.token_embedding_table(idx)
        pos_embd = self.positional_embedding_table(torch.arange(T, device=self.device))
        x = token_embd + pos_embd

        # Transformer blocks
        x = self.blocks(x)

        # Layer normalization
        x = self.ln_f(x)

        # Generate logits
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            # Reshape logits and targets for computing cross-entropy loss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)

            # Compute cross-entropy loss
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            # Extract the last block_size tokens for generation
            idx_con = idx[:, -block_size:]

            # Forward pass for generating the next token
            logits, _ = self.forward(idx_con)

            # Extract the last token's logits
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token
            sample_idx = torch.multinomial(probs, num_samples=1)

            # Concatenate the sampled token to the input for the next iteration
            idx = torch.cat((idx, sample_idx), dim=1)

        return idx


model = BigramModel()
model.to(device)

# PyTorch optimizer model AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

st = time.time()
for curr_itr in range(max_iter):
    if curr_itr % eval_interval == 0:
        losses = estimate_loss()
        print(f"At iteration {curr_itr}, the train loss: {losses['train']:.4f} and val loss: {losses['val']:.4f}")
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
et = time.time()
print('Took', (et - st) % 60, 'seconds')
print('Final loss', loss.item())

input_idx = torch.zeros((1, 1), dtype=torch.long, device=device)
# generated tokens
g_idx = model.generate(input_idx, 300)

out = decode(g_idx[0].tolist())
print(out)
