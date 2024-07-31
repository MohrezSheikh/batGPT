# Import necessary libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

# Define hyperparameters
hyperparameters = {
    'batch_size': 16,
    'block_size': 32,
    'max_iters': 10000,
    'eval_interval': 100,
    'learning_rate': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_iters': 200,
    'n_embd': 64,
    'n_head': 4,
    'n_layer': 4,
    'dropout': 0.0
}

# Load data
def load_data(url):
    response = requests.get(url)
    with open('BatmanCorpus.txt', 'w', encoding='utf-8') as file:
        file.write(response.text)
    with open('BatmanCorpus.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# Preprocess data
def preprocess_data(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, vocab_size, stoi, itos, encode, decode

# Define model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.query = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.value = nn.Linear(hyperparameters['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(hyperparameters['block_size'], hyperparameters['block_size'])))
        self.dropout = nn.Dropout(hyperparameters['dropout'])

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(hyperparameters['n_embd'], hyperparameters['n_embd'])
        self.dropout = nn.Dropout(hyperparameters['dropout'])

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(hyperparameters['dropout'])
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(hyperparameters['vocab_size'], hyperparameters['n_embd'])
        self.position_embedding_table = nn.Embedding(hyperparameters['block_size'], hyperparameters['n_embd'])
        self.blocks = nn.Sequential(*[Block(hyperparameters['n_embd'], n_head=hyperparameters['n_head']) for _ in range(hyperparameters['n_layer'])])
        self.ln_f = nn.LayerNorm(hyperparameters['n_embd'])
        self.lm_head = nn.Linear(hyperparameters['n_embd'], hyperparameters['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=hyperparameters['device']))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -hyperparameters['block_size']:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Load data
text = load_data("https://raw.githubusercontent.com/MohrezSheikh/batGPT/master/BatmanCorpus.txt")

# Preprocess data
train_data, val_data, vocab_size, stoi, itos, encode, decode = preprocess_data(text)

# Define hyperparameters
hyperparameters['vocab_size'] = vocab_size

# Initialize model
model = BigramLanguageModel()

# Move model to device
model.to(hyperparameters['device'])

# Print number of parameters
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

# Define optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameters['learning_rate'])

# Train model
for iter in range(hyperparameters['max_iters']):
    if iter % hyperparameters['eval_interval'] == 0 or iter == hyperparameters['max_iters'] - 1:
        losses = estimate_loss(model, train_data, val_data, hyperparameters)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch(train_data, hyperparameters['batch_size'], hyperparameters['block_size'])
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=hyperparameters['device'])
print(decode(model.generate(context, hyperparameters['block_size'] * 2)[0].tolist()))