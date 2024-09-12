# app.py
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define parameters from your Colab training environment
batch_size = 16
block_size = 32
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.1
device = 'cpu'

# Exact character set used in Colab
chars = [' ', '"', '&', "'", '(', ')', ',', '-', '.', '4', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
         'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 
         'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\x92']
vocab_size = len(chars)  # This should be 61 to match the trained model

# Create mappings for characters to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode function to convert input string to integers
def encode(input_string):
    return [stoi.get(c, 0) for c in input_string]  # Default to 0 if character is not found

# Decode function to convert integers back to string
def decode(output_tensor):
    return ''.join([itos.get(i, '?') for i in output_tensor])  # Use '?' if index is not found

# Define the model architecture
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # Apply transformer blocks
        x = self.ln_f(x)    # Final layer normalization
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)  # Calculate loss

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]  # Condition on the last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on the last token's logits
            probs = torch.softmax(logits, dim=-1)  # Convert to probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample the next token
            idx = torch.cat((idx, idx_next), dim=1)  # Append the new token to the sequence
        return idx

# Define the transformer block
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# Multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# Single attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# Feedforward layer
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = BigramLanguageModel()
model.load_state_dict(torch.load('triptuner_model.pth', map_location=device))
model.eval()

# Generate information based on location name
def generate_info(location_name):
    context = torch.tensor([encode(location_name)], dtype=torch.long, device=device)
    generated_info = model.generate(context, max_new_tokens=1000)
    return decode(generated_info[0].tolist())

# Define the API endpoint
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    location_name = data.get('location_name')
    if not location_name:
        return jsonify({'error': 'Location name is required'}), 400

    generated_text = generate_info(location_name)
    return jsonify({'location_name': location_name, 'information': generated_text})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
