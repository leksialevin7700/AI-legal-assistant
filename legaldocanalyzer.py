

import os
import zipfile

import zipfile
import os

# Paths
outer_zip_path = "/content/7152317.zip"  # Your main zip file
extract_outer_path = "/content/dataset"  # First extraction

# First extraction (7152317.zip)
with zipfile.ZipFile(outer_zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_outer_path)

print(f"✅ First extraction complete: {extract_outer_path}")

# Now extract the inner dataset.zip
inner_zip_path = os.path.join(extract_outer_path, "dataset.zip")
final_extract_path = "/content/dataset_extracted"

if os.path.exists(inner_zip_path):
    with zipfile.ZipFile(inner_zip_path, 'r') as zip_ref:
        zip_ref.extractall(final_extract_path)
    print(f"✅ Second extraction complete: {final_extract_path}")
else:
    print("⚠️ Inner dataset.zip not found!")

import os

# List extracted folder structure
for root, dirs, files in os.walk(final_extract_path):
    print(f"📂 {root}")
    for dir_name in dirs:
        print(f"📁 {dir_name}")
    for file_name in files:
        print(f"📄 {file_name}")
    print("-" * 50)

"""##extracting the files in json format from the structure"""

import os
import json

# Path to dataset
dataset_path = "/content/dataset_extracted/dataset"

# Folders to load
folders = ["IN-Abs", "IN-Ext", "UK-Abs"]

# Data structure
legal_data = []

# Function to load text files from a given folder
def load_text_files(folder_path):
    data = {}
    for category in ["judgement", "summary"]:
        cat_path = os.path.join(folder_path, category)
        if os.path.exists(cat_path):
            data[category] = {}
            for file_name in sorted(os.listdir(cat_path)):  # Sorting ensures pairing is correct
                file_path = os.path.join(cat_path, file_name)
                if file_name.endswith(".txt"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        data[category][file_name] = f.read()
    return data

# Loop through each main folder (IN-Abs, IN-Ext, UK-Abs)
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)

    # Check for train/test data or normal structure
    if folder == "IN-Abs":
        # Process train & test data
        for split in ["train-data", "test-data"]:
            split_path = os.path.join(folder_path, split)
            if os.path.exists(split_path):
                split_data = load_text_files(split_path)
                for file_name in split_data.get("judgement", {}):
                    legal_data.append({
                        "category": folder,
                        "split": split,
                        "file": file_name,
                        "judgement": split_data["judgement"].get(file_name, ""),
                        "summary": split_data["summary"].get(file_name, ""),
                    })
    else:
        # Process normal structure (judgement/summary)
        split_data = load_text_files(folder_path)
        for file_name in split_data.get("judgement", {}):
            legal_data.append({
                "category": folder,
                "split": "full",
                "file": file_name,
                "judgement": split_data["judgement"].get(file_name, ""),
                "summary": split_data["summary"].get(file_name, ""),
            })

print(f"✅ Loaded {len(legal_data)} documents from {folders}!")

# Save structured dataset to JSON
json_path = "/content/legal_dataset.json"
with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(legal_data, json_file, indent=4, ensure_ascii=False)

print(f"✅ Dataset saved as {json_path}")

"""#pre-processing the legal text for trnsformer model"""

import re
import json
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt_tab')

# loading dataset
json_path = "/content/legal_dataset.json"
with open(json_path, "r", encoding="utf-8") as f:
    legal_data = json.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"[^\w\s.]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d{4}\s\w+", "", text)

    #sentence tokenization
    sentences = sent_tokenize(text)

    return " ".join(sentences)

#preprocessing the whole text
for entry in legal_data:
    entry["judgement"] = preprocess_text(entry["judgement"])
    entry["summary"] = preprocess_text(entry["summary"])

# here is saving of cleaned dataset
cleaned_json_path = "/content/legal_dataset_cleaned.json"
with open(cleaned_json_path, "w", encoding="utf-8") as json_file:
    json.dump(legal_data, json_file, indent=4, ensure_ascii=False)

print(f"Preprocessed and saved dataset as {cleaned_json_path}")

from collections import Counter

word_freq = Counter()
for entry in legal_data:
    words = entry["judgement"].split()
    word_freq.update(words)


top_words = word_freq.most_common(50)

print("top 50 legal terms in dataset:")
for word, freq in top_words:
    print(f"{word}: {freq}")

import os
import collections

def count_tokens(directory):
    """
    Reads all text files in a directory and counts unique tokens.
    """
    token_counts = collections.Counter()

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):  # Read only text files
                try:
                    # Try reading with utf-8 first
                    with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                        words = f.read().split()  # Basic tokenization (space-split)
                        token_counts.update(words)
                except UnicodeDecodeError:
                    # If utf-8 fails, try 'latin-1' or 'ISO-8859-1'
                    try:
                        with open(os.path.join(root, file), "r", encoding="latin-1") as f:
                            words = f.read().split()
                            token_counts.update(words)
                    except UnicodeDecodeError:
                        print(f"Skipping file {file} due to encoding issues.")
                        # You can add more encodings to try here, or log the skipped files

    return token_counts

# Path to dataset (modify based on your extracted location)
dataset_path = "/content/dataset_extracted/dataset"

# Count unique tokens
token_counts = count_tokens(dataset_path)

# Display vocabulary statistics
unique_tokens = len(token_counts)
print(f"Total Unique Tokens in Dataset: {unique_tokens}")

# Get most common words
most_common_words = token_counts.most_common(50)
print("\nTop 50 Most Common Words:", most_common_words)

"""#since our vocab is too large we need to use the hugging face tokes

Initializes a Byte Pair Encoding (BPE) tokenizer → Efficient for legal text.

Uses pre_tokenizers.Whitespace() → Splits words before BPE merges subwords.

Trains tokenizer on dataset with:
vocab_size=50000

min_frequency=2 (removes rare words)

#First we will convert out encoded files to utf-8 since some files are encoded in latin-1
"""

import os

def convert_to_utf8(file_path):
    """Convert a file to UTF-8 encoding and overwrite it."""
    try:
        # Try opening with UTF-8
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        encoding_used = "utf-8"  # No conversion needed
    except UnicodeDecodeError:
        try:
            # Try Latin-1 if UTF-8 fails
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
            encoding_used = "latin-1"
        except UnicodeDecodeError:
            print(f"❌ Skipping file {file_path} due to unknown encoding")
            return False

    # If conversion was needed, overwrite file with UTF-8
    if encoding_used == "latin-1":
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Converted {file_path} from {encoding_used} to UTF-8.")

    return True  # File is now UTF-8

# Convert all text files in dataset
dataset_path = "/content/dataset_extracted/dataset"  # Update if needed
converted_files = []

for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(root, file)
            if convert_to_utf8(file_path):
                converted_files.append(file_path)

print(f"✅ Converted {len(converted_files)} files to UTF-8.")

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors

# Initialize Byte Pair Encoding (BPE) tokenizer
tokenizer = Tokenizer(models.BPE())

# Pre-tokenization (basic splitting before applying BPE)
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Training setup
trainer = trainers.BpeTrainer(
    vocab_size=50000,  # Target vocabulary size
    min_frequency=2,   # Remove very rare words
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]  # Required for Transformer
)

# Collect all text files from dataset
dataset_files = []
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".txt"):
            dataset_files.append(os.path.join(root, file))

# Train tokenizer
tokenizer.train(files=dataset_files, trainer=trainer)

# Save the trained tokenizer
tokenizer.save("legal_tokenizer.json")

print("✅ Tokenizer trained and saved successfully!")

"""#Loading and tokenizing the data with our tokenizer"""

import torch
from tokenizers import Tokenizer
import os

# Load our trained tokenizer
tokenizer = Tokenizer.from_file("legal_tokenizer.json")

# Paths to judgement and summary files
dataset_path = "/content/dataset_extracted/dataset"

judgement_dirs = [
    os.path.join(dataset_path, "IN-Abs/train-data/judgement"),
    os.path.join(dataset_path, "IN-Ext/judgement"),  # ✅ Corrected path
    os.path.join(dataset_path, "UK-Abs/train-data/judgement"),
]

summary_dirs = [
    os.path.join(dataset_path, "IN-Abs/train-data/summary"),
    os.path.join(dataset_path, "IN-Ext/summary"),  # ✅ Corrected path
    os.path.join(dataset_path, "UK-Abs/train-data/summary"),
]

# Define max sequence length for tokenization
MAX_SEQ_LENGTH = 512

# Function to read text from files and truncate
def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read().strip()
        return text[:MAX_SEQ_LENGTH]  # ✅ Truncate before tokenization

# Tokenize dataset
input_texts, target_texts = [], []

for judgement_dir, summary_dir in zip(judgement_dirs, summary_dirs):
    if not os.path.exists(judgement_dir) or not os.path.exists(summary_dir):
        print(f"⚠️ Skipping: {judgement_dir} or {summary_dir} does not exist.")
        continue  # Skip directories that do not exist

    for file in os.listdir(judgement_dir):
        if file.endswith(".txt"):
            judgement_path = os.path.join(judgement_dir, file)
            summary_path = os.path.join(summary_dir, file)  # Corresponding summary

            if os.path.exists(summary_path):  # Ensure both exist
                input_text = read_text(judgement_path)
                target_text = read_text(summary_path)

                input_texts.append(input_text)
                target_texts.append(target_text)

print(f"✅ Loaded {len(input_texts)} judgement-summary pairs!")

# **Tokenization WITHOUT truncation (handled manually before)**
input_encodings = tokenizer.encode_batch(input_texts)
target_encodings = tokenizer.encode_batch(target_texts)

# Convert tokenized output into tensors
input_ids = [torch.tensor(encoding.ids, dtype=torch.long) for encoding in input_encodings]
target_ids = [torch.tensor(encoding.ids, dtype=torch.long) for encoding in target_encodings]

# **Ensure fixed-length padding**
input_ids = torch.nn.utils.rnn.pad_sequence(
    input_ids, batch_first=True, padding_value=0
)
target_ids = torch.nn.utils.rnn.pad_sequence(
    target_ids, batch_first=True, padding_value=0
)

print(f"✅ Tokenized, truncated manually, and padded dataset! Max length: {MAX_SEQ_LENGTH}")

# Save preprocessed data
torch.save((input_ids, target_ids), "preprocessed_legal_data.pt")
print("✅ Preprocessed dataset saved successfully!")

"""##Transformer Architecture

# positional embeddings
"""

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Position indices (0, 1, 2, ... max_len)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute sine and cosine terms
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        # Add batch dimension
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1)]

"""#multihead attention for to understand the context of each query and word for all keys(text) so that it understands the whole aspects of a single token.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0  # Ensure divisibility

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # Each head gets a portion

        # Linear layers for Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        # 1️⃣ Transform inputs: (batch, seq_len, d_model) → (batch, num_heads, seq_len, head_dim)
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2️⃣ Compute attention scores (scaled dot-product)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3️⃣ Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # Softmax over last dimension

        # 4️⃣ Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, V)

        # 5️⃣ Concatenate heads: (batch, num_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(attn_output)  # Final output projection

"""##feed forward network to find non-linear patterns and make model understands different importance of words at different places"""

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))  # Apply ReLU activation

"""#Each Encoder Layer consists of:

Multi-Head Attention

Feedforward Layer

Layer Normalization

Residual Connections
"""

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, hidden_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply Multi-Head Attention with Residual Connection or skip connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm

        # Apply Feedforward Layer with Residual Connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm

        return x

"""#Transformer Decoder Layer

Each Decoder Layer has:

Masked Multi-Head Attention (prevents looking ahead)

Multi-Head Attention on Encoder Output

Feedforward Layer

Layer Normalization

Residual Connections
"""

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, hidden_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1️⃣ Masked Self-Attention (prevents looking ahead)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm

        # 2️⃣ Cross Attention with Encoder Output
        attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))  # Add & Norm

        # 3️⃣ Feedforward Layer
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))  # Add & Norm

        return x

"""#Complete Transformer Model

Now, we stack multiple encoder & decoder layers into a full Transformer.
"""

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, hidden_dim, max_length, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(d_model, vocab_size)

    def encode(self, src, src_mask=None):
        x = self.embedding(src)  # ✅ Correct embedding first
        x = x + self.pos_encoding(x)  # ✅ Add positional encoding
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt)  # ✅ Correct embedding first
        x = x + self.pos_encoding(x)  # ✅ Add positional encoding
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return output

"""#Now that we have built our custom Transformer architecture, we need to train it using supervised learning.

We will:

Define Loss Function: Cross-entropy loss.

Set Optimizer: AdamW with weight decay.

Implement Training Loop: Mini-batch training.

Handle Teacher Forcing: Improve decoder learning.

# Training Configuration
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
VOCAB_SIZE = 50000  # Based on our tokenizer
D_MODEL = 512       # Embedding size
NUM_HEADS = 8       # Number of attention heads
NUM_LAYERS = 6      # Number of transformer layers
HIDDEN_DIM = 2048   # Feedforward hidden layer size
MAX_LENGTH = 1024   # Maximum token length per sequence
BATCH_SIZE = 64   # Training batch size
EPOCHS = 50         # Number of training epochs
LR = 5e-5           # Learning rate
PATIENCE = 5

# Load preprocessed dataset
input_ids, target_ids = torch.load("preprocessed_legal_data.pt")

# Create dataset & dataloader
dataset = TensorDataset(input_ids, target_ids)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, HIDDEN_DIM, MAX_LENGTH)
model = model.to("cuda")

"""#loss function & Optimizer

We use CrossEntropyLoss to train the model.
"""

# Loss function (ignore padding index)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Padding token ID is 0

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LR)

"""#Training Loop with Teacher Forcing

We will:

Use teacher forcing to train the decoder efficiently.

Iterate over mini-batches for training.
"""

from torch.optim.lr_scheduler import ReduceLROnPlateau
# Learning Rate Scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Early Stopping Variables
best_loss = float("inf")
patience_counter = 0

# Training function with Early Stopping & LR Scheduling
def train_model(model, dataloader, optimizer, loss_fn, scheduler, epochs, patience):
    global best_loss, patience_counter
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to("cuda"), tgt.to("cuda")

            optimizer.zero_grad()

            # Prepare decoder input (shifted right target sequence)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Get model predictions
            predictions = model(src, tgt_input)

            # Compute loss (reshape predictions and targets)
            loss = loss_fn(predictions.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{batch_idx}/{len(dataloader)}] | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"epoch {epoch+1} completed! Average Loss: {avg_loss:.4f}")

        # Update Learning Rate
        scheduler.step(avg_loss)

        # Early Stopping Check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model, "best_legal_transformer.pth")  # Save best model
            print(f" Model improved! Saved new best model with loss {avg_loss:.4f}")
        else:
            patience_counter += 1
            print(f" No improvement for {patience_counter}/{patience} epochs.")

        # Stop training if patience exceeded
        if patience_counter >= patience:
            print(" early Stopping triggered! Stopping training.")
            break

# Train the model
train_model(model, dataloader, optimizer, loss_fn, scheduler, EPOCHS, PATIENCE)

import torch
from torch.utils.data import DataLoader, TensorDataset

# Load preprocessed dataset
input_ids, target_ids = torch.load("preprocessed_legal_data.pt")

# Split into training & validation (90% train, 10% validation)
split_ratio = 0.9
split_idx = int(len(input_ids) * split_ratio)

train_input_ids, val_input_ids = input_ids[:split_idx], input_ids[split_idx:]
train_target_ids, val_target_ids = target_ids[:split_idx], target_ids[split_idx:]

# Create validation dataset & dataloader
val_dataset = TensorDataset(val_input_ids, val_target_ids)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"✅ Validation Data Loaded: {len(val_input_ids)} samples")

# Re-initialize the model
model = Transformer(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, HIDDEN_DIM, MAX_LENGTH)

# Load the trained weights from the best model
model.load_state_dict(torch.load("best_legal_transformer.pth"))
model = model.to("cuda")  # Move to GPU
model.eval()  # Set model to evaluation mode

print("✅ Best trained model loaded for validation!")

import torch.nn.functional as F

def validate_model(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    all_predictions, all_targets = [], []

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to("cuda"), tgt.to("cuda")

            # Shifted input for decoder
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Get model predictions
            predictions = model(src, tgt_input)

            # Compute loss
            loss = loss_fn(predictions.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
            total_loss += loss.item()

            # Convert token predictions to text
            predicted_tokens = predictions.argmax(dim=-1).cpu().numpy()
            target_tokens = tgt_output.cpu().numpy()

            all_predictions.extend(predicted_tokens)
            all_targets.extend(target_tokens)

    avg_loss = total_loss / len(dataloader)
    print(f"✅ Validation Loss: {avg_loss:.4f}")
    return all_predictions, all_targets, avg_loss

# Define the loss function (same as training)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)  # Padding token ID = 0

# Run validation
predictions, targets, val_loss = validate_model(model, val_dataloader, loss_fn)

!pip install nltk rouge-score

from rouge_score import rouge_scorer

# Function to compute ROUGE score
def compute_rouge(predictions, targets, tokenizer):
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = []

    for pred, target in zip(predictions, targets):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        target_text = tokenizer.decode(target, skip_special_tokens=True)

        score = rouge.score(target_text, pred_text)
        scores.append(score["rougeL"].fmeasure)  # Use ROUGE-L F1-score

    return sum(scores) / len(scores)

# Compute ROUGE
rouge_score = compute_rouge(predictions, targets, tokenizer)
print(f"✅ ROUGE Score: {rouge_score:.4f}")

import nltk
from nltk.translate.bleu_score import sentence_bleu

# Function to compute BLEU score
def compute_bleu(predictions, targets, tokenizer):
    bleu_scores = []

    for pred, target in zip(predictions, targets):
        pred_text = tokenizer.decode(pred, skip_special_tokens=True)
        target_text = tokenizer.decode(target, skip_special_tokens=True)

        # Tokenize sentences
        pred_tokens = pred_text.split()
        target_tokens = [target_text.split()]

        # Compute BLEU
        score = sentence_bleu(target_tokens, pred_tokens)
        bleu_scores.append(score)

    return sum(bleu_scores) / len(bleu_scores)

# Compute BLEU
bleu_score = compute_bleu(predictions, targets, tokenizer)
print(f"✅ BLEU Score: {bleu_score:.4f}")

!pip install stable-baselines3

import numpy as np

# Reward Function: Higher BLEU & ROUGE = Higher Reward
def compute_reward(pred_text, target_text, tokenizer):
    pred_tokens = pred_text.split()
    target_tokens = [target_text.split()]

    # Compute BLEU Score (for fluency)
    bleu = sentence_bleu(target_tokens, pred_tokens)

    # Compute ROUGE Score (for relevance)
    rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_score = rouge.score(target_text, pred_text)

    rougeL = rouge_score["rougeL"].fmeasure  # ROUGE-L F1 score

    # Combine into final reward (weighted sum)
    reward = (0.7 * rougeL) + (0.3 * bleu)

    return reward

!pip install gymnasium

!pip install -U shimmy gymnasium

!pip install stable-baselines3[extra] gymnasium transformers rouge-score sacrebleu

import gymnasium as gym  # 🔼 Updated import
import numpy as np
import torch
import json
from gymnasium import spaces  # 🔼 Updated import
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
import sacrebleu
from tokenizers import Tokenizer

# 1. Load Tokenizer
tokenizer = Tokenizer.from_file("legal_tokenizer.json")

# 2. Load Model with Security Fix
model = torch.load(
    "best_legal_transformer.pth",
    map_location="cuda",
    weights_only=True  # 🔼 Security fix for untrusted models
)

# 3. Load Dataset with Security Fix
input_ids, target_ids = torch.load(
    "preprocessed_legal_data.pt",
    weights_only=True  # 🔼 Security fix
)
val_dataset = torch.utils.data.TensorDataset(input_ids, target_ids)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# Hyperparameters
MAX_LENGTH = 512
VOCAB_SIZE = 50000

def pad_or_truncate(seq, max_length):
    """Ensure fixed sequence length for observation space."""
    if len(seq) > max_length:
        return seq[:max_length]
    return seq + [0] * (max_length - len(seq))

class LegalTextEnv(gym.Env):
    """PPO Environment for Legal Text Summarization"""

    def __init__(self, model, dataloader, tokenizer):
        super(LegalTextEnv, self).__init__()

        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.current_batch = iter(self.dataloader)

        # Corrected Observation Space
        self.observation_space = spaces.Box(
            low=0,
            high=VOCAB_SIZE,
            shape=(MAX_LENGTH,),
            dtype=np.int32
        )

        # Action Space (token selection)
        self.action_space = spaces.Discrete(VOCAB_SIZE)

        # Tracking
        self.current_step = 0
        self.current_input_ids = []

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        super().reset(seed=seed)  # 🔼 Required for Gymnasium

        try:
            self.input_texts, self.target_texts = next(self.current_batch)
        except StopIteration:
            self.current_batch = iter(self.dataloader)
            self.input_texts, self.target_texts = next(self.current_batch)

        # Process input text
        input_text = self.tokenizer.decode(
            self.input_texts[0].tolist(),
            skip_special_tokens=True
        )
        self.current_input_ids = pad_or_truncate(
            self.tokenizer.encode(input_text).ids,
            MAX_LENGTH
        )

        self.current_step = 0
        return np.array(self.current_input_ids, dtype=np.int32), {}  # 🔼 Gymnasium requires info dict

    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1

        # Decode generated token
        pred_text = self.tokenizer.decode([action], skip_special_tokens=True)
        target_text = self.tokenizer.decode(
            self.target_texts[0].tolist(),
            skip_special_tokens=True
        )

        # Calculate enhanced reward
        reward = self._compute_reward(pred_text, target_text)

        # Termination conditions
        terminated = bool(self.current_step >= MAX_LENGTH or action == 0)  # 🔼 Ensure boolean
        truncated = False  # 🔼 Gymnasium requires truncated flag

        return (
            np.array(self.current_input_ids, dtype=np.int32),
            reward,
            terminated,
            truncated,
            {}  # 🔼 Gymnasium requires info dict
        )

    def _compute_reward(self, pred_text, target_text):
        """Enhanced reward function with length penalty"""
        # Original scores
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge = scorer.score(target_text, pred_text)['rougeL'].fmeasure
        bleu = sacrebleu.corpus_bleu([pred_text], [[target_text]]).score / 100

        # Length penalty
        pred_len = len(pred_text.split())
        target_len = len(target_text.split())
        length_penalty = np.exp(-abs(pred_len/target_len - 1))

        # Weighted reward
        return (0.5 * rouge) + (0.3 * bleu) + (0.2 * length_penalty)

    def render(self):
        pass

# 4. Environment Validation
env = LegalTextEnv(model, val_dataloader, tokenizer)
check_env(env)  # 🔼 Validate environment compatibility

# 5. Create Vectorized Environment
env = DummyVecEnv([lambda: env])

# 6. Optimized PPO Configuration
ppo_model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cuda",
    verbose=1,
    learning_rate=1e-5,  # 🔼 **Reduce learning rate (stabilize training)**
    batch_size=256,  # ✅ Keep batch size moderate
    n_steps=2048,  # ✅ Keep steps per update as is
    gamma=0.98,  # ✅ Slightly higher discount factor
    ent_coef=0.05,  # ✅ Reduce entropy coefficient (improve policy stability)
    clip_range=0.1,  # 🔽 **Reduce clip range (prevent over-clipping)**
    max_grad_norm=0.5,  # ✅ Keep max gradient clipping
    vf_coef=2.0,  # 🔼 **Increase value function weight (fix variance issue)**
    n_epochs=5,  # ✅ Reduce epochs (prevent overfitting)
    policy_kwargs={
        "net_arch": [512, 256],  # ✅ Keep architecture same
        "activation_fn": torch.nn.ReLU,
        "ortho_init": True
    }
)

# 7. Add Evaluation Callback
from stable_baselines3.common.callbacks import EvalCallback
eval_callback = EvalCallback(
    env,
    best_model_save_path="./best_ppo/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True
)

# 8. Train with Progress Bar
try:
    ppo_model.learn(
        total_timesteps=150000,
        callback=eval_callback,
        progress_bar=True
    )
except KeyboardInterrupt:
    print("Training interrupted. Saving final model...")
finally:
    ppo_model.save("ppo_finetuned_legal_transformer")
    print("✅ Training complete. Model saved.")

from stable_baselines3 import PPO

# Load trained PPO model
ppo_model = PPO.load("ppo_finetuned_legal_transformer")
print("✅ PPO Model Loaded Successfully!")

import torch
import numpy as np
from tokenizers import Tokenizer
from stable_baselines3 import PPO

# ✅ Load tokenizer properly
tokenizer = Tokenizer.from_file("legal_tokenizer.json")  # JSON tokenizers should be loaded with `from_file()`

# ✅ Load PPO-trained Transformer model
ppo_model = PPO.load("ppo_finetuned_legal_transformer")  # Load fine-tuned PPO model

MAX_LENGTH = 512  # Maximum sequence length

def pad_or_truncate(sequence, max_length):
    """Ensure input sequence has fixed length."""
    if len(sequence) > max_length:
        return sequence[:max_length]  # Truncate if too long
    return sequence + [tokenizer.token_to_id("[PAD]")] * (max_length - len(sequence))  # Pad if too short

def generate_summary(input_text, max_length=MAX_LENGTH):
    """Generate a summary using the fine-tuned PPO model with proper token selection."""
    # Tokenize input text & pad/truncate
    input_ids = tokenizer.encode(input_text).ids
    input_ids_padded = pad_or_truncate(input_ids, max_length)

    # Convert to tensor
    input_tensor = torch.tensor(input_ids_padded, dtype=torch.long).unsqueeze(0).to("cuda")

    # Ensure tensor shape matches PPO expectations
    input_array = input_tensor.cpu().numpy()

    # Generate summary using PPO model
    generated_tokens = []
    obs = input_array  # Initial observation

    for _ in range(max_length):
        action, _ = ppo_model.predict(obs, deterministic=True)

        # ✅ Ensure valid token selection
        action = np.clip(action[0], 0, tokenizer.get_vocab_size() - 1)
        generated_tokens.append(action)

        if action == tokenizer.token_to_id("[SEP]"):  # Stop if SEP token is reached
            break

        # ✅ Update observation with new token
        obs = np.array([pad_or_truncate(generated_tokens, max_length)])

    # Decode generated tokens to text
    summary_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return summary_text

# ✅ Sample Test
test_text = "The court held that the contract was not enforceable due to a lack of mutual agreement..."
generated_summary = generate_summary(test_text)

print("🔹 **Original Legal Text:**")
print(test_text)
print("\n✅ **Generated Summary (After Fixes):**")
print(generated_summary)



