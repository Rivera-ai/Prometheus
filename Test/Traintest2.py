import math
import random
import tqdm
import numpy as np
import torch
from torch.optim import AdamW
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from PrometheusCore import Prometheus
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# constants
NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRAD_ACCUM_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY = 100
PRIME_LENGTH = 64
GENERATE_EVERY = 500
GENERATE_LENGTH = 256
SEQ_LEN = 256

# helpers
def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

# model definition
model = Prometheus(
    num_text_tokens = 256,
    transformer = dict(
        dim = 384,
        depth = 8,
        dim_head = 64,
        heads = 8
    )
).to(device=device)

class TextSamplerDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = []
        
        # Convertir textos a bytes y concatenarlos
        for item in data:
            text = item['text']
            if not isinstance(text, str):
                text = str(text)
            bytes_array = np.frombuffer(text.encode('utf-8'), dtype=np.uint8)
            self.data.extend(bytes_array)
        
        self.data = torch.tensor(self.data, dtype=torch.long)
        self.data_length = len(self.data)

    def __len__(self):
        return self.data_length // self.seq_len

    def __getitem__(self, index):
        rand_start = torch.randint(0, self.data_length - self.seq_len - 1, (1,))
        full_seq = self.data[rand_start : rand_start + self.seq_len + 1].long()
        return full_seq

# Cargar OpenWebText
print("Loading OpenWebText dataset...")
dataset = load_dataset("stas/openwebtext-10k")

# Dividir en train y validation
train_size = int(0.9 * len(dataset['train']))
val_size = len(dataset['train']) - train_size

train_data = dataset['train'].select(range(train_size))
val_data = dataset['train'].select(range(train_size, len(dataset['train'])))

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Crear datasets
train_dataset = TextSamplerDataset(train_data, SEQ_LEN)
val_dataset = TextSamplerDataset(val_data, SEQ_LEN)

# Crear dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# optimizer
optim = AdamW(model.parameters(), lr=LEARNING_RATE)

train_loader = cycle(train_loader)
val_loader = cycle(val_loader)

# training
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10.0, desc="training"):
    model.train()
    
    for _ in range(GRAD_ACCUM_EVERY):
        data = next(train_loader)
        loss = model(data.to(device=device))
        (loss / GRAD_ACCUM_EVERY).backward()
    
    print(f'loss: {loss.item():.3f}')
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optim.step()
    optim.zero_grad()

    if divisible_by(i, VALIDATE_EVERY):
        model.eval()
        with torch.no_grad():
            valid_data = next(val_loader)
            loss = model(valid_data.to(device=device))
            print(f'\nvalid loss: {loss.item():.3f}\n')

    if divisible_by(i, GENERATE_EVERY):
        model.eval()
        inp = random.choice(val_dataset)[:PRIME_LENGTH]
        inp = inp.to(device=device)
        prime = decode_tokens(inp)
        print(f"\nprime: {prime}\n")
        prompt = inp[None, ...]
        sampled = model.generate_text_only(prompt, GENERATE_LENGTH)
        base_decode_output = decode_tokens(sampled[0])
        print(f"\ngenerated: {base_decode_output}\n")