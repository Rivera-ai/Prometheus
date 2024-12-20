from pathlib import Path
import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from einops import rearrange
from einops.layers.torch import Rearrange
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from datasets import load_dataset
import numpy as np
from Model.PrometheusModel import Prometheus, print_modality_sample

# Configuración general
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = Path('./checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

# Configuración de hiperparámetros
dim_latent = 16
autoencoder_train_steps = 15000
transfusion_train_steps = 2500
text_train_steps = 10000
batch_size = 32
text_seq_len = 256

# Utilidades
def save_checkpoint(model, epoch, loss, stage, checkpoint_dir='checkpoints'):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    checkpoint_file = checkpoint_path / f'checkpoint_{stage}_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_file)
    latest_checkpoint = checkpoint_path / f'checkpoint_{stage}_latest.pt'
    torch.save(checkpoint, latest_checkpoint)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

# Dataset MNIST
class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST('./data/mnist', download=True)
        self.transform = T.Compose([
            T.PILToTensor(),
            T.RandomResizedCrop((28, 28), scale=(0.8, 1.))
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = self.transform(pil)
        return tensor(labels), (digit_tensor / 255).float()

# Dataset de Texto
class TextDataset(Dataset):
    def __init__(self, data, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.data = []
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

# Componentes del autoencoder
class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

encoder = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1),
    nn.Conv2d(4, 8, 4, 2, 1),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Conv2d(8, dim_latent, 1),
    Rearrange('b d ... -> b ... d'),
    Normalize()
).to(device)

decoder = nn.Sequential(
    Rearrange('b ... d -> b d ...'),
    nn.Conv2d(dim_latent, 8, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, 2, 1),
    nn.Conv2d(4, 1, 3, padding=1),
).to(device)

# Función principal de entrenamiento
def main():
    print("Fase 1: Entrenamiento del Autoencoder")
    # Entrenamiento del autoencoder
    mnist_dataset = MnistDataset()
    autoencoder_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    autoencoder_iter_dl = cycle(autoencoder_dataloader)
    autoencoder_optimizer = AdamW([*encoder.parameters(), *decoder.parameters()], lr=3e-4)

    with tqdm(total=autoencoder_train_steps) as pbar:
        for step in range(autoencoder_train_steps):
            _, images = next(autoencoder_iter_dl)
            images = images.to(device)
            latents = encoder(images)
            latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2)
            reconstructed = decoder(latents)
            loss = F.mse_loss(images, reconstructed)
            loss.backward()
            pbar.set_description(f'autoencoder loss: {loss.item():.5f}')
            autoencoder_optimizer.step()
            autoencoder_optimizer.zero_grad()
            pbar.update()

    print("\nFase 2: Entrenamiento Inicial de Prometheus con MNIST")
    # Configuración inicial de Prometheus
    model = Prometheus(
        num_text_tokens=256,  # Aumentado para manejar bytes
        dim_latent=dim_latent,
        modality_default_shape=(14, 14),
        modality_encoder=encoder,
        modality_decoder=decoder,
        add_pos_emb=True,
        modality_num_dim=2,
        transformer=dict(
            dim=384,
            depth=8,
            dim_head=64,
            heads=8,
        )
    ).to(device)

    # Entrenamiento con MNIST
    dataloader = model.create_dataloader(mnist_dataset, batch_size=16, shuffle=True)
    iter_dl = cycle(dataloader)
    optimizer = AdamW(model.parameters_without_encoder_decoder(), lr=3e-4)

    with tqdm(total=transfusion_train_steps) as pbar:
        for step in range(transfusion_train_steps):
            model.train()
            loss = model(next(iter_dl))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            pbar.set_description(f'prometheus mnist loss: {loss.item():.3f}')
            pbar.update()

            if (step + 1) % 500 == 0:
                save_checkpoint(model, step, loss.item(), 'mnist')
                
    # Guardar el modelo después del entrenamiento con MNIST
    save_checkpoint(model, transfusion_train_steps, loss.item(), 'mnist_final')
    
    print("\nFase 3: Entrenamiento con Texto")
    # Cargar OpenWebText
    print("Cargando OpenWebText dataset...")
    text_dataset = load_dataset("stas/openwebtext-10k")
    train_dataset = TextDataset(text_dataset['train'], text_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    train_iter = cycle(train_loader)
    
    # Cargar el modelo guardado
    latest_checkpoint_path = checkpoint_dir / 'checkpoint_mnist_final_epoch_2500.pt'
    epoch, loss = load_checkpoint(model, latest_checkpoint_path)
    print(f"Modelo cargado desde época {epoch} con pérdida {loss}")

    # Entrenamiento con texto
    text_optimizer = AdamW(model.parameters(), lr=1e-4)
    
    with tqdm(total=text_train_steps) as pbar:
        for step in range(text_train_steps):
            model.train()
            data = next(train_iter)
            loss = model(data.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            text_optimizer.step()
            text_optimizer.zero_grad()
            pbar.set_description(f'text loss: {loss.item():.3f}')
            pbar.update()

            if (step + 1) % 500 == 0:
                save_checkpoint(model, step, loss.item(), 'text')
                
    # Guardar el modelo final
    save_checkpoint(model, text_train_steps, loss.item(), 'final')

if __name__ == "__main__":
    main()