from shutil import rmtree
from pathlib import Path
import random
import torch
from torch import nn, tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image
import pandas as pd
import ast
from PrometheusCore import Prometheus, print_modality_sample, EncoderV1, DecoderV1
import numpy as np
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup directories
rmtree('./results', ignore_errors=True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok=True, parents=True)

# Checkpoint functions remain the same
def save_checkpoint(model, epoch, loss, checkpoint_dir='checkpoints'):
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss
    }
    
    checkpoint_file = checkpoint_path / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_file)
    latest_checkpoint = checkpoint_path / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_checkpoint)

def collate_fn(batch):
    # Separar las captions y las imágenes
    captions, images = zip(*batch)

    # Aplicar padding a las captions
    padded_captions = pad_sequence(captions, batch_first=True, padding_value=0)

    # Convertir imágenes en un solo tensor (PyTorch lo maneja bien si tienen tamaño uniforme)
    images = torch.stack(images)

    return padded_captions, images

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def divisible_by(num, den):
    return (num % den) == 0

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

class Flickr30kDataset(Dataset):
    def __init__(self, csv_path, images_dir):
        """
        Args:
            csv_path: Ruta al archivo CSV con los datos de Flickr30k
            images_dir: Directorio que contiene las imágenes
        """
        self.images_dir = Path(images_dir)
        
        # Cargar y procesar el CSV
        self.df = pd.read_csv(csv_path)
        # Convertir las strings de lista a listas reales
        self.df['raw'] = self.df['raw'].apply(ast.literal_eval)
        
        # Filtrar por split=='train' si es necesario
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        
        self.transform = T.Compose([
            T.Resize(286),  # Resize más grande para crop
            T.RandomCrop(256),  # Random crop para data augmentation
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Cargar imagen
        image_path = self.images_dir / row['filename']
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image)
        
        # Seleccionar una caption aleatoria de las 5 disponibles
        caption = random.choice(row['raw'])
        caption = np.frombuffer(caption.encode('utf-8'), dtype=np.uint8)
        caption = tensor(caption, dtype=torch.long)
        
        return caption, image_tensor

# Larger encoder for 256x256 images
dim_latent = 256  # Increased latent dimension for more complex images

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

encoder = EncoderV1(dim_latent=dim_latent)

# Decoder modificado para manejar las dimensiones correctamente
decoder = DecoderV1(dim_latent=dim_latent)

# Create dataset and dataloaders
dataset = Flickr30kDataset(csv_path="/teamspace/studios/this_studio/flickr30k/flickr_annotations_30k.csv", images_dir="/teamspace/studios/this_studio/flickr30k/flickr30k-images/")

# Training parameters
batch_size = 8  # Reduced batch size due to larger images
accum_steps = 4
autoencoder_train_steps = 500  # Increased steps for more complex dataset
learning_rate = 1e-4  # Adjusted learning rate

autoencoder_optimizer = AdamW([*encoder.parameters(), *decoder.parameters()], lr=learning_rate)
autoencoder_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
autoencoder_iter_dl = cycle(autoencoder_dataloader)

print('training autoencoder')

with tqdm(total=autoencoder_train_steps) as pbar:
    for step in range(autoencoder_train_steps):
        total_loss = 0
        autoencoder_optimizer.zero_grad()
        
        for _ in range(accum_steps):
            _, images = next(autoencoder_iter_dl)
            images = images.to(device)

            latents = encoder(images)
            latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.1)
            #print("Latents shape:", latents.shape)
            reconstructed = decoder(latents)

            loss = F.mse_loss(images, reconstructed)
            loss = loss / accum_steps  # Normalizar la pérdida
            total_loss += loss.item()
            
            loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 1.0)
        
        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()

        pbar.set_description(f'loss: {total_loss:.5f}')
        pbar.update()

        # Guardar muestras periódicamente
        if step % 500 == 0:
            with torch.no_grad():
                save_image(
                    reconstructed[0].cpu().clamp(min=-1., max=1.),
                    str(results_folder / f'reconstruction_{step}.png'),
                    normalize=True
                )

# Initialize Prometheus with new parameters
model = Prometheus(
    num_text_tokens=256,  # Increased for text vocabulary
    dim_latent=dim_latent,
    modality_default_shape=(16, 16),  # Adjusted for 256x256 images
    modality_encoder=encoder,
    modality_decoder=decoder,
    add_pos_emb=True,
    modality_num_dim=2,
    transformer=dict(
        dim=256,  # Increased transformer dimensions
        depth=8,  # More layers
        dim_head=64,
        heads=8,
    )
).to(device)

dataloader = model.create_dataloader(dataset, batch_size=batch_size, shuffle=True)
iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters_without_encoder_decoder(), lr=learning_rate)
transfusion_train_steps = 100000  # Increased steps

print('training transfusion with autoencoder')

with tqdm(total=transfusion_train_steps) as pbar:
    for index in range(transfusion_train_steps):
        step = index + 1
        model.train()

        data = next(iter_dl)
        #print(f"Data from dataloader: {data}")
        loss = model(data)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Increased grad clip

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'loss: {loss.item():.3f}')
        pbar.update()

        if divisible_by(step, 5):
            save_checkpoint(model=model, epoch=step, loss=loss.item())
            one_multimodal_sample = model.sample(max_length=256)  # Increased for longer captions

            print_modality_sample(one_multimodal_sample)

            if len(one_multimodal_sample) < 2:
                continue

            caption, image, *_ = one_multimodal_sample

            filename = f'{step}.png'
            save_image(
                image[1].cpu().clamp(min=-1., max=1.),
                str(results_folder / filename),
                normalize=True
            )