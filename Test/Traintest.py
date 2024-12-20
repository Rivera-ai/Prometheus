from shutil import rmtree
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
from Model.PrometheusModel import Prometheus, print_modality_sample

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rmtree('./results', ignore_errors = True)
results_folder = Path('./results')
results_folder.mkdir(exist_ok = True, parents = True)

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
    
    # También guardar como último checkpoint
    latest_checkpoint = checkpoint_path / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_checkpoint)

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

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data/mnist',
            download = True
        )   

        self.transform = T.Compose([
            T.PILToTensor(),
            T.RandomResizedCrop((28, 28), scale = (0.8, 1.))
        ])

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = self.transform(pil)
        return tensor(labels), (digit_tensor / 255).float()

dataset = MnistDataset()

autoencoder_train_steps = 1500
dim_latent = 16

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)

encoder = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding = 1),
    nn.Conv2d(4, 8, 4, 2, 1),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Conv2d(8, dim_latent, 1),
    Rearrange('b d ... -> b ... d'),
    Normalize()
).to(device=device)

decoder = nn.Sequential(
    Rearrange('b ... d -> b d ...'),
    nn.Conv2d(dim_latent, 8, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, 2, 1),
    nn.Conv2d(4, 1, 3, padding = 1),
).to(device=device)

autoencoder_optimizer = AdamW([*encoder.parameters(), *decoder.parameters()], lr = 3e-4)
autoencoder_dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)

autoencoder_iter_dl = cycle(autoencoder_dataloader)

print('training autoencoder')

with tqdm(total = autoencoder_train_steps) as pbar:
    for _ in range(autoencoder_train_steps):
        _, images = next(autoencoder_iter_dl)
        images = images.to(device)

        latents = encoder(images)
        latents = latents.lerp(torch.randn_like(latents), torch.rand_like(latents) * 0.2) # add a bit of noise to latents
        reconstructed = decoder(latents)

        loss = F.mse_loss(images, reconstructed)

        loss.backward()

        pbar.set_description(f'loss: {loss.item():.5f}')

        autoencoder_optimizer.step()
        autoencoder_optimizer.zero_grad()

        pbar.update()

model = Prometheus(
    num_text_tokens = 10,
    dim_latent = dim_latent,
    modality_default_shape = (14, 14),
    modality_encoder = encoder,
    modality_decoder = decoder,
    add_pos_emb = True,
    modality_num_dim = 2,
    transformer = dict(
        dim = 64,
        depth = 4,
        dim_head = 32,
        heads = 8,
    )
).to(device=device)

dataloader = model.create_dataloader(dataset, batch_size = 16, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters_without_encoder_decoder(), lr = 3e-4)

transfusion_train_steps = 2500

print('training transfusion with autoencoder')

with tqdm(total = transfusion_train_steps) as pbar:
    for index in range(transfusion_train_steps):
        step = index + 1

        model.train()

        loss = model(next(iter_dl))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        pbar.set_description(f'loss: {loss.item():.3f}')

        pbar.update()

        # eval

        if divisible_by(step, 500):
            save_checkpoint(
                model=model,
                epoch=step,
                loss=loss.item()
            )
            one_multimodal_sample = model.sample(max_length = 10)

            print_modality_sample(one_multimodal_sample)

            if len(one_multimodal_sample) < 2:
                continue

            maybe_label, maybe_image, *_ = one_multimodal_sample

            filename = f'{step}.{maybe_label[1].item()}.png'

            save_image(
                maybe_image[1].cpu().clamp(min = 0., max = 1.),
                str(results_folder / filename),
            )