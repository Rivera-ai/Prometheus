import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL 
from transformers import EncodecModel
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import logging
from Prometheus.Model.PrometheusModel import Prometheus
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultimodalDataset(Dataset):
    def __init__(self, max_length=128):
        logger.info("Inicializando datasets...")
        self.text_dataset = load_dataset("stas/openwebtext-10k", split="train")
        self.audio_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", split="validation")
        self.image_dataset = load_dataset("nlphuji/flickr30k", split="test").select(range(100))
        
        logger.info("Inicializando modelos...")
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.encodec = EncodecModel.from_pretrained('facebook/encodec_24khz')
        
        self.image_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.max_length = max_length
        self.length = max(len(self.text_dataset), len(self.audio_dataset), len(self.image_dataset))
        logger.info(f"Dataset inicializado con {self.length} muestras")

    def __len__(self):
        return self.length

    def get_modality_sample(self, idx):
        modalities = []

        # Texto
        text_item = self.text_dataset[idx % len(self.text_dataset)]
        text_tokens = self.tokenizer(
            text_item['text'],
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.squeeze()
        modalities.append(text_tokens)

        # Audio 
        audio_item = self.audio_dataset[idx % len(self.audio_dataset)]
        waveform = torch.tensor(audio_item['audio']['array'])
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            audio_latents = self.encodec.encode(waveform.unsqueeze(0))[0][0]
        modalities.append((1, audio_latents)) # tipo 1 para audio

        # Imagen
        image_item = self.image_dataset[idx % len(self.image_dataset)]
        image = self.image_transform(image_item['image'].convert('RGB'))
        with torch.no_grad():
            image_latents = self.vae.encode(image.unsqueeze(0)).latent_dist.sample()
        modalities.append((0, image_latents.squeeze())) # tipo 0 para imagen

        return modalities

    def __getitem__(self, idx):
        return self.get_modality_sample(idx)

class EncoderAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodec = EncodecModel.from_pretrained('facebook/encodec_24khz')
        
    def forward(self, x):
        return self.encodec.encode(x)[0][0]

class DecoderAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.encodec = EncodecModel.from_pretrained('facebook/encodec_24khz')
        
    def forward(self, x):
        return self.encodec.decode([(x, None)])

class PrometheusTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
        # Inicializar Prometheus
        logger.info("Inicializando modelo Prometheus...")
        # En la clase PrometheusTrainer, modificamos la inicialización del modelo:

        self.model = Prometheus(
            num_text_tokens=32128,  # Tamaño del vocabulario de Llama
            transformer=dict(
            dim=1024,
            depth=12,
            heads=16
            ),
            dim_latent=(768, 1024, 512),  # dimensiones para texto, audio e imagen
            channel_first_latent=(False, True, True),  # configuración para cada modalidad
            add_pos_emb=(True, True, True),
            modality_encoder=( 
                None,  # para texto
                EncoderAudio(),  # para audio
                AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").encode  # para imagen
                ),
            modality_decoder=(
                None,  # para texto
                DecoderAudio(),  # para audio
                AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").decode  # para imagen
            ),
            modality_default_shape=(
                (128,),  # texto - longitud máxima de secuencia
                (1, 16000),  # audio 
                (3, 256, 256)  # imagen
            ),
            fallback_to_default_shape_if_invalid=True,
            modality_num_dim=(1, 2, 3),  # dimensiones para cada modalidad
            text_loss_weight=1.0,
            flow_loss_weight=1.0,
            velocity_consistency_loss_weight=0.1,
            reconstruction_loss_weight=0.2,
            modality_encoder_decoder_requires_batch_dim=True
            ).to(self.device)
        
        # Dataset y dataloader
        self.dataset = MultimodalDataset()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        # Optimizador
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.dataloader, 
            desc="Training",
            leave=True,
            dynamic_ncols=True
        )
        
        for batch_idx, modalities in enumerate(progress_bar):
            self.optimizer.zero_grad()

            # Forward pass
            loss = self.model(
                modalities=modalities,
                return_loss=True
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Actualizar métricas
            total_loss += loss.item()
            progress_bar.set_postfix(
                {'loss': f"{loss.item():.4f}"}
            )

        return total_loss / len(self.dataloader)

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }
        
        if is_best:
            path = f'prometheus_best.pt'
        else:
            path = f'prometheus_epoch_{epoch+1}.pt'
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint guardado en {path}")

    def train(self):
        logger.info("Iniciando entrenamiento...")
        best_loss = float('inf')

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            avg_loss = self.train_epoch()
            
            # Guardar checkpoints
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(epoch, avg_loss, is_best=True)
            
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch, avg_loss)
            
            logger.info(f"Epoch {epoch+1} completada. Pérdida promedio: {avg_loss:.4f}")

        logger.info("¡Entrenamiento completado!")

def main():
    class Config:
        batch_size = 8
        num_workers = 4
        learning_rate = 1e-4
        num_epochs = 10
        save_every = 2

    trainer = PrometheusTrainer(Config())
    trainer.train()

if __name__ == "__main__":
    main()