import torch
from pathlib import Path
from torchvision.utils import save_image
from Model.PrometheusModel import Prometheus
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn import Module
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim = -1)


dim_latent = 16

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


def load_model_for_inference(checkpoint_path):
    # Crear el modelo con la misma configuración que al entrenar
    model = Prometheus(
        num_text_tokens = 10,
        dim_latent = 16,
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
    ).to(device)
    
    # Cargar los pesos guardados
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def generate_from_number(model, number, save_dir='generated'):
    """Genera una imagen a partir de un número"""
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    with torch.no_grad():
        # Convertir número a tensor
        number_tensor = torch.tensor([number]).to(device)
        
        # Generar muestra
        sample = model.sample(max_length=10)
        
        if len(sample) >= 2:
            _, maybe_image, *_ = sample
            
            # Guardar imagen
            save_image(
                maybe_image[1].cpu().clamp(min=0., max=1.),
                f'{save_dir}/generated_{number}.png'
            )
            return maybe_image[1]
        return None

def batch_generate(model, numbers=[0,1,2,3,4,5,6,7,8,9], save_dir='generated'):
    """Genera imágenes para múltiples números"""
    for num in numbers:
        print(f"Generando imagen para número {num}")
        generate_from_number(model, num, save_dir)

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar modelo
    model = load_model_for_inference('checkpoints/checkpoint_latest.pt')
    
    # Método 1: Generar una imagen para un número específico
    generate_from_number(model, 7)
    
    # Método 2: Generar imágenes para varios números
    batch_generate(model)
    
    # Método 3: Inferencia interactiva
    while True:
        try:
            number = int(input("Ingresa un número (0-9) o -1 para salir: "))
            if number == -1:
                break
            if 0 <= number <= 9:
                generate_from_number(model, number)
            else:
                print("Por favor ingresa un número entre 0 y 9")
        except ValueError:
            print("Por favor ingresa un número válido")