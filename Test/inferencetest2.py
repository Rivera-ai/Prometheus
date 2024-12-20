from pathlib import Path
import torch
from torch import tensor
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from Model.PrometheusModel import Prometheus
import numpy as np
import torch.nn as nn
from torch.nn import Module
from einops.layers.torch import Rearrange

dim_latent = 16
class InferenceConfig:
    dim_latent = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path('./checkpoints/checkpoint_text_epoch_9999.pt')

class Normalize(Module):
    def forward(self, x):
        return F.normalize(x, dim=-1)

def load_trained_model():
    """
    Carga el modelo Prometheus entrenado desde el último checkpoint.
    """
    encoder = nn.Sequential(
    nn.Conv2d(1, 4, 3, padding=1),
    nn.Conv2d(4, 8, 4, 2, 1),
    nn.ReLU(),
    nn.Dropout(0.05),
    nn.Conv2d(8, dim_latent, 1),
    Rearrange('b d ... -> b ... d'),
    Normalize()).to(InferenceConfig.device)

    decoder = nn.Sequential(
    Rearrange('b ... d -> b d ...'),
    nn.Conv2d(dim_latent, 8, 1),
    nn.ReLU(),
    nn.ConvTranspose2d(8, 4, 4, 2, 1),
    nn.Conv2d(4, 1, 3, padding=1),
    ).to(InferenceConfig.device)

    model = Prometheus(
        num_text_tokens=256,
        dim_latent=InferenceConfig.dim_latent,
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
    ).to(InferenceConfig.device)

    checkpoint = torch.load(InferenceConfig.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def text_to_text(model, input_text, seq_len=100, temperature=1.5, min_p=0.1):
    """
    Genera texto a partir de texto de entrada usando generate_text_only.
    
    Args:
        model (Prometheus): Modelo cargado
        input_text (str): Texto de entrada
        seq_len (int): Longitud máxima del texto a generar
        temperature (float): Temperatura para el muestreo
        min_p (float): Filtro de probabilidad mínima
    Returns:
        str: Texto generado
    """
    # Convertir texto a bytes
    bytes_array = torch.tensor(
        [ord(c) for c in input_text], 
        dtype=torch.long,
        device=InferenceConfig.device
    ).unsqueeze(0)  # Agregar dimensión de batch
    
    with torch.no_grad():
        generated = model.generate_text_only(
            bytes_array,
            seq_len=seq_len,
            temperature=temperature,
            min_p=min_p
        )
        
        # Convertir bytes a texto
        generated_text = ''.join([chr(b) for b in generated.cpu().numpy().flatten()])
        return generated_text

def number_to_image(model, digit, save_path='generated_digit.png'):
    """
    Genera una imagen a partir de un número usando sample.
    
    Args:
        model (Prometheus): Modelo cargado
        digit (int): Dígito a generar (0-9)
        save_path (str): Ruta donde guardar la imagen generada
    """
    assert 0 <= digit <= 9, "El dígito debe estar entre 0 y 9"
    
    # Crear el tensor del dígito
    digit_tensor = torch.tensor([digit], dtype=torch.long, device=InferenceConfig.device)
    
    with torch.no_grad():
        # Usamos sample con el dígito como prompt
        modality_samples = model.sample(
            prompt=[digit_tensor],
            max_length=1000,  # Suficientemente largo para generar la imagen
            modality_steps=16,
            fixed_modality_shape=(28, 28)
        )
        
        # Extraemos la imagen generada del resultado
        for sample in modality_samples:
            if isinstance(sample, tuple) and sample[0] == 0:  # modalidad 0 es imagen
                generated_image = sample[1]
                torchvision.utils.save_image(generated_image[0], save_path)
                return generated_image[0]
        
        raise ValueError("No se encontró la imagen generada en los resultados")

def main():
    # Cargar el modelo
    print("Cargando modelo...")
    model = load_trained_model()
    
    # Ejemplo de texto a texto
    input_text = "Este es un texto de ejemplo"
    print("\nGenerando texto a partir de:", input_text)
    generated_text = text_to_text(model, input_text)
    print("Texto generado:", generated_text)
    
    # Ejemplo de número a imagen
    digit = 9
    print(f"\nGenerando imagen para el dígito {digit}...")
    number_to_image(model, digit, f'digit_{digit}.png')
    print(f"Imagen guardada como 'digit_{digit}.png'")

if __name__ == "__main__":
    main()