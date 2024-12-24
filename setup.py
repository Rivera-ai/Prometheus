from setuptools import setup, find_packages

setup(
    name="PrometheusCore",
    version="0.1.1",
    description="Prometheus is a native multimodal model and Prometheus is based on the paper: Transfusion",
    author="Rivera.ai/Fredy Rivera",
    author_email="riveraaai200678@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "av",
        "tqdm",
        "torchmetrics",
        "einops",
        "einx",
        "ema_pytorch",
        "rotary_embedding_torch",
        "tqdm",
        "jaxtyping",
        "beartype",
        "loguru",
    ],
    python_requires=">=3.8",
)
