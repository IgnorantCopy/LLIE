import os
import numpy as np
import lightning as pl
import torchvision.transforms as transforms
from PIL import Image
import plotly.graph_objects as go

from src.llie.utils.config import load_config, get_model


def visualize_image(image: np.ndarray):
    h, w = image.shape[1], image.shape[2]
    fig = go.Figure(data=[go.Image(z=image)])
    fig.update_layout(height=h, width=w)
    fig.show()


def main():
    ckpt_path = "logs/retinex_net/2025-09-08_11-26-58/lightning_logs/version_0/checkpoints/epoch=199-step=19400.ckpt"
    config_path = os.path.join(os.path.dirname(ckpt_path), "../../../config.yaml")
    config = load_config(config_path)
    model_config, train_config, data_config = config["model"], config["train"], config["data"]
    pl.seed_everything(getattr(data_config, "seed", 42))

    image = Image.open("/home/nju-student/mkh/datasets/LLIE/LOLv1/eval15/low/1.png")
    transform = transforms.ToTensor()
    image = transform(image).to("cuda")
    image = image.unsqueeze(0)

    model = get_model(config)
    model.to("cuda")
    high = model.predict_step(image).squeeze(0).detach().cpu().numpy()
    visualize_image(high)


if __name__ == '__main__':
    main()