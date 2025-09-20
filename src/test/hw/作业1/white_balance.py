import torch
from torchvision.transforms import ToTensor
from PIL import Image


def preprocess(image: torch.Tensor):
    r, g, b = image.split(1, dim=0)
    x = torch.cat([r, g, b, r * g, r * b, g * b, r * r, g * g, b * b, r * g * b, torch.ones_like(r)], dim=0)
    return x.flatten(1)


def train():
    input_image = ToTensor()(Image.open('input.png').convert('RGB'))
    x = preprocess(input_image)

    output_image = ToTensor()(Image.open('output.png').convert('RGB'))
    y = output_image.flatten(1)

    w = torch.linalg.inv(x @ x.T) @ x @ y.T
    return w


def test(weight: torch.Tensor):
    test_image = ToTensor()(Image.open('test.png').convert('RGB'))
    _, h, w = test_image.shape
    x = preprocess(test_image)
    output_image = weight.T @ x
    output_image = output_image.reshape(-1, h, w)
    output_image = torch.clip(output_image, 0, 1) * 255
    return output_image.to(torch.uint8)


if __name__ == '__main__':
    result = test(train())
    result = Image.fromarray(result.permute(1, 2, 0).numpy())
    result.save('./result.png')