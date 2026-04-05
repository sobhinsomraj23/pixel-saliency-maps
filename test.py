from PIL import Image
import torch
import torchvision.transforms as transforms
from config import IMAGE_SIZE, DEVICE


def load_image(image_path):
    """
    Loads an image from disk.
    """
    image = Image.open(image_path).convert("RGB")
    return image


def preprocess_image(image):
    """
    Preprocess the image to match model requirements.
    Also enables gradient tracking on input.
    """

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)  # Shape: (1, 3, H, W)

    # Move to device
    image_tensor = image_tensor.to(DEVICE)

    # 🔥 VERY IMPORTANT: Enable gradient tracking
    image_tensor.requires_grad = True

    return image_tensor