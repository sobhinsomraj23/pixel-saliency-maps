import torch
import torchvision.transforms as transforms
from PIL import Image
from config import IMAGE_SIZE, DEVICE

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0) #shape :(1,3,H,W)
    image_tensor = image_tensor.to(DEVICE)
    image_tensor.requires_grad = True #Enable gradient tracking
    return image_tensor
