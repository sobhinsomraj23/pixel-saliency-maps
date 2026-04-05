import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_NAME = "resnet50"
PRETRAINED = True

# Image configuration
IMAGE_SIZE = 224

# Visualization
COLORMAP = "hot"