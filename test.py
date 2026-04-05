import torch
import torchvision.models as models
from config import DEVICE, MODEL_NAME, PRETRAINED


def load_model():
    """
    Loads a pretrained model and prepares it for saliency computation.
    """

    if MODEL_NAME == "resnet50":
        model = models.resnet50(pretrained=PRETRAINED)
    else:
        raise ValueError(f"Model {MODEL_NAME} not supported yet.")

    # Move model to device
    model = model.to(DEVICE)

    # Set to evaluation mode
    model.eval()

    return model