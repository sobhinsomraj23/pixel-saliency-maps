import torch
import torchvision.models as models
from config import DEVICE, MODEL_NAME, PRETRAINED

# loads a pretrained model and prepares it for saliency computation
def load_model():
    if MODEL_NAME == "resnet50":
        model = models.resnet50(pretrained=PRETRAINED)
    else:
        raise ValueError(f"Model {MODEL_NAME} not supported yet")
    
    model = model.to(DEVICE)

    model.eval()

    return model