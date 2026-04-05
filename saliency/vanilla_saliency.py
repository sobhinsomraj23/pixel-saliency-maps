import torch

def compute_vanilla_saliency(model, image_tensor):
    # Forward pass
    output = model(image_tensor) #shape : (1,1000)

    # get predicted class
    predicted_class = output.argmax(dim=1)

    # zero out previous gradients
    model.zero_grad()

    # Backward pass for the predicted class score
    output[0, predicted_class].backward()

    # Get gradients w.r.t input image
    gradients = image_tensor.grad.data #shape : (1,3,H,W)

    # Take absolute value
    saliency, _ = torch.max(gradients.abs(), dim=1) #Shape: (1,H,W)

    return saliency.squeeze().cpu().numpy(), predicted_class.item()