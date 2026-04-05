import torch
import torch.nn as nn

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_outputs =[]

    # Register forward and backward hooks on ReLU layers
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.forward_outputs.append(output)

        # Modify gradient flow by only allowing positive gradients where forward activation is +ve
        def backward_hook(module, input, output):
            forward_output = self.forward_outputs.pop()
            positive_mask = (forward_output > 0).float()
            grad_output = grad_output[0]

            modified_grad = positive_mask * torch.clamp(grad_output, min=0.0)
            return (modified_grad,)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)

    # Generate guided backpropogation saliency map
    def generate_saliency(self, image_tensor):
        # Forward pass
        output = self.model(image_tensor)

        # Predicted class
        predicted_class = output.argmax(dim=1)

        # zero gradients
        self.model.zero_grad()

        # Backward pass
        output[0, predicted_class].backward()

        # Get gradients
        gradients = image_tensor.grad.data

        # Process gradients
        saliency, _ = torch.max(gradients.abs(), dim=1)

        return saliency.squeeze().cpu().numpy(), predicted_class.item()
