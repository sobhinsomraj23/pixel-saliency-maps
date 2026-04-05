import argparse

from models.model_loader import load_model
from utils.image_utils import load_image, preprocess_image
from utils.visualization import plot_saliency, overlay_saliency
from saliency.vanilla_saliency import compute_vanilla_saliency
from saliency.guided_backprop import GuidedBackprop


def main(image_path):
    # Load model
    model = load_model()

    # Load and preprocess image
    original_image = load_image(image_path)
    image_tensor = preprocess_image(original_image)

    # Vanilla Saliency
    vanilla_saliency, pred_class_vanilla = compute_vanilla_saliency(model, image_tensor)

    # Guided Backprop
    # Important: reset gradients
    image_tensor.grad = None

    guided_bp = GuidedBackprop(model)
    guided_saliency, pred_class_guided = guided_bp.generate_saliency(image_tensor)

    # Visualization
    print(f"Predicted Class (Vanilla): {pred_class_vanilla}")
    print(f"Predicted Class (Guided): {pred_class_guided}")

    plot_saliency(original_image, vanilla_saliency, title="Vanilla Saliency")
    plot_saliency(original_image, guided_saliency, title="Guided Backprop")

    overlay_saliency(original_image, vanilla_saliency)
    overlay_saliency(original_image, guided_saliency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel Saliency Maps via Input Gradients")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")

    args = parser.parse_args()

    main(args.image)