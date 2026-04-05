import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from models.model_loader import load_model
from utils.image_utils import load_image, preprocess_image
from saliency.vanilla_saliency import compute_vanilla_saliency
from saliency.guided_backprop import GuidedBackprop
from utils.visualization import normalize_saliency

import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = load_model()


# -------- Save Heatmap --------
def save_saliency_map(saliency, filename):
    path = os.path.join(RESULT_FOLDER, filename)
    plt.imsave(path, saliency, cmap="hot")
    return filename


# -------- Save Overlay --------
def save_overlay(original_image, saliency, filename):
    saliency = normalize_saliency(saliency)

    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.imshow(saliency, cmap="jet", alpha=0.5)
    plt.axis("off")

    path = os.path.join(RESULT_FOLDER, filename)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return filename


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("image")

        if file:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # Load + preprocess
            image = load_image(upload_path)
            tensor = preprocess_image(image)

            # -------- Vanilla --------
            vanilla_saliency, _ = compute_vanilla_saliency(model, tensor)
            vanilla_saliency = normalize_saliency(vanilla_saliency)

            # Reset gradients
            tensor.grad = None

            # -------- Guided --------
            guided_bp = GuidedBackprop(model)
            guided_saliency, _ = guided_bp.generate_saliency(tensor)
            guided_saliency = normalize_saliency(guided_saliency)

            # -------- Save outputs --------
            vanilla_file = save_saliency_map(vanilla_saliency, "vanilla.png")
            guided_file = save_saliency_map(guided_saliency, "guided.png")

            overlay_vanilla = save_overlay(image, vanilla_saliency, "overlay_vanilla.png")
            overlay_guided = save_overlay(image, guided_saliency, "overlay_guided.png")

            return render_template(
                "index.html",
                original=f"uploads/{filename}",
                vanilla=f"results/{vanilla_file}",
                guided=f"results/{guided_file}",
                vanilla_overlay=f"results/{overlay_vanilla}",
                guided_overlay=f"results/{overlay_guided}",
            )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)