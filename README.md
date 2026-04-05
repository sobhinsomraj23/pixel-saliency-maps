# Pixel Saliency Maps via Input Gradient Backpropagation

## Overview
This project focuses on understanding and interpreting deep learning image classification models by analyzing pixel-level importance using gradient-based methods.
Given an input image and a pretrained convolutional neural network, the system computes the gradient of the predicted class score with respect to each input pixel. This allows us to identify which regions of the image most influenced the model's decision.
The project also compares two approaches:
- Vanilla Saliency Maps
- Guided Backpropagation

The goal is to move beyond prediction and enable explainability in computer vision models.

---

## Key Concepts

- Backpropagation through input tensors
- Gradient-based attribution
- Pixel-level importance estimation
- ReLU-based gradient filtering
- Model interpretability

---

## Methodology

1. Load a pretrained image classification model (ResNet50).
2. Preprocess the input image to match training distribution.
3. Perform a forward pass to obtain class scores.
4. Select the predicted class.
5. Compute gradients of the class score with respect to input pixels.
6. Generate saliency maps:
   - Vanilla: raw gradients
   - Guided Backpropagation: filtered gradients using modified ReLU backward pass
7. Normalize and visualize results.
8. Overlay saliency maps on the original image for interpretability.

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PIL
- Flask (for web interface)
- HTML and CSS

---

## Setup Instructions

1. Clone the repository

   git clone https://github.com/sobhinsomraj23/pixel-saliency-maps.git
   cd pixel-saliency-maps

2. Create and activate virtual environment

   python -m venv venv
   venv\Scripts\activate    (Windows)

3. Install dependencies

   pip install -r requirements.txt

---

## Running the Project

### Option 1: Command Line

Run saliency computation directly:

   python main.py --image path_to_image.jpg

This will generate and display:
- Original image
- Vanilla saliency map
- Guided backpropagation map
- Overlay visualizations

---

### Option 2: Web Interface

Start the Flask application:

   python app.py

Open browser and go to:

   http://127.0.0.1:5000

Upload an image to view:
- Original image
- Vanilla saliency
- Guided backpropagation
- Overlay visualizations

---

## Output Explanation

- Vanilla Saliency
  Shows raw gradient magnitudes. Often noisy and edge-focused.

- Guided Backpropagation
  Filters gradients through ReLU, producing sharper and more meaningful visualizations.

- Overlay Maps
  Highlight important regions directly on the input image.

---

## Observations

- Vanilla saliency tends to highlight edges and high-frequency patterns.
- Guided backpropagation produces cleaner and more interpretable maps.
- The model may sometimes focus on background features, revealing dataset bias.

---

## Applications

- Model debugging
- Explainable AI systems
- Medical image analysis
- Autonomous systems validation
- Bias detection in datasets

---

## Limitations

- Gradient-based methods can be noisy
- Sensitive to small input perturbations
- Do not always capture semantic understanding
- Only reflects local sensitivity

---

## Future Improvements

- Add Grad-CAM for spatial localization
- Implement SmoothGrad for noise reduction
- Support multiple model architectures
- Add class label mapping (ImageNet labels)
- Build interactive comparison tools
- Deploy as a web application

---

## Conclusion

This project demonstrates how gradient-based techniques can be used to interpret deep learning models at the pixel level. It bridges the gap between prediction and explanation, enabling a deeper understanding of model behavior.