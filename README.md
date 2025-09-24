# Fossil Interpretability

This repo demonstrates **machine learning interpretability** applied to fossil image classification.  
The goal is to make ML models more transparent and to explore whether models focus on the same features paleontologists use (e.g., shell ridges, suture lines, spines).

---

## Why Interpretability?
Interpretability is crucial in scientific domains:
- **Trust**: Verify that models are not making predictions based on spurious features (e.g., background, cracks in rock).  
- **Discovery**: Highlight morphological features that may be scientifically relevant.  
- **Education**: Provide intuitive visualizations for how deep learning models "see" fossils.  

---

## Methods
We implement several common interpretability techniques:
- **Grad-CAM**: Highlights class-discriminative regions of images.  
- **Saliency Maps (Vanilla + SmoothGrad)**: Pixel-level sensitivity maps.  
- **Occlusion Sensitivity**: Mask regions to test prediction changes.  
- **Integrated Gradients**: Attributions along a path from baseline to input.  

Each method overlays a **heatmap** on fossil images to show what the model uses for classification.

---

## Repository Structure

```
fossil-interpretability/
│
├── notebooks/
│ ├── 01_train_classifier.ipynb # Train/fine-tune CNN on fossil images
│ ├── 02_gradcam.ipynb # Grad-CAM visualizations
│ ├── 03_occlusion.ipynb # Occlusion sensitivity
│ └── 04_compare_methods.ipynb # Side-by-side interpretability comparison
│
├── src/
│ ├── data.py # dataset loader
│ ├── models.py # model definitions (ResNet, etc.)
│ ├── interpret.py # interpretability utilities
│
├── README.md
└── requirements.txt
```


---

## Quickstart
1. Clone this repo  
   ```bash
   git clone https://github.com/your-username/computer-vision-interpretability-paleontology-v0
   cd fossil-interpretability
   ```

2. Install requirements
```
pip install -r requirements.txt
```

3. Run the first notebook to train a classifier:
```
jupyter notebook notebooks/01_train_classifier.ipynb
```
