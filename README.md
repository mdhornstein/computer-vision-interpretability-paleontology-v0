## Loading the TF 1.13 weights and saving them in a modern format. 

This section describes how to load pre-trained TensorFlow 1.13 model weights and convert them to a format compatible with modern TensorFlow versions. The process uses a Docker container to manage TF 1.13 dependencies and ensure compatibility.

### Prerequisites

- Docker and Docker Compose installed
- Model checkpoint files from TF 1.13 training
- Apple Silicon compatibility (uses community TensorFlow build without AVX requirements)

### Setup Instructions

#### 1. Download TensorFlow 1.13 Community Build

From the root directory of the repository:
```bash
./download_wheel_v1.sh
```
This script downloads a community-built TensorFlow 1.13.1 wheel that works on Apple Silicon without AVX instruction requirements.


#### 2. Build the Docker Image

```bash
./docker-build.sh
```
The Docker image includes:

* Python 3.6 (compatible with TF 1.13)
* Community TensorFlow 1.13.1 wheel (downloaded in step 1)
* TensorFlow Slim models from the r1.13.0 tag (cloned from GitHub in the Dockerfile) 
*  Required dependencies (h5py, numpy, etc.)

#### 3. Start the Container

```bash
./docker-up.sh -d
```
This runs the container in detached mode, keeping it running in the background.

#### 4. Access the Container
```bash
docker compose exec tf1 bash
```

### Running the Conversion

#### 1. Input Requirements
The script expects your TensorFlow 1.13 checkpoint files to be located at:

```
models/fid_classification/tf1/
├── model.ckpt-60.data-00000-of-00001
├── model.ckpt-60.index
└── model.ckpt-60.meta
```

#### 2. Run the Conversion Script 
Once inside the container, run the conversion script:
```bash
python notebooks/03_convert_weights_to_tf2.py
```

#### 3. Output 
The conversion script creates a modern TensorFlow SavedModel at: 
```
models/fid_classification/tf1_from_community_allow_train/1/
├── saved_model.pb
└── variables/
    ├── variables.data-00000-of-00001
    └── variables.index
```


We run TF 1.13 in a Docker image to pull together all dependencies automatically. Here are the steps to build and run the image. From the root directory of the repo, 

### Donwload a community build of TensorFlow 1.13 

1. Run `./download_wheel_v1.sh` to download the wheel with the community build, which will be used by the Docker container. 
1. Run `./docker-build.sh`
1. Run `./docker-up.sh -d` to run the image in a deatched way.
1. Run `docker compose exec tf1 bash` to conneect to the running detached image.

Run the following
```
python notebooks/03_convert_weights_to_tf2.py
``` 
This script assumes that the model is in `models/fid_classification/tf1`, in the following form: 
```
model.ckpt-60.data-00000-of-00001  model.ckpt-60.index	model.ckpt-60.meta
```

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
   cd computer-vision-interpretability-paleontology-v0
   ```

2. Install requirements
```
pip install -r requirements.txt
```

3. Run the first notebook to train a classifier:
```
jupyter notebook notebooks/01_train_classifier.ipynb
```
