# PLAN.md  

## 📌 Project Goal  
Build a minimal but useful repository for **machine learning interpretability** on fossil images.  
- Dataset: public fossil image collection (e.g., FID-reduced fossil dataset).  
- Model: simple convolutional neural network (CNN).  
- Interpretability: Grad-CAM, saliency maps, occlusion sensitivity, integrated gradients.  
- Evaluation: compare interpretability maps against paleontological features.  

## 

---

## 🚀 Workflow  

### Step 1: Clone Repo in Colab  
- Always start a new Colab session by cloning the repo:  

```bash
%cd /content
!git clone https://github.com/your-username/fossil-interpretability.git
%cd fossil-interpretability
```

- Then open notebooks directly from GitHub tab in Colab (not from “Recent” or “Drive”).  

---

### Step 2: Prepare Dataset  
Notebook: `notebooks/00_prepare_dataset.ipynb`  

- Download fossil dataset (e.g., `reduced-FID/`).  
- Split into `data/train` (80%) and `data/val` (20%).  
- Verify directory structure:  

```
data/
  train/
    ammonoid/
    brachiopod/
    ...
  val/
    ammonoid/
    brachiopod/
    ...
```

- Add `.gitignore` rule so that `data/` is never committed.  

---

### Step 3: Train a Baseline Model  
Notebook: `notebooks/01_train_model.ipynb`  

- Use PyTorch `ImageFolder` to load `data/train` and `data/val`.  
- Apply standard transforms (resize, normalize).  
- Define a small CNN or use pretrained ResNet18.  
- Train a **binary classification task** (e.g., ammonoid vs brachiopod).  
- Save model weights to `models/` (ignored by git).  

---

### Step 4: Interpret the Model  
Notebook: `notebooks/02_interpret_model.ipynb`  

Apply multiple interpretability methods to trained models:  
- ✅ **Grad-CAM**: visualize class-discriminative regions.  
- ✅ **Saliency maps**: highlight pixels most affecting predictions.  
- ✅ **Occlusion sensitivity**: mask regions to see prediction shifts.  
- ✅ **Integrated gradients**: path-based attribution method.  

Store results in `outputs/`.  

---

### Step 5: Evaluate Interpretations  
- Compare highlighted regions with **paleontological features** (e.g., shell sutures, ribbing, symmetry).  
- Document qualitative alignment in notebook markdown.  
- Future: consult paleontological references to ground evaluation.  

---

### Step 6: Version Control & GitHub  
- **Never commit `data/` or `models/`.**  
- Commit only:  
  - `notebooks/`  
  - `src/`  
  - `PLAN.md` and `README.md`  
  - `.gitignore`  
  - small `outputs/` visualizations (safe to share).  

Git workflow inside Colab:  

```bash
!git add notebooks/00_prepare_dataset.ipynb
!git commit -m "Add dataset prep notebook"
!git push origin main
```

---

## 🛠 Roadmap  

- [ ] Binary fossil classification + Grad-CAM  
- [ ] Add saliency maps  
- [ ] Add occlusion sensitivity  
- [ ] Add integrated gradients  
- [ ] Evaluate alignment with paleontological features  

---

✅ With this plan, you’ll have a reproducible pipeline from **dataset → training → interpretability → paleontological evaluation**.  
