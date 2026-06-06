# Project Overview

The codebase is an integrated machine learning and deep learning pipeline designed to classify breast cancer tumors as either benign or malignant. It spans two distinct analytical domains: 
1. **Deep Learning (Computer Vision):** Classification of microscopic biopsy images utilizing the BreaKHis dataset.
2. **Standard Machine Learning (Tabular Data):** Diagnostic classification using numerical clinical features from the Wisconsin Breast Cancer Dataset (WBCD) and Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---

# Architecture & Design

### Architectural Approach
The system follows a modular, decoupled architecture where data loading, model construction, execution, and deployment are strictly separated. This adheres to the Single Responsibility Principle, ensuring that alterations in network architecture do not require refactoring of functions used by multiple models.

### Dependency Flow
1. **Data Layer:** `DeepLearning_DataLoader.py` / `MachineLearning_DataLoader.py` process raw data (images or `.xlsx` files).
2. **Model Layer:** `DeepLearning_Build.py` constructs standard Torchvision models and replaces classification heads.
3. **Execution Layer:** `DeepLearning_Main.py` / `WBCD_Original_Models.py` / `WDBC_Diagnostic_Models.py`  orchestrate the training, validation, and testing of models.
4. **Presentation/Serving Layer:** `modal_app.py` acts as the cloud entry point, mounting saved weights (`.pth`) to containerized cloud instances.

---

# File-by-File Deep Dive



## Machine Learning

## `MachineLearning_DataLoader.py`
**Purpose:** Data ingestion for the tabular ML components.

### Functions
* `Load_Data`: Utilizes pandas to read the specific columns required from the local Excel files containing the WBCD/WDBC clinical parameters.
* `main`: Routes the extracted data to the respective analysis modules.

## `WBCD_Original_Models.py` & `WDBC_Diagnostic_Models.py`
**Purpose:** Evaluates tabular clinical parameters utilizing shallow neural networks and standard ML techniques (like SMOTE for oversampling).

### Classes
* `MedicalNN`: A standard PyTorch feed-forward multi-layer perceptron (MLP) for numeric features.

### Functions
* `lock_seeds`: Ensures exact reproducibility by locking `random`, `numpy`, and `torch` seeds.
* `prepare_data`: Splits datasets, applies `StandardScaler`, and manages class balancing.
* `plot_roc_curves` / `plot_model_comparison`: Uses `matplotlib` and `seaborn` to output performance visualizations.
* `train_torch_model`: Executes the feed-forward network training cycle.
* `WBCD_Original_Analysis` / `WDBC_Diagnostic_Analysis`: Pipeline controllers connecting data extraction to model execution.
* `Cross_Validate_...`: Executes K-Fold cross-validation to guarantee statistical validity of the tabular models.

---
## Deep Learning
## `DeepLearning_Build.py`
**Purpose:** Acts as the network topology factory for the computer vision pipeline.
* **Role:** Fetches standard `torchvision` and `timm` models, strips their fully connected output layers, and appends custom sequential blocks tailored for binary classification (malignant vs. benign).

### Functions
* `build_densenet_model`, `build_resnet_model`, `build_inception_model`:
  * **Inputs:** `device` (CPU/GPU routing), `retrain` (boolean to load pretrained model or local .pth file), `UF_flag` (unfreeze layers beyond classifier), `weights_path` (path for .pth file).
  * **Logic:** Instantiates base models. Replaces `classifier` with custome made classifier made to reduce outputed features to 2 classes, malignant or benign.
  Chooses to keep pretrained weights or load local weights.
  Chooses to unfreeze the last couple of layers (depending on model), by default, only classifier is unfrozen.
  * **Outputs:** Returns the compiled `nn.Module`.
* `build_model`: 
  * **Logic:** A master router function that calls the specific model builder based on a string argument.

## `DeepLearning_DataLoader.py`
**Purpose:** Handles the filesystem traversal, split generation, and manifest creation for the BreaKHis image dataset.

### Functions
* `generate_split_manifest`: Iterates through the raw image directory, extracts labels based on file paths, and builds an exhaustive `.xlsx` index to track training/validation/testing splits. 
* `load_BreaKHis_manifest`: Reads the manifest into memory.
* `main`: Orchestrates the dataset preparation.

## `DeepLearning_Main.py`
**Purpose:** The core engine for training and evaluating the deep learning models.

### Classes
* `BreaKHisDataset(Dataset)`: 
  * **Responsibility:** A custom dataset class that reads image paths, loads them using `PIL.Image`, and applies defined `transformations` (resize, tensor conversion, normalization), and returns PyTorch tensors and imae label.
* `FastFocalLoss`:
  * **Responsibility:** Wraps focal loss as a reusable object matching BCE's interface, making the train loop more concise.

### Functions
* `prepare_dataloaders`: Initializes `WeightedRandomSampler` to handle class imbalances at the batch level. Returns mapped PyTorch `DataLoader` objects.
* `train_model`: The primary training loop. Executes forward passes, gradient calculations (`loss.backward()`), optimizer steps, and logs metrics (Loss, Accuracy, AUC, F1).
 As well as validation, saving the best model according to AUC, F1, and loss per epoch, and calculating a dynamic threshold that works on minimising the false positive rate and maximising the true positive rate. It also implements warmup epochs, learning rate reduction based on patience.
Finally, the test data is also carried out here, with the use of TTA (Test Time Augmentation) to try to get the best possible performance for a model.

* `Run_Model`: Controller function that binds the data loaders, model builders, and training loop.
All changable parameters are passed from this function from three neat dictionaries for each of the three phases, building, preparing data, and training.

---

## Optimization & Training Strategy

This pipeline employs several advanced optimization techniques to ensure mathematical stability, prevent overfitting on small medical datasets, and accelerate hardware execution.

### 1. Optimizer: AdamW (Decoupled Weight Decay)
The network utilizes the **AdamW** optimizer rather than standard Adam or SGD. In highly parameterized vision models (like Inception-ResNet-v2), standard Adam struggles with L2 regularization because the weight decay penalty is mixed into the gradient's moving averages. AdamW decouples the weight decay step from the gradient update, applying a flat decay rate to all weights equally. This enforces strict regularization, preventing the model from memorizing the training noise of the histology images.

### 2. Objective Function: Focal Loss
Medical datasets typically suffer from severe class imbalances (e.g., an overwhelming number of malignant samples compared to specific benign subtypes). To combat this, the pipeline replaces standard Cross-Entropy with **Focal Loss**. 
* Focal Loss dynamically scales the loss based on prediction confidence. 
* It mathematically down-weights the penalty for "easy" classifications and exponentially increases the penalty for hard-to-classify, edge-case tumors, forcing the optimizer to focus on the hardest visual features.

### 3. Class Imbalance: Weighted Random Sampling
In conjunction with Focal Loss, the `DataLoader` utilizes PyTorch's `WeightedRandomSampler`. Instead of sequentially iterating through the dataset, the sampler assigns a probability weight to every image based on the inverse frequency of its class. This guarantees that every single batch sent to the GPU contains a mathematically balanced ratio of benign and malignant samples, preventing the gradients from collapsing toward the majority class.

### 4. Transfer Learning & Layer Freezing
To prevent "catastrophic forgetting" of the ImageNet-trained feature extractors, the architecture utilizes a staggered unfreezing strategy:
* **Initial State:** The massive foundational convolutional blocks (Stem, A, and B blocks) are strictly frozen (`requires_grad = False`). 
* **Custom Head:** A custom heavily regularized Multi-Layer Perceptron (1024 -> 256 -> 1 with 50% Dropout) is attached and trained from scratch.
* **Fine-Tuning:** Only classifier is trained at the begening, then model with trained classifer is loaded and retrained with only the final couple blocks of the network unfrozen to allow the network to adapt to the medical dataset.

### 5. Hardware Acceleration: Automatic Mixed Precision (AMP)
The training loop is wrapped in PyTorch's `autocast` context manager alongside a `GradScaler`. 
* Instead of running the entire network in standard 32-bit floating-point math (`FP32`), AMP dynamically downcasts specific mathematically safe operations (like convolutions) to 16-bit (`FP16`). 
* This drastically reduces VRAM consumption, allowing for larger batch sizes while significantly accelerating matrix multiplications on the Tensor Cores of modern NVIDIA GPUs.
---

# Links

* **BreaKHis_v1 Dataset:** https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
* **GITHUB:** https://github.com/KhaledAMagdi/NNBCD.git
* **Deployment main:** https://quartz-i--breakhis-classifier-ui.modal.run/
* **Deployment backup:** https://khaledmadgdi--breakhis-classifier-ui.modal.run 
