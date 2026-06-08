# Project Overview

The codebase is an integrated machine learning and deep learning pipeline designed to classify breast cancer tumors as either benign or malignant. It spans two distinct analytical domains: 
1. **Deep Learning (Computer Vision):** Classification of microscopic biopsy images utilizing the BreaKHis dataset.
2. **Standard Machine Learning (Tabular Data):** Diagnostic classification using numerical clinical features from the Wisconsin Breast Cancer Dataset (WBCD) and Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---
# Model Performance Report

## Densenet-169

### Validation Model Performance
| **Info** | Epoch: 20/25 | Threshold: 0.59 | LR: 0.00000125 |
| :--- | :--- | :--- | :--- |
| **Loss** | Train: 0.0078 | Validation: 0.1542 | - |
| **Metrics** | AUCROC: 0.8721 | F-1: 0.8387 | - |
| **Score** | Recall: 0.7675 | Acc: 0.7917 | Total Images: 1152 |
| **Detected** | TP: 624 | TN: 288 | Total Clears: 912 |
| **Errors** | FN: 189 | FP: 51 | Total Errors: 240 |

### Testing Model Performance
| **Info** | Loss: 0.0565 | Threshold: 0.59 | - |
| :--- | :--- | :--- | :--- |
| **Metrics** | AUCROC: 0.9547 | F-1: 0.9266 | - |
| **Score** | Recall: 0.9441 | Acc: 0.9033 | Total Images: 1438 |
| **Detected** | TP: 878 | TN: 421 | Total Clears: 1299 |
| **Errors** | FN: 52 | FP: 87 | Total Errors: 139 |

---

## Resnet-50

### Validation Model Performance
| **Info** | Epoch: 25/25 | Threshold: 0.29 | LR: 0.00000031 |
| :--- | :--- | :--- | :--- |
| **Loss** | Train: 0.0086 | Validation: 0.1522 | - |
| **Metrics** | AUCROC: 0.8394 | F-1: 0.8554 | - |
| **Score** | Recall: 0.8438 | Acc: 0.7986 | Total Images: 1152 |
| **Detected** | TP: 686 | TN: 234 | Total Clears: 920 |
| **Errors** | FN: 127 | FP: 105 | Total Errors: 232 |

### Testing Model Performance
| **Info** | Loss: 0.0815 | Threshold: 0.29 | - |
| :--- | :--- | :--- | :--- |
| **Metrics** | AUCROC: 0.9196 | F-1: 0.9196 | - |
| **Score** | Recall: 0.9903 | Acc: 0.8880 | Total Images: 1438 |
| **Detected** | TP: 921 | TN: 356 | Total Clears: 1277 |
| **Errors** | FN: 9 | FP: 152 | Total Errors: 161 |

---

## Inception-ResNet-V2

### Validation Model Performance
| **Info** | Epoch: 21/25 | Threshold: 0.35 | LR: 0.00000125 |
| :--- | :--- | :--- | :--- |
| **Loss** | Train: 0.0194 | Validation: 0.1082 | - |
| **Metrics** | AUCROC: 0.8505 | F-1: 0.8542 | - |
| **Score** | Recall: 0.8290 | Acc: 0.8003 | Total Images: 1152 |
| **Detected** | TP: 674 | TN: 248 | Total Clears: 922 |
| **Errors** | FN: 139 | FP: 91 | Total Errors: 230 |

### Testing Model Performance
| **Info** | Loss: 0.0545 | Threshold: 0.35 | - |
| :--- | :--- | :--- | :--- |
| **Metrics** | AUCROC: 0.9249 | F-1: 0.9019 | - |
| **Score** | Recall: 0.9688 | Acc: 0.8637 | Total Images: 1438 |
| **Detected** | TP: 901 | TN: 341 | Total Clears: 1242 |
| **Errors** | FN: 29 | FP: 167 | Total Errors: 196 |

---

# Model Recreation and Training Methodology

## 1. Dataset Characteristics (BreaKHis_v1)
* **Total Patients:** 82
* **Total Images:** 7909
   * Benign: 2480
   * Malignant: 5429

## 2. Data Splitting Strategy
* **Ratio:** 70% Train / 15% Validation / 15% Test
* **Methodology:** Split conducted at the patient level first, followed by class stratification (benign/malignant), before applying the 70/15/15 split.
* **Rationale:** Prevents data leakage by ensuring no single patient's images appear in multiple subsets.

## 3. Multi-Stage Training Pipeline
Models were trained using a progressive unfreezing strategy:
1.  **Stage 1 (Classifier Initialization):** Unfreeze and train only the classification head. Save the model.
2.  **Stage 2 (Partial Fine-Tuning):** Load the Stage 1 model. Unfreeze the classifier and the final architecture blocks. Retrain using a lower learning rate and warmup epochs. Save the model.
   3.  **Stage 3 (Iterative Fine-Tuning):** Repeat the Stage 2 process for further optimization.

## 4. Checkpointing and Model Selection
* **Automated Checkpointing:** At the end of each validation epoch, the model's AUC, F1-score, and Loss are compared against previous bests. The highest-performing model for each of these three metrics is checkpointed. Only these three saved models advance to the testing phase.
* **Manual Selection:** An operator evaluates the three checkpointed models using their validation and test metrics, selecting the optimal version to initialise for the next training stage.

---
# Architecture & Design

### Architectural Approach
The system follows a modular, decoupled architecture where data loading, model construction, execution, and deployment are strictly separated. This adheres to the Single Responsibility Principle, ensuring that alterations in network architecture do not require refactoring of functions used by models.

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
## Hardware-Level PyTorch Optimization Pipeline

To achieve maximum GPU throughput, this pipeline is configured to treat data processing not as a sequence of logical operations, but as a physical supply chain. The optimizations are split into two distinct layers: the I/O Pipeline and the Compute Pipeline.

### Part 1: The I/O Pipeline
These configurations in the `DataLoader` prevent the GPU from sitting idle while waiting for the CPU to supply the next batch of images.

* **`num_workers` (Parallel CPU Decoding):** Loading an image requires fetching binary from the SSD, decompressing the PNG into a NumPy array, applying transformations, and converting it to a Tensor. This is pure CPU and I/O work. 
   * Setting `num_workers=n` spawns dedicated, parallel background processes. While the GPU is processing Batch 1, the CPU workers are simultaneously decoding Batches 2 through n. The next batch is instantly handed over without blocking the GPU.
* **`pin_memory=True` (Direct Memory Access):** GPUs physically cannot read standard "pageable" host RAM over the PCIe bus. Without pinning, the CPU must first copy data into a temporary locked staging area before transfer.
    * `pin_memory=True` locks the memory immediately upon batch creation. This bypasses the CPU staging area, allowing the GPU to use Direct Memory Access (DMA) to pull the data across the PCIe bus at maximum physical bandwidth.
* **`persistent_workers=True` (Eliminating Epoch Stalls):** By default, PyTorch kills all background CPU worker processes at the end of an epoch and spawns new ones at the start of the next. Spawning OS processes is expensive, causing a multi-second stall where GPU utilization drops to 0%.
    * This setting keeps the worker processes alive in RAM between epochs, making the transition from Epoch N to Epoch N+1 completely seamless.

### Part 2: The Compute Pipeline (Executing the Math)
These optimizations in the training loop maximize how much math the GPU executes per clock cycle.

* **Batch Size Maximization:** Modern NVIDIA GPUs possess thousands of CUDA cores. Small batch sizes leave the majority of the silicon powered on but completely idle.
   * The `batch_size` is pushed to the maximum the VRAM allows before an Out of Memory error. Fully saturating the CUDA cores ensures maximum parallel throughput and images processed per second. 
* **Automatic Mixed Precision (`autocast` & `GradScaler`):** Modern architectures contain dedicated Tensor Cores hardwired to perform 16-bit floating-point (`FP16`) matrix multiplications exponentially faster than standard CUDA cores processing 32-bit floats (`FP32`).
   * Wrapping the forward pass in PyTorch's `autocast` dynamically downcasts mathematically safe operations (like convolutions) to `FP16`. This cuts VRAM consumption by 50% and massively accelerates computation. A `GradScaler` is strictly required and utilized to artificially multiply the loss before backpropagation, preventing extremely small `FP16` gradients from mathematically underflowing to zero.

---

### Hardware Precautions & Tuning Guide

While these optimization techniques maximize the capabilities of NVIDIA's CUDA architecture, they must be scaled according to your physical hardware. Proceed with caution when testing parameter combinations to avoid severe system bottlenecks or out-of-memory (OOM) crashes.

* **Memory Load Risk:** Using `pin_memory=True` alongside multiple `persistent_workers` creates a significant, static load on your system's RAM. Ensure your machine has sufficient overhead before pushing these values high.
* **Baseline Recommendation:** Start conservatively with `num_workers=2` and `batch_size=32`. Incrementally increase these parameters while monitoring system stability.
* **Bottleneck Diagnosis:** If GPU utilization is high but frequently plummets mid-epoch while VRAM remains low, your CPU is failing to supply data fast enough. Assuming your CPU is not already at 100% capacity, you should increase `num_workers` or `batch_size`.
* **Thermal Throttling:** Components under this load will generate immense heat. Modern NVIDIA GPUs default to a thermal limit of 87°C. Do not ignore minor thermal throttling. When the GPU hits this limit, the firmware panics, rapidly cutting and restoring power. This erratic voltage cycling will cripple your training speed far more than running at a stable, slightly lower clock speed. Ensure adequate cooling.
* **VRAM Arithmetic:** Maintain an awareness of your memory budget. The volume of data processed depends on the model size, parameter unfreezing logic, and tensor dimensions. Seemingly minor adjustments like changing `transforms.Resize()` from `224x224` to `299x299` can instantly cause a fatal OOM exception.
* **Reference Setup Performance:**  **Specs:** NVIDIA RTX 3060 Ti (8GB VRAM), Intel Core i5-11400, 32GB @ 3200MHz RAM, Windows 11 Pro, CUDA 12.1.
* **Benchmarks:** A complete train/validate/test loop on the BreaKHis dataset (25 epochs, `num_workers=6`, `batch_size=64`) averages **10 to 15 minutes** depending on the specific model topology (DenseNet vs. Inception) and the ratio of unfrozen weights. Under certain thermal conditions, pushing to `batch_size=128` actually reduced throttling by optimizing the compute-to-transfer ratio, allowing the GPU to run more efficiently.
---

# Links

* **BreaKHis_v1 Dataset:** https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
* **GITHUB:** https://github.com/KhaledAMagdi/NNBCD.git
* **Deployment main:** https://quartz-i--breakhis-classifier-ui.modal.run/
* **Deployment backup:** https://khaledmadgdi--breakhis-classifier-ui.modal.run 
