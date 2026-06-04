import os
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, models
from torchvision.ops import sigmoid_focal_loss
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix


# ==========================================
# 1. Dataset Class & Custom Loss Function
# ==========================================
class BreaKHisDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class FastFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs,
            targets.float(),
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean'
        )


# ==========================================
# 2. DataLoaders & Splitting
# ==========================================
def prepare_dataloaders(df, sampler_flag, B2M_ratio, batches=32, workers=0):
    print("\nPreparing DataLoaders...")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    print(f"Training: {len(train_df)} images ({train_df['patient_id'].nunique()} patients)")
    print(f"Benign: {train_df[train_df['label'] == 0].shape[0]} images |"
          f" Malignant: {train_df[train_df['label'] == 1].shape[0]} images")
    print(f"Validation: {len(val_df)} images ({val_df['patient_id'].nunique()} patients)")
    print(f"Benign: {val_df[val_df['label'] == 0].shape[0]} images |"
          f" Malignant: {val_df[val_df['label'] == 1].shape[0]} images")

    weights = [B2M_ratio if label == 1 else 1.0 for label in train_df['label']]
    sampler_weights = torch.DoubleTensor(weights)

    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True
    )

    train_dataset = BreaKHisDataset(train_df, transform=train_transform)
    val_dataset = BreaKHisDataset(val_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,

        sampler=sampler if sampler_flag else None,
        shuffle=False if sampler_flag else True,

        batch_size=batches,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batches,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True if workers > 0 else False
    )

    return train_loader, val_loader


# ==========================================
# 3. Model Architecture
# ==========================================
def build_densenet_model(device, retrain, D4N5_flag, weights_path):
    print("\nDownloading and Building DenseNet-169...")
    model = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)

    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1)
    )

    if retrain:
        print(f"Loading model: {os.path.basename(weights_path)}")
        model.load_state_dict(
            torch.load(
                weights_path,
                map_location=device,
                weights_only=True
            )
        )

    for param in model.parameters():
        param.requires_grad = False

#----------------Layers to be unfrozen----------------#
    for param in model.classifier.parameters():
        param.requires_grad = True

    if D4N5_flag:
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

        for param in model.features.norm5.parameters():
            param.requires_grad = True

    return model.to(device)


# ==========================================
# 4. The Training Engine
# ==========================================
def train_model(model, train_loader, val_loader, device,
                epochs, warmup_epochs, learning_rate, weight_decay,
                positive_weight, threshold, sampler_flag, focal_flag, TTA_flag, save_location):
    print(f"\nStarting Training for {epochs} Epochs...")

    if sampler_flag:
        if focal_flag:
            criterion = FastFocalLoss(alpha=0.5, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        if focal_flag:
            criterion = FastFocalLoss(alpha=positive_weight, gamma=2.0)
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight]).to(device))

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs
        )
    else:
        warmup_scheduler = None

    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == 'cuda'))

    best_f1 = 0.0
    best_auc = 0.0
    best_loss = 999999.0

    best_model_wts_f1 = None
    best_model_wts_auc = None
    best_model_wts_loss = None

    BM_f1 = {}
    BM_auc = {}
    BM_loss = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        all_labels = []
        all_probs = []
        all_preds = []

        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

                if TTA_flag:
                    img_hflip = torch.flip(inputs, dims=[3])
                    img_vflip = torch.flip(inputs, dims=[2])
                    img_rot90 = torch.rot90(inputs, k=1, dims=[2, 3])

                    prob_orig  = torch.sigmoid(outputs)
                    prob_hflip = torch.sigmoid(model(img_hflip))
                    prob_vflip = torch.sigmoid(model(img_vflip))
                    prob_rot90 = torch.sigmoid(model(img_rot90))

                    probs = (prob_orig + prob_hflip + prob_vflip + prob_rot90) / 4.0
                else:
                    probs = torch.sigmoid(outputs)

                preds = (probs >= threshold).float()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        epoch_val_loss = val_loss / len(val_loader.dataset)

        all_labels_cpu = np.array(all_labels)
        all_probs_cpu  = np.array(all_probs)
        all_preds_cpu  = np.array(all_preds)

        val_acc = accuracy_score(all_labels_cpu, all_preds_cpu)
        val_auc = roc_auc_score(all_labels_cpu, all_probs_cpu)
        val_f1 = f1_score(all_labels_cpu, all_preds_cpu, zero_division=0)
        val_recall = recall_score(all_labels_cpu, all_preds_cpu, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(all_labels_cpu, all_preds_cpu).ravel()
        TE = fn + fp
        TC = tp + tn
        TI = TE + TC
        current_lr = optimizer.param_groups[0]['lr']

        print('\r' + ' ' * 100 + '\r', end='')
        # print(f"Info     | Epoch:  {epoch+1:2}/{epochs}  | Threshold:  {threshold}     | LR: {current_lr:.8f}")
        # print(f"Loss     | Train:  {epoch_train_loss:.4f} | Validation: {epoch_val_loss:.4f}  |")
        # print(f"Metrics  | AUCROC: {val_auc:.4f} | F-1: {val_f1:.4f}         |")
        # print(f"Score    | Recall: {val_recall:.4f} | Acc: {val_acc:.4f}         | Total Images: {TI}")
        # print(f"Detected | TP:      {tp:<4}  | TN:   {tn:<3}           | Total Clears: {TC}")
        # print(f"Errors   | FN:      {fn:<4}  | FP:   {fp:<3}           | Total Errors: {TE}")

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(val_loss)

        # time.sleep(1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            BM_f1 = {
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'train_loss': epoch_train_loss,
                'learning_rate': current_lr,
                'auc': val_auc,
                'acc': val_acc,
                'recall': val_recall,
                'f1': val_f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'TE': TE,
                'TC': TC,
                'TI': TI
            }
            # print(f"F1 value improved, Tracker Updated: {best_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts_auc = copy.deepcopy(model.state_dict())
            BM_auc = {
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'train_loss': epoch_train_loss,
                'learning_rate': current_lr,
                'auc': val_auc,
                'acc': val_acc,
                'recall': val_recall,
                'f1': val_f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'TE': TE,
                'TC': TC,
                'TI': TI
            }
            # print(f"AUC value improved, Tracker Updated: {best_auc:.4f}")

        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            BM_loss = {
                'epoch': epoch + 1,
                'val_loss': epoch_val_loss,
                'train_loss': epoch_train_loss,
                'learning_rate': current_lr,
                'auc': val_auc,
                'acc': val_acc,
                'recall': val_recall,
                'f1': val_f1,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'TE': TE,
                'TC': TC,
                'TI': TI
            }
            # print(f"Loss value improved, Tracker Updated: {best_loss:.4f}")
        # time.sleep(1)
        # print("-"*70)
        # time.sleep(1)

    Best_Models = {'F1':   BM_f1,
                   'AUC':  BM_auc,
                   'Loss': BM_loss}

    print("--- Training Complete ---")
    for name, BM in Best_Models.items():
        print(f"--- Final Best {name} Model Performance ---")
        print(f"Info     | Epoch:  {BM['epoch']:2}/{epochs}  | Threshold:  {threshold}     | LR: {BM['learning_rate']:.8f}")
        print(f"Loss     | Train:  {BM['train_loss']:.4f} | Validation: {BM['val_loss']:.4f}  |")
        print(f"Metrics  | AUCROC: {BM['auc']:.4f} | F-1: {BM['f1']:.4f}         |")
        print(f"Score    | Recall: {BM['recall']:.4f} | Acc: {BM['acc']:.4f}         | Total Images: {BM['TI']}")
        print(f"Detected | TP:      {BM['tp']:<4}  | TN:   {BM['tn']:<3}           | Total Clears: {BM['TC']}")
        print(f"Errors   | FN:      {BM['fn']:<4}  | FP:   {BM['fp']:<3}           | Total Errors: {BM['TE']}")
        print("-" * 70)

    path_f1 = save_location + "_f1.pth"
    path_auc = save_location + "_auc.pth"
    path_loss = save_location + "_loss.pth"
    torch.save(best_model_wts_f1, path_f1)
    torch.save(best_model_wts_auc, path_auc)
    torch.save(best_model_wts_loss, path_loss)
    print("Models saved to " + save_location)
    print("-" * 70)


# ==========================================
# 5. Main Execution
# ==========================================
def DenseNet_Model(Data, device, pretrained_model_path, save_path):
    start = time.perf_counter()

    df = Data.copy()
    total_benign_images = df[df['label'] == 0].shape[0]
    total_malignant_images = df[df['label'] == 1].shape[0]
    B2M_ratio = total_benign_images/total_malignant_images

    retrain      = True
    D4N5_flag    = True
    focal_flag   = True
    TTA_flag     = True
    sampler_flag = True

    epoch_num    = 25
    warmup_num   = 5       # phase0 0   , phase1/2 5   , phase3 2
    LR           = 1e-5    # phase0 1e-4, phase1/2 1e-5, phase3 5e-6
    weight_decay = 1e-4    # phase0 1e-4, phase1/2 1e-4, phase3 3e-4
    TH           = 0.3

    batches = 64
    workers = 6

    weights_path  = pretrained_model_path
    save_location = save_path

    train_loader, val_loader = prepare_dataloaders(df, sampler_flag, B2M_ratio, batches, workers)

    model_kwargs = {
        'device':       device,
        'retrain':      retrain,
        'D4N5_flag':    D4N5_flag,
        'weights_path': weights_path
    }

    train_kwargs = {
        'train_loader':    train_loader,
        'val_loader':      val_loader,
        'device':          device,
        'epochs':          epoch_num,
        'warmup_epochs':   warmup_num,
        'learning_rate':   LR,
        'weight_decay':    weight_decay,
        'positive_weight': B2M_ratio,
        'threshold':       TH,
        'sampler_flag':    sampler_flag,
        'focal_flag':      focal_flag,
        'TTA_flag':        TTA_flag,
        'save_location':   save_location
    }

    model = build_densenet_model(**model_kwargs)
    train_model(model=model, **train_kwargs)

    #----------------------------------------------------------------------
    #
    # external_path = "DenseNetModels/MultiFactorTrackers/Densenet_ClaD4N5_L0_TTAFocalSampler_v2"
    # weights_path = {
    #     '_loss_loss': external_path + "_loss_loss.pth",
    #     '_auc_loss': external_path + "_auc_loss.pth",
    #     '_f1_loss': external_path + "_f1_loss.pth",
    #     '_loss_auc': external_path + "_loss_auc.pth",
    #     '_auc_auc': external_path + "_auc_auc.pth",
    #     '_f1_auc': external_path + "_f1_auc.pth",
    #     '_loss_f1': external_path + "_loss_f1.pth",
    #     '_auc_f1': external_path + "_auc_f1.pth",
    #     '_f1_f1': external_path + "_f1_f1.pth"
    # }
    #
    # model_kwargs['weights_path'] = weights_path
    # for name, path in model_kwargs['weight_path'].items():
    #     model_start = time.perf_counter()
    #     train_kwargs['save_location'] = "DenseNetModels/MultiFactorTrackers/Densenet_ClaD4N5_L1_TTAFocalSampler_v2"+name
    #     model = build_densenet_model(**model_kwargs)
    #     train_model(model=model, **train_kwargs)
    #     model_end = time.perf_counter()
    #     print("\n" + "-" * 30)
    #     print(f"Elapsed time: {model_end - model_start:.6f} seconds")
    #     print("-" * 30)
    #     del model
    #     torch.cuda.empty_cache()

    end = time.perf_counter()
    print("\n" + "-" * 30)
    print(f"Elapsed time: {end - start:.6f} seconds")
    print("-" * 30)