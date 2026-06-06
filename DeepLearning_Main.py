import copy
import time
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.ops import sigmoid_focal_loss
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, confusion_matrix, roc_curve
from DeepLearning_Build import build_model


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
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

class FastFocalLoss(nn.Module):
    #function is pure math, putting it in train loop means
    #plugging in the alpha gamma and values every time unlike
    #BCE that is already an object
    #with class: loss = criterion(outputs, targets)
    #without:    loss = sigmoid_focal_loss(outputs, targets, alpha=alpha_val, gamma=gamma_val)
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
def prepare_dataloaders(df, posW, sampler_flag, model_name, batches=32, workers=0):
    print("\nPreparing DataLoaders...")
    # Using the mean and std of Imagenet is a common practice. They are calculated based on millions of images.
    # If you want to train from scratch on your own dataset, you can calculate the new mean and std.
    # Otherwise, using the Imagenet pretrained model with its own mean and std is recommended.
    # In other words, this is the mean of the RGB channels of the Imagenet dataset.
    # The std is the standard deviation of the RGB channels of the Imagenet dataset.
    # InceptionResnetV2 input is 299x299, while Resnet50 & DenseNet169 input is 224x224

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    IMAGE_SIZE = [299, 299] if model_name == 'inception' else [224, 224]

    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),

        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    test_transform = val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    train_df = df[df['split'] == 'train'].copy().reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].copy().reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].copy().reset_index(drop=True)

    sampler = None
    if sampler_flag:
        balanced_epoch_size = int(train_df['label'].value_counts().max() * 2)
        sampler_weights = torch.DoubleTensor([posW if label == 1 else 1 for label in train_df['label']])
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=balanced_epoch_size,
            replacement=True
        )

    train_dataset = BreaKHisDataset(train_df, transform=train_transform)
    val_dataset   = BreaKHisDataset(val_df,   transform=val_transform)
    test_dataset  = BreaKHisDataset(test_df,  transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=not sampler_flag,
        batch_size=batches,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batches,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0
    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batches,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0
    )

    print_dict = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    for name, split_df in print_dict.items():
        print(f"{name}: {len(split_df)} images ({split_df['patient_id'].nunique()} patients)")
        print(f"Malignant: {split_df[split_df['label'] == 1].shape[0]:<4} images | "
              f"Benign: {split_df[split_df['label'] == 0].shape[0]:<4} images\n")

    return train_loader, val_loader, test_loader


# ==========================================
# 3. The Training Engine
# ==========================================
def train_model(model, train_loader, val_loader, test_loader, device,
                epochs, warmup_epochs, learning_rate, weight_decay,
                positive_weight, sampler_flag, focal_flag, TTA_flag, save_location):
    print(f"\nStarting Training for {epochs} Epochs...")

    weight = 1 if sampler_flag else positive_weight

    if focal_flag:
        alpha = weight / (weight + 1.0)
        criterion = FastFocalLoss(alpha=alpha, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]).to(device))

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs) if warmup_epochs > 0 else None
    main_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.amp.GradScaler(device=device.type, enabled=(device.type == 'cuda'))

    best_f1   = 0.0
    best_auc  = 0.0
    best_loss = 999999.0
    best_model_wts_f1   = None
    best_model_wts_auc  = None
    best_model_wts_loss = None
    BM_f1   = {}
    BM_auc  = {}
    BM_loss = {}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_train_samples = 0

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
            total_train_samples += inputs.size(0)
        print('\r' + ' ' * 100 + '\r', end='', flush=True)

        epoch_train_loss = running_loss / total_train_samples #by ashrofa !!

        model.eval()
        val_loss   = 0.0
        val_labels = []
        val_probs  = []

        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validating", leave=False):
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                probs = torch.sigmoid(outputs)

            val_labels.extend(labels.cpu().numpy())
            val_probs.extend(probs.cpu().numpy())
        print('\r' + ' ' * 100 + '\r', end='', flush=True)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_labels_cpu = np.array(val_labels)
        val_probs_cpu  = np.array(val_probs)
        current_lr = optimizer.param_groups[0]['lr']

        fpr, tpr, thresholds = roc_curve(val_labels_cpu, val_probs_cpu)
        j_scores = tpr - fpr
        optimal_threshold = thresholds[np.argmax(j_scores)]
        preds = (val_probs_cpu >= optimal_threshold).astype(int)

        val_acc        = accuracy_score(val_labels_cpu, preds)
        val_auc        = roc_auc_score(val_labels_cpu, val_probs_cpu)
        val_f1         = f1_score(val_labels_cpu, preds, zero_division=0)
        val_recall     = recall_score(val_labels_cpu, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(val_labels_cpu, preds).ravel()
        TE = fn + fp
        TC = tp + tn
        TI = TE + TC

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step(epoch_val_loss)

        # print(f"Info     | Epoch:  {epoch+1:2}/{epochs}  | Threshold:  {optimal_threshold:.2f}     | LR: {current_lr:.8f}")
        # print(f"Loss     | Train:  {epoch_train_loss:.4f} | Validation: {epoch_val_loss:.4f}  |")
        # print(f"Metrics  | AUCROC: {val_auc:.4f} | F-1: {val_f1:.4f}         |")
        # print(f"Score    | Recall: {val_recall:.4f} | Acc: {val_acc:.4f}         | Total Images: {TI}")
        # print(f"Detected | TP:      {tp:<4}  | TN:   {tn:<3}           | Total Clears: {TC}")
        # print(f"Errors   | FN:      {fn:<4}  | FP:   {fp:<3}           | Total Errors: {TE}")
        BM_tmp = {
            'epoch': epoch + 1, 'val_loss': epoch_val_loss, 'train_loss': epoch_train_loss,
            'threshold': optimal_threshold, 'learning_rate': current_lr,
            'auc': val_auc, 'acc': val_acc, 'recall': val_recall, 'f1': val_f1,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'TE': TE, 'TC': TC, 'TI': TI
        }
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_wts_f1 = copy.deepcopy(model.state_dict())
            BM_f1 = BM_tmp
            # print(f"F1 value improved, Tracker Updated: {best_f1:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_wts_auc = copy.deepcopy(model.state_dict())
            BM_auc = BM_tmp
            # print(f"AUC value improved, Tracker Updated: {best_auc:.4f}")
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts_loss = copy.deepcopy(model.state_dict())
            BM_loss =BM_tmp
            # print(f"Loss value improved, Tracker Updated: {best_loss:.4f}")
        # print("-" * 70)

    Best_Models = {'F1': BM_f1, 'AUC': BM_auc, 'Loss': BM_loss}

    print("--- Training Complete ---")
    for name, BM in Best_Models.items():
        print(f"--- Validation Best {name} Model Performance ---")
        print(f"Info     | Epoch:  {BM['epoch']:2}/{epochs}  | Threshold:  {BM['threshold']:.2f}     | LR: {BM['learning_rate']:.8f}")
        print(f"Loss     | Train:  {BM['train_loss']:.4f} | Validation: {BM['val_loss']:.4f}  |")
        print(f"Metrics  | AUCROC: {BM['auc']:.4f} | F-1: {BM['f1']:.4f}         |")
        print(f"Score    | Recall: {BM['recall']:.4f} | Acc: {BM['acc']:.4f}         | Total Images: {BM['TI']}")
        print(f"Detected | TP:      {BM['tp']:<4}  | TN:   {BM['tn']:<3}           | Total Clears: {BM['TC']}")
        print(f"Errors   | FN:      {BM['fn']:<4}  | FP:   {BM['fp']:<3}           | Total Errors: {BM['TE']}")
        print("-" * 70)

    path_f1   = save_location + "_f1.pth"
    path_auc  = save_location + "_auc.pth"
    path_loss = save_location + "_loss.pth"
    torch.save(best_model_wts_f1, path_f1)
    torch.save(best_model_wts_auc, path_auc)
    torch.save(best_model_wts_loss, path_loss)

    best_model_wts = {'F1': best_model_wts_f1, 'AUC': best_model_wts_auc, 'Loss': best_model_wts_loss}

    print("Models saved to " + save_location)
    print("-" * 70)
    print("-" * 70)
    print(f"\nStarting Testing...")

    for name, model_wts in best_model_wts.items():
        model.load_state_dict(model_wts)
        model.eval()
        TH = Best_Models[name]['threshold']
        test_loss = 0.0
        test_labels = []
        test_probs  = []
        test_preds  = []

        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).to(device)

            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item() * inputs.size(0)

                if TTA_flag:
                    out_hflip = model(torch.flip(inputs, dims=[3]))
                    out_vflip = model(torch.flip(inputs, dims=[2]))
                    out_rot90 = model(torch.rot90(inputs, k=1, dims=[2, 3]))

                    probs = (torch.sigmoid(outputs  ) +
                             torch.sigmoid(out_hflip) +
                             torch.sigmoid(out_vflip) +
                             torch.sigmoid(out_rot90)) / 4.0
                else:
                    probs = torch.sigmoid(outputs)

                preds = (probs >= TH).float()

            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
        print('\r' + ' ' * 100 + '\r', end='', flush=True)

        test_loss /= len(test_loader.dataset)

        test_labels_cpu = np.array(test_labels)
        test_probs_cpu  = np.array(test_probs)
        test_preds_cpu  = np.array(test_preds)

        test_acc       = accuracy_score(test_labels_cpu, test_preds_cpu)
        test_auc       = roc_auc_score(test_labels_cpu, test_probs_cpu)
        test_f1        = f1_score(test_labels_cpu, test_preds_cpu, zero_division=0)
        test_recall    = recall_score(test_labels_cpu, test_preds_cpu, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(test_labels_cpu, test_preds_cpu).ravel()
        TE = fn + fp
        TC = tp + tn
        TI = TE + TC

        print('\r' + ' ' * 100 + '\r', end='')
        print(f"--- Testing Best {name} Model Performance ---")
        print(f"Info     | Loss:   {test_loss:.4f} | Threshold:  {TH:.2f}    |")
        print(f"Metrics  | AUCROC: {test_auc:.4f} | F-1: {test_f1:.4f}         |")
        print(f"Score    | Recall: {test_recall:.4f} | Acc: {test_acc:.4f}         | Total Images: {TI}")
        print(f"Detected | TP:      {tp:<4}  | TN:   {tn:<3}           | Total Clears: {TC}")
        print(f"Errors   | FN:      {fn:<4}  | FP:   {fp:<3}           | Total Errors: {TE}")
    print("--- Testing Complete ---")
    print("--------- Done ---------")
    print("-" * 70)


# ==========================================
# 4. Main Execution
# ==========================================
def print_table(title, args_dict):
    table = Table(show_header=False, box=None)
    table.add_column(justify="left")
    table.add_column(justify="center")
    table.add_column(justify="left")
    for key, val in args_dict.items():
        if key == "df": continue
        val_str = "DataLoader"      if "DataLoader" in str(val) else str(val)
        val_str = str(f"{val:.4f}") if key=="positive_weight" or key=="posW" else val_str
        table.add_row(key,"→",val_str)
    console = Console()
    console.print(Panel(table, title=title, expand=False))


def Run_Model(Data, device, model_name, pretrained_model_path, save_path):
    start = time.perf_counter()

    df = Data.copy()
    total_benign_images = df[df['label'] == 0].shape[0]
    total_malignant_images = df[df['label'] == 1].shape[0]
    posW = total_benign_images/total_malignant_images

    retrain      = True if pretrained_model_path else False
    UF_flag      = True if retrain else False
    focal_flag   = True
    TTA_flag     = True
    sampler_flag = True

    epoch_num    = 25
    warmup_num   = 5    if retrain else 0      # phase0 0   , phase1/2 5
    LR           = 1e-5 if retrain else 1e-4   # phase0 1e-4, phase1/2 1e-5
    weight_decay = 1e-4

    batches = 128
    workers = 6

    weights_path  = pretrained_model_path
    save_location = save_path

    dataloader_kwargs = {
        'df': df,
        'posW': posW,
        'sampler_flag': sampler_flag,
        'model_name': model_name,
        'batches': batches,
        'workers': workers
    }
    model_kwargs = {
        'model_name':   model_name,
        'device':       device,
        'retrain':      retrain,
        'UF_flag':      UF_flag,
        'weights_path': weights_path
    }
    train_kwargs = {
        'device':          device,
        'epochs':          epoch_num,
        'warmup_epochs':   warmup_num,
        'learning_rate':   LR,
        'weight_decay':    weight_decay,
        'positive_weight': posW,
        'sampler_flag':    sampler_flag,
        'focal_flag':      focal_flag,
        'TTA_flag':        TTA_flag,
        'save_location':   save_location
    }

    print_table("DataLoader Arguments", dataloader_kwargs)
    print_table("Model Arguments", model_kwargs)
    print_table("Train Arguments", train_kwargs)

    loaders = prepare_dataloaders(**dataloader_kwargs)
    model = build_model(**model_kwargs)
    train_model(model, *loaders, **train_kwargs)

    end = time.perf_counter()
    print("\n" + "-" * 36)
    print(f"{model_name} time: {end - start:.0f} seconds")

