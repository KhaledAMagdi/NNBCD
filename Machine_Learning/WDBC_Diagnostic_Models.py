#Helpers & Visualizations
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

#PyTorch (Neural Networks)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#Data Processing & Evaluation
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
#Machine Learning Models
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ==========================================
# 1. Reproducibility
# ==========================================
def lock_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 2. Data Preprocessing & Preparation
# ==========================================
def prepare_data(data, target_col):
    data[target_col] = data[target_col].map({'B': 0, 'M': 1})
    X = data.drop(columns=target_col).astype(float)
    y = data[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    return X_train_res, X_test_scaled, y_train_res, y_test, data


# ==========================================
# 3. Plotting Utilities
# ==========================================
def plot_roc_curves(model_results, y_test):
    plt.figure(figsize=(9, 7))
    for res in model_results:
        if res['y_proba'] is not None:
            fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
            plt.plot(fpr, tpr, label=f"{res['Model']} (AUC = {res['ROC AUC']:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results_df):
    plt.figure(figsize=(10, 5))
    melted_df = results_df.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1 Score"])
    sns.barplot(data=melted_df, x="Model", y="value", hue="variable")
    plt.title("Model Metric Comparison")
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. PyTorch Neural Network Architecture
# ==========================================
class MedicalNN(nn.Module):
    def __init__(self, input_size):
        super(MedicalNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)


def train_torch_model(X_train, X_test, y_train, y_test, epochs=200, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training PyTorch on: {device}")

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MedicalNN(X_train.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    pos_weight = torch.tensor([2.0]).to(device)

    for epoch in tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
        model.train()
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            weights = torch.where(batch_y == 1, pos_weight, torch.tensor(1.0).to(device))
            loss = nn.functional.binary_cross_entropy_with_logits(outputs, batch_y, weight=weights)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
        preds = (probs >= 0.4).astype(int)

    return probs.flatten(), preds.flatten(), model


# ==========================================
# 5. Analysis Function
# ==========================================
def mean_values(data, target_col):
    benign = data.groupby(target_col).get_group(0)
    malignant = data.groupby(target_col).get_group(1)
    print("--- Dataset Mean ---")
    print(data.mean())
    print("\n")
    print("--- Benign Mean ---")
    print(benign.mean())
    print("\n")
    print("--- Malignant Mean ---")
    print(malignant.mean())
    print("\n")


def analyze_mistakes(y_test, predictions, model_name="Model"):
    print("\nInvestigating PyTorch Neural Network Mistakes...")
    incorrect_mask = np.array(y_test) != np.array(predictions)
    original_row_numbers = y_test.index[incorrect_mask] + 2
    mistakes_df = pd.DataFrame({
        'True Label': np.array(y_test)[incorrect_mask],
        'Predicted Label': np.array(predictions)[incorrect_mask]
    }, index=original_row_numbers)
    mistakes_df.index.name = 'Original Row'
    print(f"\n{model_name} Mistakes: {len(mistakes_df)} out of {len(y_test)} patients.")
    print(mistakes_df.to_string())
    print("\n")

    return mistakes_df


# ==========================================
# 6. Main Execution Function
# ==========================================
def WDBC_Diagnostic_Analysis(data, target_col="diagnosis"):
    lock_seeds()

    print("Preparing data...\n")
    X_train, X_test, y_train, y_test, data = prepare_data(data, target_col)
    mean_values(data, target_col)

    results = []
    confusion_matrices = {}

    print("Training...")
    torch_probs, torch_preds, torch_model = train_torch_model(X_train, X_test, y_train, y_test)

    results.append({
        "Model": "PyTorch NN",
        "Accuracy": accuracy_score(y_test, torch_preds),
        "Precision": precision_score(y_test, torch_preds, zero_division=0),
        "Recall": recall_score(y_test, torch_preds, zero_division=0),
        "F1 Score": f1_score(y_test, torch_preds, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, torch_probs),
        "y_proba": torch_probs
    })
    confusion_matrices["PyTorch NN"] = confusion_matrix(y_test, torch_preds)

    analyze_mistakes(y_test, torch_preds, model_name="PyTorch NN")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "K-Neighbors": KNeighborsClassifier(),
        "MLP Neural Network": MLPClassifier(max_iter=1000, random_state=42)
    }

    for name, model in tqdm(models.items(), desc="Training Models", unit="model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
            "y_proba": y_proba
        })
        confusion_matrices[name] = confusion_matrix(y_test, y_pred)

    print("\n--- Final Model Comparison (Diagnostic) ---")
    results_df = pd.DataFrame(results)
    print(results_df.drop(columns='y_proba').to_string(index=False, float_format="%.4f"))

    fig, axes = plt.subplots(2,  3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                    xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
        axes[idx].set_title(name)
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.show()

    plot_roc_curves(results, y_test)
    plot_model_comparison(results_df)


def Cross_Validate_Diagnostic(data, target_col="diagnosis", n_splits=5):
    print(f"\nRunning {n_splits}-Fold Cross Validation...")

    df = data.copy()
    df[target_col] = df[target_col].map({'B': 0, 'M': 1})
    X = df.drop(columns=target_col).values
    y = df[target_col].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    cv_scores = {
        "PyTorch NN": [], "Logistic Regression": [], "Random Forest": [],
        "SVM": [], "K-Neighbors": [], "MLP Neural Network": []
    }

    fold = 1
    for train_index, test_index in skf.split(X, y):
        print(f"\n--- Processing Fold {fold}/{n_splits} ---")

        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_test_scaled = scaler.transform(X_test_fold)

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_fold)

        y_train_series = pd.Series(y_train_res)
        y_test_series = pd.Series(y_test_fold)

        torch_probs, _, _ = train_torch_model(X_train_res, X_test_scaled, y_train_series, y_test_series)
        cv_scores["PyTorch NN"].append(roc_auc_score(y_test_fold, torch_probs))

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "K-Neighbors": KNeighborsClassifier(),
            "MLP Neural Network": MLPClassifier(max_iter=1000, random_state=42)
        }

        for name, model in models.items():
            model.fit(X_train_res, y_train_res)
            y_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") \
                                                                else model.predict(X_test_scaled)
            cv_scores[name].append(roc_auc_score(y_test_fold, y_proba))

        fold += 1

    print("--- Final Cross-Validation Results (AUC) ---")
    for model_name, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{model_name:>20}: {mean_score:.4f} (±{std_score:.4f})")