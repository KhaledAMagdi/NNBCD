import os
import time
import random
import torch
import pandas as pd
from DeepLearning_Main import Run_Model

def generate_split_manifest(root_dir, save_path="dataset_splits.xlsx"):
    print("------------------------------------")
    print("Generating new manifest....")
    data = []

    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                parts = os.path.normpath(file_path).split(os.sep)

                label_str = parts[-6]
                tumor_type = parts[-4]
                patient_id = parts[-3]
                magnification = parts[-2]

                data.append({
                    'image_path': file_path,
                    'patient_id': patient_id,
                    'tumor_type': tumor_type,
                    'magnification': magnification,
                    'label': 0 if label_str == 'benign' else 1
                })

    df = pd.DataFrame(data)
    patient_labels = df.groupby('patient_id')['label'].first()

    benign_patients    = patient_labels[patient_labels == 0].index.tolist()
    malignant_patients = patient_labels[patient_labels == 1].index.tolist()

    random.seed(42)
    random.shuffle(benign_patients)
    random.shuffle(malignant_patients)

    def split_patients(patient_list):
        n = len(patient_list)
        train_end = int(n * 0.70)
        val_end   = train_end + int(n * 0.15)

        train_set = set(patient_list[:train_end])
        val_set   = set(patient_list[train_end:val_end])
        test_set  = set(patient_list[val_end:])

        return train_set, val_set, test_set

    b_train, b_val, b_test = split_patients(benign_patients)
    m_train, m_val, m_test = split_patients(malignant_patients)

    train_patients = b_train.union(m_train)
    val_patients   = b_val.union(m_val)

    def assign_split(pid):
        if pid in train_patients: return 'train'
        if pid in val_patients: return 'val'
        return 'test'

    df['split'] = df['patient_id'].apply(assign_split)

    train_patients = df[df['split'] == 'train']['patient_id'].nunique()
    val_patients   = df[df['split'] == 'val']['patient_id'].nunique()
    test_patients  = df[df['split'] == 'test']['patient_id'].nunique()

    print(f"Total Patient Split -> Train: {train_patients} | "
          f"Val: {val_patients} | Test: {test_patients}")
    print("Image Level Summary:")
    print(df.groupby(['split', 'label']).size().unstack(fill_value=0))

    df.to_excel(save_path, index=False)
    print(f"\nManifest saved to: {save_path}")


def load_BreaKHis_manifest(manifest_path):
    print("Loading Dataset Manifest...")
    try:
        df = pd.read_excel(manifest_path)
        train_patients = df[df['split'] == 'train']['patient_id'].nunique()
        val_patients   = df[df['split'] == 'val']['patient_id'].nunique()
        test_patients  = df[df['split'] == 'test']['patient_id'].nunique()
        print(f"Total Patient Split -> Train: {train_patients} | "
              f"Val: {val_patients} | Test: {test_patients}")
        print("Image Level Summary:")
        print(df.groupby(['split', 'label']).size().unstack(fill_value=0))
        print("------------------------------------")
        return df
    except FileNotFoundError:
        print("Dataset Manifest not found.")
        generate_split_manifest("G:/datasets/BreaKHis_v1", manifest_path)
        print("------------------------------------")
        return pd.read_excel(manifest_path)


def main():
    #--------------------------------------------------------------------------------------------------#

    print("------------------------------------")
    print("--- Deep Learning Model Training ---")
    print("------------------------------------")

    #--------------------------------------------------------------------------------------------------#
    #---System Check---#

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"-> System Check: Using {device.type.upper()}")
    print("------------------------------------")

    #--------------------------------------------------------------------------------------------------#
    #---Data Preparation---#

    df = load_BreaKHis_manifest("dataset_splits.xlsx")

    #--------------------------------------------------------------------------------------------------#
    #---DenseNet---#

    # print("---------- DenseNet Model ----------")
    # pretrained_model_path = "Models/DenseNetModels/Densenet_x.pth" # or None if training a new model
    # save_path = "Models/DenseNetModels/Densenet_x"
    # # path without ".pth", as in training we save 3 models, path gets appended with "_auc/f1/loss.pth"
    # name = 'densenet'
    # Run_Model(df.copy(), device, name, pretrained_model_path, save_path)
    # print("------------------------------------")

    #--------------------------------------------------------------------------------------------------#
    #---ResNet---#

    # print("----------- ResNet Model -----------")
    # pretrained_model_path  = "Models/ResNetModels/Resnet_x.pth"
    # save_path = "Models/ResNetModels/Resnet_x"
    # name = 'resnet'
    # Run_Model(df.copy(), device, name, pretrained_model_path, save_path)
    # print("------------------------------------")

    #--------------------------------------------------------------------------------------------------#
    # ---Inception---#

    print("---------- Inception Model ---------")
    pretrained_model_path = None
    save_path = "Models/InceptionModels/Inception_v67_Cla"
    name = 'inception'
    Run_Model(df.copy(), device, name, pretrained_model_path, save_path)
    print("------------------------------------")

    #--------------------------------------------------------------------------------------------------#

if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("\n" + "-" * 36)
    print(f"Elapsed time: {end - start:.6f} seconds")
    print("-" * 36)