import os
import time
import torch
import pandas as pd
from DenseNet_Model import DenseNet_Model
from Inception_Model import Inception_Model
from ResNet_Model import ResNet_Model


def build_BreaKHis_dataframe(root_dir):
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)

                parts = os.path.normpath(file_path).split(os.sep)

                label_str = parts[-6]
                tumor_type = parts[-4]
                patient_id = parts[-3]
                magnification = parts[-2]

                label = 0 if label_str == 'benign' else 1

                data.append({
                    'image_path': file_path,
                    'patient_id': patient_id,
                    'tumor_type': tumor_type,
                    'magnification': magnification,
                    'label': label
                })

    df = pd.DataFrame(data)
    total_benign_images = df[df['label'] == 0].shape[0]
    total_malignant_images = df[df['label'] == 1].shape[0]
    total_benign_patients = df[df['label'] == 0]['patient_id'].nunique()
    total_malignant_patients = df[df['label'] == 1]['patient_id'].nunique()

    print(f"\nBenign Images: {total_benign_images} | "
          f"Malignant Images: {total_malignant_images}")
    print(f"Benign Patients: {total_benign_patients} | "
          f"Malignant Patients: {total_malignant_patients}")

    return df


def main():
    #-------------------------------------------------#
    #---System Check---#
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Check: Using {device.type.upper()}")

    #-------------------------------------------------#
    #---Data Preparation---#
    dataset_path = "BreaKHis_v1"
    df = build_BreaKHis_dataframe(dataset_path)

    #-------------------------------------------------#
    #---DenseNet---#
    pretrained_model_path = "DenseNetModels/MultiFactorTrackers/Densenet_Cla_TTAFocalSampler_v2_loss.pth"
    save_path = "DenseNetModels/Densenet_vxx" #save path without ".pth" as it auto assigns
    DenseNet_Model(df.copy(), device, pretrained_model_path, save_path)

    #-----------------------------------------------------------------------------------------------------------------#
    #---ResNet---#
    pretrained_model_path  = "ResNetModels/Resnet_Cla_v0_auc.pth"
    save_path = "ResNetModels/Resnet_0_ClaL4_v0_auc"
    ResNet_Model(df.copy(), device, pretrained_model_path, save_path)

    #-----------------------------------------------------------------------------------------------------------------#
    # ---Inception---#
    pretrained_model_path = "InceptionModels/Inception_x.pth"
    save_path = "InceptionModels/Inception_x"
    Inception_Model(df.copy(), device, pretrained_model_path, save_path)

    # -----------------------------------------------------------------------------------------------------------------#


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    end = time.perf_counter()
    print("\n" + "-" * 30)
    print(f"Elapsed time: {end - start:.6f} seconds")
    print("-" * 30)