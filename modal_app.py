import modal

app = modal.App("breakhis-classifier")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "gradio", "Pillow", "numpy", "fastapi", "uvicorn", "timm")
    .add_local_file("Models/DenseNetModels/Densenet_TH_T2_ClaD4N5_auc_f1_f1.pth", "/root/weights/densenet.pth")
    .add_local_file("Models/ResNetModels/Resnet_v1_T2_auc_f1_auc.pth", "/root/weights/resnet.pth")
    .add_local_file("Models/InceptionModels/Inception_v10_Cla_auc.pth", "/root/weights/inception.pth")
)

@app.function(image=image, min_containers=1, max_containers=1)
@modal.asgi_app()
def ui():
    import gradio as gr
    import torch
    import torch.nn as nn
    from torchvision import transforms, models
    from fastapi import FastAPI
    import timm

    web_app = FastAPI()

    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    THRESHOLD_D = 0.59
    THRESHOLD_R = 0.29
    THRESHOLD_I = 0.41

    def build_densenet(path):
        m = models.densenet169(weights=None)
        m.classifier = nn.Sequential(
            nn.Linear(m.classifier.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return m.to(device).eval()
    def build_resnet(path):
        m = models.resnet50(weights=None)
        m.fc = nn.Sequential(
            nn.Linear(m.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return m.to(device).eval()
    def build_inception(path):
        m = timm.create_model('inception_resnet_v2', pretrained=False)
        m.classif = nn.Sequential(
            nn.Linear(m.classif.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return m.to(device).eval()


    model_D = build_densenet("/root/weights/densenet.pth")
    model_R = build_resnet("/root/weights/resnet.pth")
    model_I = build_inception("/root/weights/inception.pth")

    def predict_all(image):
        pred_D =  predict_densnet(image)
        pred_R = predict_resnet(image)
        pred_I =predict_inception(image)

        return pred_D, pred_R, pred_I


    def predict_densnet(image):
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model_D(tensor)).item()
        label      = "Malignant 🔴" if prob >= THRESHOLD_D else "Benign 🟢"
        mal_pct    = round(prob * 100, 2)
        ben_pct    = round((1 - prob) * 100, 2)
        confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
        pred     = f"Prediction:  {label}\nMalignant:   {mal_pct}%\nBenign:      {ben_pct}%\nThreshold:   {THRESHOLD_D}\nConfidence:   {confidence}"
        return pred

    def predict_resnet(image):
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model_R(tensor)).item()
        label      = "Malignant 🔴" if prob >= THRESHOLD_R else "Benign 🟢"
        mal_pct    = round(prob * 100, 2)
        ben_pct    = round((1 - prob) * 100, 2)
        confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
        pred     = f"Prediction:  {label}\nMalignant:   {mal_pct}%\nBenign:      {ben_pct}%\nThreshold:   {THRESHOLD_R}\nConfidence:   {confidence}"
        return pred

    def predict_inception(image):
        tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model_I(tensor)).item()
        label      = "Malignant 🔴" if prob >= THRESHOLD_I else "Benign 🟢"
        mal_pct    = round(prob * 100, 2)
        ben_pct    = round((1 - prob) * 100, 2)
        confidence = "High" if abs(prob - 0.5) > 0.3 else "Medium" if abs(prob - 0.5) > 0.15 else "Low"
        pred     = f"Prediction:  {label}\nMalignant:   {mal_pct}%\nBenign:      {ben_pct}%\nThreshold:   {THRESHOLD_I}\nConfidence:   {confidence}"
        return pred

    with gr.Blocks(title="BreakHis Classifier") as demo:
        gr.Markdown("# 🔬 BreakHis Breast Cancer Classifier\nUpload a breast histology image to classify it as **Benign** or **Malignant**.")
        with gr.Row():
            img_in = gr.Image(type="pil", label="Upload Histology Image", height=350)
            btn    = gr.Button("Analyze", variant="primary", scale=0)
        gr.Markdown("### Result")
        with gr.Row():
            result_D = gr.Textbox(label="DenseNet-169 Prediction", lines=5)
            result_R = gr.Textbox(label="ResNet-50 Prediction", lines=5)
            result_I = gr.Textbox(label="InceptionResNetV2 Prediction", lines=5)
        btn.click(fn=predict_all, inputs=[img_in], outputs=[result_D, result_R, result_I], queue=False)

    return gr.mount_gradio_app(web_app, demo, path="/")
