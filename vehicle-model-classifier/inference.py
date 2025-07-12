import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from huggingface_hub import hf_hub_download

from model_utils import BaseModel


device = None
class_names = None
models = None
transform = None
examples = None


def main():
    global device, class_names, models, transform, examples

    IMG_SIZE = 384
    MODEL_NAMES = ["fold1.pth", "fold2.pth", "fold3.pth"]
    REPO_ID = "myneighborh/vehicle-model-classifier"
    MODEL_DIR = "model"
    SAMPLE_DIR = "sample"

    device, class_names = setup_environment()
    download_models(MODEL_NAMES, MODEL_DIR, REPO_ID)
    models = load_models(MODEL_NAMES, MODEL_DIR, len(class_names), device)
    transform = get_transform(IMG_SIZE)
    examples = load_examples(SAMPLE_DIR)


def setup_environment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    submission_df = pd.read_csv("class_list.csv")
    class_names = submission_df.columns.tolist()
    return device, class_names


def download_models(model_names, model_dir, repo_id):
    os.makedirs(model_dir, exist_ok=True)
    for name in model_names:
        local_path = os.path.join(model_dir, name)
        if not os.path.exists(local_path):
            print(f"Downloading {name} from HF hub...")
            hf_hub_download(repo_id=repo_id, filename=name, local_dir=model_dir, local_dir_use_symlinks=False)
            print(f"Downloaded {name}")


def load_models(model_names, model_dir, num_classes, device):
    models = []
    for model_file in model_names:
        model = BaseModel(num_classes=num_classes)
        path = os.path.join(model_dir, model_file)
        checkpoint = torch.load(path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        models.append(model)
    return models


def get_transform(img_size):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def predict(image: Image.Image, actual_label: str):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    ensemble_probs = []
    for model in models:
        with torch.no_grad():
            output = model(input_tensor)
            prob = F.softmax(output, dim=1)
            ensemble_probs.append(prob.cpu())

    final_probs = torch.stack(ensemble_probs).mean(dim=0).squeeze()
    pred_label = class_names[final_probs.argmax().item()]
    return f"Predicted: {pred_label}  |  Actual: {actual_label}"


def load_examples(train_dir):
    examples = []
    for class_folder in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_folder)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(class_path, file)
                    examples.append([image_path, class_folder])
                    break
    return examples


if __name__ == '__main__':
    main()