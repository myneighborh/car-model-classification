import torch
import torch.nn as nn
import timm


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = timm.create_model(
            'convnext_base_384_in22ft1k',
            pretrained=True,
            features_only=False
        )
        self.feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.head(x)
        return x


def build_model(model_path, num_classes, device):
    model = BaseModel(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model
