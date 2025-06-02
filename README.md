# car-model-classification

1. Baseline: Resnet18
2. Resnet18 -> EfficientNet-B3
3. EfficientNet-B3 / img_size = 300 / Dropout=0.3
4. EfficientNet-V2-S
5. EfficientNet-V2-M / img_size=384
6. EfficientNet-V2-M / img_size=384 / label_smoothing=0.1
7. EfficientNet-V2-M / img_size=384 / scheduler = CosineAnnealingLR => 실수로 1epoch에서 min scheduler로 수렴
8. EfficientNet-V2-M / img_size=384 / scheduler = CosineAnnealingLR / EPOCH = 25
9. EfficientNet-V2-M / img_size=384 / scheduler = CosineAnnealingLR / EPOCH = 25 / log_loss, val_acc, val_loss 기준
