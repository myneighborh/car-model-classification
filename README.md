# car-model-classification

1. Baseline: Resnet18
2. Resnet18 -> EfficientNet-B3
3. efficientnet-b3 / img_size = 300 / Dropout=0.3
4. efficientnetv2_rw_s
5. efficientnetv2_rw_m / img_size=384
6. efficientnetv2_rw_m / img_size=384 / label_smoothing=0.1
7. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR => 실수로 1epoch에서 min scheduler로 수렴
8. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 25
9. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_logloss
10. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_val_acc
11. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_val_acc
12. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_logloss
13. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR / EPOCH = 10 / log_loss, val_acc, val_loss 기준 / best_val_acc
14. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR /
    transforms.Resize((CFG['IMG_SIZE'] + 32, CFG['IMG_SIZE'] + 32)),  # 약간 크게 리사이즈 후
    transforms.RandomResizedCrop(CFG['IMG_SIZE'], scale=(0.8, 1.0)),  # 랜덤 크롭
    transforms.RandomHorizontalFlip(),                                # 좌우 뒤집기
    transforms.RandomRotation(10),                                    # ±10도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변형
15. 14 -> MixUp
16. 14 -> MixUp -> stratifiedkfold 5 -> 에러로 4까지만 학습, 4개 앙상블
17. 16 -> 5crop TTA 적용
