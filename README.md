# car-model-classification

1. Baseline: Resnet18
   0.4284
2. Resnet18 -> EfficientNet-B3
   0.4283
3. efficientnet-b3 / img_size = 300 / Dropout=0.3
   0.6022
4. efficientnetv2_rw_s
   0.3677
5. efficientnetv2_rw_m / img_size=384
   0.3003
6. efficientnetv2_rw_m / img_size=384 / label_smoothing=0.1
   0.3502
7. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR => 실수로 1epoch에서 min scheduler로 수렴
   0.2579
8. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 25
   0.3082
9. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_logloss
   0.2671 
10. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_val_acc
    0.3208
11. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_val_acc
    0.3877
12. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_logloss
    0.3135
13. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR / EPOCH = 10
    transforms.Resize((CFG['IMG_SIZE'] + 32, CFG['IMG_SIZE'] + 32)),  # 약간 크게 리사이즈 후
    transforms.RandomResizedCrop(CFG['IMG_SIZE'], scale=(0.8, 1.0)),  # 랜덤 크롭
    transforms.RandomHorizontalFlip(),                                # 좌우 뒤집기
    transforms.RandomRotation(10),                                    # ±10도 회전
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변형
    0.2194
14. 13 -> MixUp
    0.1956
15. 14 -> stratifiedkfold 5 -> 에러로 4까지만 학습, 4개 앙상블
    0.1723
16. 15 -> 5crop TTA 적용
    0.1762
