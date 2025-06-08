# car-model-classification

1. Baseline: Resnet18  
   0.4284

2. Resnet18 -> EfficientNet-B3  
   0.4283  
   효과 없음

3. efficientnet-b3 / img_size = 300 / Dropout=0.3  
   0.6022  
   드롭아웃 적용 => 오히려 성능 저하

4. efficientnetv2_rw_s  
   0.3677  
   성능 향상

5. efficientnetv2_rw_m / img_size=384  
   0.3003   
   성능 더욱 향상

6. efficientnetv2_rw_m / img_size=384 / label_smoothing=0.1  
    0.3502   
    라벨 스무딩 적용: 성능 하락

7. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR => 실수로 1epoch에서 min scheduler로 수렴  
    0.2579   
    스케쥴러 적용하여 학습률 조정시 성능 향상

8. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1) / EPOCH = 25  
    0.3082   
    ReduceLROnPlauteau 올바른 위치에 스케쥴러 적용 시 성능 향상되지 않음

9. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)  
    / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_logloss  
    0.2671   
    logloss 기준이 더 성능 좋음

10. efficientnetv2_rw_m / img_size=384 / scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)  
    / EPOCH = 20 / log_loss, val_acc, val_loss 기준 / best_val_acc  
    0.3208   
    val_acc 기준 성능 안좋음

11. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20  
    / log_loss, val_acc, val_loss 기준 / best_val_acc  
    0.3877   
    tf_efficientnet_b4_ns 모델 사용 성능 안좋음

12. tf_efficientnet_b4_ns / img_size=380 / EPOCH = 20  
    / log_loss, val_acc, val_loss 기준 / best_logloss  
    0.3135   
    tf_efficientnet_b4_ns 모델 사용 성능 안좋음, logloss 기준이 더 좋음

13. efficientnetv2_rw_m / img_size=384 / scheduler = CosineAnnealingLR / EPOCH = 10  
    transforms.Resize((CFG['IMG_SIZE'] + 32, CFG['IMG_SIZE'] + 32)),  # 약간 크게 리사이즈 후  
    transforms.RandomResizedCrop(CFG['IMG_SIZE'], scale=(0.8, 1.0)),  # 랜덤 크롭  
    transforms.RandomHorizontalFlip(),                                # 좌우 뒤집기  
    transforms.RandomRotation(10),                                    # ±10도 회전  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 색상 변형  
    0.2194   
    증강 적용 시 성능 향상

14. 13 -> MixUp  
    0.1956   
    믹스업 적용 시 성능 향상

15. 14 -> stratifiedkfold 5  
    -> 에러로 4까지만 학습, 4개 앙상블  
    0.1723   
    stratifiedkfold 적용 시 성능 향상

16. 15 -> 5crop TTA 적용  
    0.1762   
    5crop TTA 적용 시 성능 향상 없음

17. 14 -> EPOCH = 12 / optimizer = AdamW   
    0.1870

18. 17 + Albumentation   
    train_transform = A.Compose([
    A.Resize(height=CFG['IMG_SIZE'] + 32, width=CFG['IMG_SIZE'] + 32),
    A.RandomResizedCrop(size=(CFG['IMG_SIZE'], CFG['IMG_SIZE']),
                        scale=(0.8, 1.0), ratio=(0.75, 1.33), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.1),
    ], p=0.3),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.3),
        A.HueSaturationValue(p=0.3),
    ], p=0.5),
    A.CoarseDropout(max_holes=1, max_height=CFG['IMG_SIZE']//5, max_width=CFG['IMG_SIZE']//5, p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])   
   0.2053

19. 17 -> MixUp -> Cutmix -> EPOCH = 12   
    0.1782   

20. 19 -> model change: convnext_base_384_in22ft1k   
    0.1582

21. 20 -> filter 6 noises   
    0.1584

22. 20 -> Cutmix 제거
    0.1774
    
#### 노이즈:
7시리즈_G11_2016_2018_0040.jpg
GLE_클래스_W167_2019_2024_0068.jpg
SM7_뉴아트_2008_2011_0053.jpg
아반떼_N_2022_2023_0064.jpg
프리우스_4세대_2019_2022_0052.jpg
아반떼_N_2022_2023_0035.jpg

#### 노이즈 후보:
박스터_718_2017_2024_0011.jpg
더_기아_레이_EV_2024_2025_0078.jpg
