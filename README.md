# Vehicle Model Classification

The objective was to develop a deep learning model that classifies 696 actual used car models from real-world vehicle images.
To enhance performance, I experimented with various backbones, optimizers, learning rate schedulers, data augmentations, and ensemble techniques.
The final model was implemented in PyTorch using ConvNeXt as the backbone, MixUp/CutMix for augmentation, and a Stratified K-Fold ensemble with Test-Time Augmentation (TTA).

## Model Demo
https://huggingface.co/spaces/myneighborh/vehicle-model-classifier
![output](https://github.com/user-attachments/assets/8d85ad35-b6bc-4579-b1ad-177feba249e5)

## Competition Results

| Leaderboard         | Log Loss | Rank        |
|:-------------------:|:--------:|:-----------:|
| Public  | 0.14351   | 76 / 749    |
| Private | 0.14099   | 75 / 748    |

## Experiment Log

1. **Baseline: ResNet18**  
   - 0.4284

2. **ResNet18 → EfficientNet-B3**  
   - 0.4283  
   - No improvement

3. **EfficientNet-B3 / img_size = 300 / Dropout = 0.3**  
   - 0.6022  
   - Applying dropout degraded performance

4. **EfficientNetV2-RW-S**  
   - 0.3677  
   - Improved performance

5. **EfficientNetV2-RW-M / img_size = 384**  
   - 0.3003  
   - Further improvement

6. **EfficientNetV2-RW-M / img_size = 384 / label_smoothing = 0.1**  
   - 0.3502  
   - Applying label smoothing reduced performance

7. **CosineAnnealingLR Scheduler (accidentally converged at min in 1 epoch)**  
   - 0.2579  
   - Significant improvement due to early LR drop

8. **ReduceLROnPlateau / EPOCH = 25**  
   - 0.3082  
   - No improvement even with correct placement

9. **ReduceLROnPlateau / EPOCH = 20 / Best based on log_loss**  
   - 0.2671  
   - Better performance than val_acc criterion

10. **ReduceLROnPlateau / EPOCH = 20 / Best based on val_acc**  
    - 0.3208  
    - Worse performance with val_acc criterion

11. **tf_efficientnet_b4_ns / img_size = 380 / EPOCH = 20 / val_acc**  
    - 0.3877  
    - Underperformed

12. **tf_efficientnet_b4_ns / img_size = 380 / EPOCH = 20 / log_loss**  
    - 0.3135  
    - Slightly better but still underperformed

13. **EfficientNetV2-RW-M / CosineAnnealingLR / EPOCH = 10**  
    - Transforms: Resize(+32) → RandomResizedCrop → Flip → Rotate → ColorJitter  
    - 0.2194  
    - Data augmentation significantly improved performance

14. **+ MixUp**  
    - 0.1956  
    - Further improvement

15. **+ Stratified K-Fold (trained 4 folds, ensemble of 4)**  
    - 0.1723  
    - Performance boost with K-Fold

16. **+ 5Crop TTA**  
    - 0.1762  
    - No improvement from 5Crop TTA

17. **EPOCH = 12 / Optimizer = AdamW**  
    - 0.1870

18. **+ Albumentations**  
    - Pipeline: Resize → RandomResizedCrop → Flip → Rotate → ColorJitter  
    - OneOf(blur), OneOf(contrast), CoarseDropout, Normalize  
    - 0.2053  
    - Performance slightly dropped

19. **+ MixUp → CutMix / EPOCH = 12**  
    - 0.1782

20. **Model change → convnext_base_384_in22ft1k**  
    - 0.1582

21. **+ Removed 6 noisy images**  
    - 0.1584

22. **- Removed CutMix**  
    - 0.1774

23. **MixUp α = 0.2, CutMix α = 1.0**  
    - 0.1566

24. **Same config retraining**  
    - 0.1507

25. **Increased EPOCH to 15**  
    - 0.1486

26. **Soft voting ensemble (folds 13~15)**  
    - 0.1493

27. **Logit ensemble (folds 13~15)**  
    - 0.1496

28. **Error occurred**

29. **TTA (3 views) + Soft Voting**  
    - 0.1468

30. **TTA (3 views) + Logit Ensemble**  
    - 0.1469

31. **EfficientNetV2-RW-M / test / EPOCH = 20**  
    - 0.1767

32. **ConvNeXt-Base / EPOCH = 20**  
    - 0.1520

33. **TTA (5 views)**  
    - 0.1471

34. **Stratified K-Fold (3 folds) + Soft Voting**  
    - 0.1436

35. **Retrained all models (class count error occurred)**  
    - 0.1486

36. **3-Fold + 3-TTA Ensemble**  
    - 0.1435

## Removed Noise Images

- 7시리즈_G11_2016_2018_0040.jpg  
- GLE_클래스_W167_2019_2024_0068.jpg  
- SM7_뉴아트_2008_2011_0053.jpg  
- 아반떼_N_2022_2023_0064.jpg  
- 프리우스_4세대_2019_2022_0052.jpg  
- 아반떼_N_2022_2023_0035.jpg
