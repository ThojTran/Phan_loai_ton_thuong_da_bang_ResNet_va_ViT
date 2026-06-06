"""
transforms.py — Transform cho từng mô hình.
PH04: ResNet18 → get_resnet_transforms()
PH05: ViT      → get_vit_transforms()  (thêm vào đây sau, không sửa file khác)
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # ResNet18 và ViT-Base đều dùng 224x224


def get_resnet_transforms(mode: str):
    """
    mode='train' : augmentation để giảm overfitting (HAM10000 imbalanced)
    mode='val'   : chỉ resize + normalize, không augment
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),       # lật ngang ngẫu nhiên
            transforms.RandomVerticalFlip(),         # lật dọc (hợp lệ với ảnh da liễu)
            transforms.RandomRotation(15),           # xoay ±15 độ
            transforms.ColorJitter(                  # thay đổi màu sắc nhẹ
                brightness=0.2, contrast=0.2, saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_vit_transforms(mode: str):
    """ViT và ResNet đều dùng 224x224 chuẩn ImageNet transform."""
    return get_resnet_transforms(mode)