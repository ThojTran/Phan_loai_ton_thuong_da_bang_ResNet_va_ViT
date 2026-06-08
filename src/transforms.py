"""
transforms.py — Transform cho từng mô hình.
PH04: ResNet18 → get_resnet_transforms()
PH05: ViT      → get_vit_transforms()  (thêm vào đây sau, không sửa file khác)
"""

from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # ResNet18 và ViT-Base đều dùng 224x224

# function get_resnet_transforms() được định nghĩa.
def get_resnet_transforms(mode: str):
    """
    mode='train' : augmentation để giảm overfitting (HAM10000 imbalanced)
    mode='val'   : chỉ resize + normalize, không augment
    """
    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),       # lật ngang ngẫu nhiên
            transforms.RandomVerticalFlip(),         # lật dọc (hợp lệ với ảnh da liễu)
            transforms.RandomRotation(15),           # xoay ±15 độ
            transforms.ColorJitter(                  # thay đổi màu sắc nhẹ
                brightness=0.2, contrast=0.2, saturation=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif mode == "derm_train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

# function get_vit_transforms() được định nghĩa.
def get_vit_transforms(mode: str):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # ViT dùng Resize cố định, không RandomResizedCrop
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    elif mode == "derm_train":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def get_vit_transforms(mode: str):
    """ViT và ResNet đều dùng 224x224 chuẩn ImageNet transform."""
    return get_resnet_transforms(mode)