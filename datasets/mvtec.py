import os
from glob import glob

import torch.utils.data
from PIL import Image
from torchvision import transforms

mvtec_categories = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

class MVTecDataset(torch.utils.data.Dataset):
  
    def __init__(self, root, category_index, input_size=224, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )
        if is_train:
            self.image_files = glob(
                os.path.join(root, mvtec_categories[category_index], "train", "good", "*.png")
            )
        else:
            self.image_files = glob(os.path.join(root, mvtec_categories[category_index], "test", "*", "*.png"))
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            label = None
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
                label = 0
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                )
                target = self.target_transform(target)
                label = 1
            return image, target, label

    def __len__(self):
        return len(self.image_files)