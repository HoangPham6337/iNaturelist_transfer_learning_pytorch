import torch
from PIL import Image
from typing import Tuple, List, Callable
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from pipeline.preprocessing import ColorDistorter, CentralCropResize
from dataset_builder.core.utility import load_manifest_parquet

class CustomDataset(Dataset):
    def __init__(self, data_path: str, train: bool=True, img_size:Tuple[int, int]=(224, 224)):
        self.image_label_with_correct_labels: List[Tuple[str, int]] = load_manifest_parquet(data_path)
        self.train = train
        self.img_size = img_size


    def __len__(self) -> int:
        return len(self.image_label_with_correct_labels)


    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img_path, label = self.image_label_with_correct_labels[index]
        image = Image.open(img_path).convert("RGB")

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        color_ordering = worker_id % 4

        if self.train:
            transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.05, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                ColorDistorter(ordering=color_ordering),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            transform: Callable[[Image.Image], torch.Tensor] = transforms.Compose([
                CentralCropResize(central_fraction=0.875, size=self.img_size)
            ])

        image = transform(image)

        return image, label  # type: ignore