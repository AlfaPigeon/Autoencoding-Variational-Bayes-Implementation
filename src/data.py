from huggingface_hub import hf_hub_download
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

# Download ZIP
zip_path = hf_hub_download(
    repo_id="Mayank022/Cropped_Face_Dataset_128x128",
    repo_type="dataset",
    filename="output.zip"
)

# Extract
data_dir = Path("data")

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

class FaceDataset(Dataset):

    def __init__(self, root):

        self.images = list(Path(root).rglob("*.jpg"))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5, 0.5, 0.5],
                [0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(self.images[index]).convert("RGB")

        return self.transform(image)

def GetFaceDataLoader(batch_size=32, shuffle=True, num_workers=0):
    dataset = FaceDataset("data")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

def GetFaceTrainTestDataLoaders(train_size, test_size, batch_size=32, shuffle=True, num_workers=0):
    dataset = FaceDataset("data")
    remainder = len(dataset) - train_size - test_size
    if remainder < 0:
        raise ValueError(
            f"train_size + test_size ({train_size + test_size}) exceeds dataset size ({len(dataset)})"
        )
    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        dataset, [train_size, test_size, remainder]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return train_loader, test_loader