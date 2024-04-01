import os
import torch
import torchvision
from torch.utils.data import DataLoader
import wandb
import numpy as np

class OneClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset=None, split="train", transform=None):
        self.dataset = dataset
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.dataset[self.split])

    def __getitem__(self, idx):
        sample = self.dataset[self.split][idx]
        image = sample['image']
        mask = sample['label']
        image = np.array(image.convert('RGB'))
        mask = np.array(mask.convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    # Create the 'checkpoints' directory if it doesn't exist
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")

    # Save the checkpoint to the 'checkpoints' directory
    torch.save(state, os.path.join("checkpoints", filename))

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch

def get_loaders(
    dataset,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = OneClassDataset(
        dataset=dataset,
        split="train",
        transform=train_transform,
    )

    # # Use only the first 100 samples of the dataset
    # train_ds = torch.utils.data.Subset(train_ds, range(100))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = OneClassDataset(
        dataset=dataset,
        split="validation",
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = OneClassDataset(
        dataset=dataset,
        split="test",
        transform=val_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            iou_score += iou_pytorch(preds, y)  # Calculate IoU score for the batch

    pixel_accuracy = num_correct / num_pixels * 100
    print(f"dice_score: {dice_score}")
    dice_score = dice_score / len(loader)
    print(f"Got {num_correct}/{num_pixels} with acc {pixel_accuracy:.2f}")
    wandb.log({"Pixel Accuracy": pixel_accuracy})
    print(f"Dice score: {dice_score}")
    wandb.log({"Dice Score": dice_score})
    print(f"IoU score: {iou_score}")  # Print IoU score to the console
    wandb.log({"IoU Score": iou_score})
    model.train()

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # Convert tensors to 'Bool' type
    outputs_int = outputs.int()
    labels_int = labels.int()

    intersection = (outputs_int & labels_int).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs_int | labels_int).float().sum((1, 2))  # Will be zero if both are 0

    iou = (intersection + 1e-6) / (union + 1e-6)  # We smooth our division to avoid 0/0

    return iou.mean()  # Average over the batch


def save_predictions_as_imgs(
    loader, model, folder="saved_images", device="cuda"
):

    # Create directory if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            # print(f"pred_shape: {preds.shape}")
            # print(f"y_shape: {y.shape}")

        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/{idx}.png")

    model.train()