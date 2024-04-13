import os
import torch
import torchvision
from torch.utils.data import DataLoader
import wandb
import numpy as np
import torch.nn.functional as F

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

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    Load model and optimizer from a given checkpoint\n
    returns model, optimizer, epoch of loaded checkpoint
    """
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"]) if optimizer else None
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

    # Use only the first 100 samples of the dataset
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

def check_accuracy(loader, model, device="cuda", loss_fn=None, mode="val", wandb=True, img_dims=(256, 256)):
    """To compute metrics after one training epoch

    Args:
        loader (pytorch DataLoader): train, val, test loaders
        model (pytorch model): model
        device (str, optional): Defaults to "cuda".
        loss_fn (pytorch loss fn, optional): To test for validation or test loss. Defaults to None.
        mode (str, optional): For wandb logging purposes to log the metrics with a different label for validation / test. \nPossible values ["val", "test"].\n Defaults to "val".
        wandb (bool, optional): To log the metrics to wandb. Defaults to True.
        img_dims (tuple, optional): Image dimensions that model being tested accepts. Defaults to (256, 256).
    """
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    val_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            if img_dims != (256, 256):
                # Resize to fit sota models
                x = F.interpolate(x, size=img_dims)
                y = F.interpolate(y, size=img_dims)

            preds = torch.sigmoid(model(x))

            # Track validation Loss
            if loss_fn:
                loss = loss_fn(preds, y)
                val_loss += loss.item()

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    avg_val_loss = val_loss / len(loader)
    pixel_accuracy = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader)

    if mode == "val":
        print(f'Validation Loss: {avg_val_loss}')
        wandb.log({"Validation Loss": avg_val_loss}) if wandb else None

        print(f"Got {num_correct}/{num_pixels} with acc {pixel_accuracy:.2f}")
        wandb.log({"Pixel Accuracy": pixel_accuracy}) if wandb else None

        print(f"Dice score: {dice_score}")
        wandb.log({"Dice Score": dice_score}) if wandb else None

    if mode == "test":
        # print(f"Test Loss: {avg_val_loss}")
        wandb.log({"Test Loss": avg_val_loss}) if wandb else None

        print(f"Got {num_correct}/{num_pixels} with acc {pixel_accuracy:.2f}")
        wandb.log({"Test Pixel Accuracy": pixel_accuracy}) if wandb else None

        print(f"Dice score: {dice_score}")
        wandb.log({"Test Dice Score": dice_score}) if wandb else None
    model.train()


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