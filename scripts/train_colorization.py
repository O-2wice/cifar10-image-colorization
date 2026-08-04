"""Train CIFAR-10 grayscale-to-RGB colorization models.

This script mirrors the notebook experiment: a shallow convolutional colorizer
is compared with a single-layer fully connected baseline. It keeps the long
training run visible in the terminal and writes reusable artifacts for the
Quarto report.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets


SEED = 42


class GrayscaleToColorDataset(Dataset):
    """Return grayscale input and RGB target pairs from a base image dataset."""

    def __init__(self, dataset, gray_transform, color_transform):
        self.dataset = dataset
        self.gray_transform = gray_transform
        self.color_transform = color_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        gray_image = self.gray_transform(image)
        color_image = self.color_transform(image)
        return gray_image, color_image


class ColorizationCNN(nn.Module):
    """Shallow convolutional colorizer used in the project notebook."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        return x


class ColorizationLinear(nn.Module):
    """Single linear baseline from flattened grayscale pixels to RGB pixels."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32, 3 * 32 * 32)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = x.view(x.size(0), 3, 32, 32)
        return torch.sigmoid(x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def set_reproducibility(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_loaders(data_dir, batch_size, validation_size, fast_dev_run=False):
    # CIFAR-10 is natively 32x32, so no resize is needed. No Normalize() either:
    # both models end in sigmoid and can only emit [0, 1], which is exactly the
    # range ToTensor() already produces.
    color_transform = transforms.ToTensor()
    gray_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_base = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_base = datasets.CIFAR10(root=data_dir, train=False, download=True)

    full_train_dataset = GrayscaleToColorDataset(train_base, gray_transform, color_transform)
    test_dataset = GrayscaleToColorDataset(test_base, gray_transform, color_transform)

    if fast_dev_run:
        full_train_dataset = Subset(full_train_dataset, range(1024))
        test_dataset = Subset(test_dataset, range(256))

    validation_size = min(validation_size, max(1, len(full_train_dataset) // 10))
    training_size = len(full_train_dataset) - validation_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for gray_images, color_images in loader:
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)
            predictions = model(gray_images)
            loss = criterion(predictions, color_images)
            total_loss += loss.item() * gray_images.size(0)
            total_samples += gray_images.size(0)

    return total_loss / total_samples


def train_model(
    name,
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    epochs,
    checkpoint_path,
    patience,
    progress_every,
):
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        seen_samples = 0
        epoch_start = time.time()

        for batch_idx, (gray_images, color_images) in enumerate(train_loader, start=1):
            gray_images = gray_images.to(device)
            color_images = color_images.to(device)

            optimizer.zero_grad()
            predictions = model(gray_images)
            loss = criterion(predictions, color_images)
            loss.backward()
            optimizer.step()

            # Weight by batch size to match evaluate_loss, so the train and
            # validation curves are measured the same way.
            running_loss += loss.item() * gray_images.size(0)
            seen_samples += gray_images.size(0)

            if batch_idx == 1 or batch_idx % progress_every == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - epoch_start
                print(
                    f"{name} epoch {epoch + 1:02d}/{epochs} "
                    f"batch {batch_idx:04d}/{len(train_loader)} "
                    f"loss {loss.item():.4f} elapsed {elapsed / 60:.1f} min",
                    flush=True,
                )

        train_loss = running_loss / seen_samples
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"{name} epoch {epoch + 1:02d}/{epochs} complete | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"{name} checkpoint saved to {checkpoint_path}", flush=True)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"{name} early stop counter: {early_stop_counter}/{patience}", flush=True)

        if early_stop_counter >= patience:
            print(f"{name} early stopping triggered.", flush=True)
            break

    return history


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def main():
    args = parse_args()
    set_reproducibility()

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dir = args.output_dir / "models"
    metric_dir = args.output_dir / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    cnn_path = model_dir / "best_model_cnn.pth"
    fcn_path = model_dir / "best_model_fcn.pth"
    history_path = metric_dir / "training_history.json"
    results_path = metric_dir / "test_results.json"

    print(f"Using device: {device}", flush=True)
    print(f"Using torch threads: {torch.get_num_threads()}", flush=True)

    if (
        not args.force_retrain
        and cnn_path.exists()
        and fcn_path.exists()
        and history_path.exists()
        and results_path.exists()
    ):
        print("Existing training artifacts found. Use --force-retrain to rerun.", flush=True)
        return

    train_loader, val_loader, test_loader = build_loaders(
        args.data_dir,
        args.batch_size,
        args.validation_size,
        fast_dev_run=args.fast_dev_run,
    )
    print(f"Training samples: {len(train_loader.dataset):,}", flush=True)
    print(f"Validation samples: {len(val_loader.dataset):,}", flush=True)
    print(f"Test samples: {len(test_loader.dataset):,}", flush=True)

    criterion = nn.MSELoss()

    cnn_model = ColorizationCNN().to(device)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=args.learning_rate)
    cnn_history = train_model(
        "CNN",
        cnn_model,
        cnn_optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        args.epochs,
        cnn_path,
        args.patience,
        args.progress_every,
    )

    fcn_model = ColorizationLinear().to(device)
    fcn_optimizer = optim.Adam(fcn_model.parameters(), lr=args.learning_rate)
    fcn_history = train_model(
        "FCN",
        fcn_model,
        fcn_optimizer,
        criterion,
        train_loader,
        val_loader,
        device,
        args.epochs,
        fcn_path,
        args.patience,
        args.progress_every,
    )

    write_json(history_path, {"cnn": cnn_history, "fcn": fcn_history})

    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    fcn_model.load_state_dict(torch.load(fcn_path, map_location=device))

    results = {
        "cnn_test_loss": evaluate_loss(cnn_model, test_loader, criterion, device),
        "fcn_test_loss": evaluate_loss(fcn_model, test_loader, criterion, device),
    }
    write_json(results_path, results)

    print("Final test results:", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()