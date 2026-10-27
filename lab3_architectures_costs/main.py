from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

try:
    from torchinfo import summary
except ImportError:
    summary = None

try:
    from thop import profile
except ImportError:
    profile = None


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3
NUM_WORKERS = 0  # reliable default for Windows/VS Code
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    """Flatten 784 -> Linear 256 -> ReLU -> Linear 128 -> ReLU -> Linear 10."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """The CNN architecture specified in the lab sheet."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class PatchAttention(nn.Module):
    """
    Split 28x28 images into sixteen 7x7 patches, project each 49-value patch
    to 64 dimensions, use two Transformer encoder layers, mean-pool, classify.
    No positional embedding is added because the lab specification does not
    request one.
    """
    def __init__(self):
        super().__init__()
        self.patch_projection = nn.Linear(7 * 7, 64)
        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        # B, 1, 28, 28 -> B, 1, 4, 7, 4, 7 -> B, 16, 49
        b = x.shape[0]
        x = x.reshape(b, 1, 4, 7, 4, 7)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(b, 16, 49)
        x = self.patch_projection(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


def build_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])
    train_full = datasets.FashionMNIST(
        root=ROOT / "data", train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        root=ROOT / "data", train=False, download=True, transform=transform
    )
    train_set, val_set = random_split(
        train_full, [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS}
    train_loader = DataLoader(train_set, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, criterion=None):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return total_loss / total, correct / total


def train_one_model(model, train_loader, val_loader, name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * y.size(0)
            seen += y.size(0)

        train_loss = running_loss / seen
        val_loss, val_accuracy = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"{name} | epoch {epoch}/{EPOCHS} | "
            f"train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
            f"val accuracy={val_accuracy:.4%}"
        )
    return model, history


def median_latency_ms(model, batch_size, runs):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=DEVICE)
    # Warm-up runs are excluded from measurement.
    with torch.inference_mode():
        for _ in range(10):
            _ = model(x)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(runs):
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(x)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
    return statistics.median(timings)


def manual_flops():
    # One representative layer of each requested type. FLOPs convention:
    # one multiply + one add = 2 FLOPs.
    return {
        "MLP first Linear (784 -> 256)": 2 * 784 * 256,
        "CNN first Conv2d (1 -> 16, 28x28)": 2 * 3 * 3 * 1 * 16 * 28 * 28,
        "Attention QKV projections (seq=16, d=64)": 2 * 16 * 64 * 3 * 64,
        "Attention score products (seq=16, d=64)": 2 * 16 * 16 * 64,
    }


def profile_thop(model):
    if profile is None:
        return {"available": False, "reason": "thop is not installed"}
    model = model.to(DEVICE).eval()
    sample = torch.randn(1, 1, 28, 28, device=DEVICE)
    try:
        with torch.no_grad():
            macs, params = profile(model, inputs=(sample,), verbose=False)
        return {
            "available": True,
            "macs_batch1": int(macs),
            "approx_flops_batch1_if_1mac_2flops": int(macs * 2),
            "params_thop": int(params),
            "note": "THOP usually reports MACs; approximate FLOPs shown as 2 x MACs.",
        }
    except Exception as exc:
        return {"available": False, "reason": f"THOP profiling failed: {exc}"}


def save_plots(results):
    names = list(results.keys())
    accuracies = [results[n]["test_accuracy"] * 100 for n in names]
    params = [results[n]["parameters"] for n in names]
    latency = [results[n]["latency_batch1_ms"] for n in names]

    plt.figure(figsize=(7, 5))
    for name, p, acc in zip(names, params, accuracies):
        plt.scatter(p, acc, s=75)
        plt.annotate(name, (p, acc), xytext=(6, 5), textcoords="offset points")
    plt.xlabel("Trainable parameters")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fashion-MNIST: accuracy vs parameter count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_parameters.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, lat, acc in zip(names, latency, accuracies):
        plt.scatter(lat, acc, s=75)
        plt.annotate(name, (lat, acc), xytext=(6, 5), textcoords="offset points")
    plt.xlabel("Batch-1 median inference latency (ms)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Fashion-MNIST: accuracy vs latency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_vs_latency.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name in names:
        plt.plot(
            range(1, EPOCHS + 1),
            [x * 100 for x in results[name]["history"]["val_accuracy"]],
            marker="o",
            label=name,
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy (%)")
    plt.title("Validation accuracy during training")
    plt.xticks(range(1, EPOCHS + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / "validation_accuracy_by_epoch.png", dpi=160)
    plt.close()


def write_report(results, manual):
    lines = [
        "# Lab 3: Architectures and their costs",
        "",
        "## Objective",
        "Implement comparable MLP, CNN, and attention architectures in PyTorch,",
        "measure parameter counts, FLOPs/MACs and inference latency, train for five",
        "epochs, and compare accuracy against computational cost.",
        "",
        "## Experimental setup",
        f"- Dataset: Fashion-MNIST; official training set split into 55,000 train and 5,000 validation examples; official 10,000-example test set.",
        "- Transform: `ToTensor()` then `Normalize((0.2860,), (0.3530,))`.",
        f"- Batch size: {BATCH_SIZE}; optimizer: Adam; learning rate: {LEARNING_RATE}; epochs: {EPOCHS}.",
        f"- Device used: `{DEVICE}`; random seed: {SEED}.",
        "- Inference latency is measured after 10 warm-up runs; median of 100 runs for batch 1 and 20 runs for batch 128.",
        "",
        "## Architecture and parameter counts",
        "| Model | Trainable parameters | Test accuracy | Batch-1 median (ms) | Batch-128 median (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['parameters']:,} | {r['test_accuracy']:.4%} | "
            f"{r['latency_batch1_ms']:.3f} | {r['latency_batch128_ms']:.3f} |"
        )
    lines += [
        "",
        "### Important specification note",
        "The handout asks for models targeting 100k–150k parameters but also gives fixed layer definitions.",
        "Those definitions do not all fall in that range: the prescribed MLP and CNN parameter counts differ substantially.",
        "This implementation follows the explicit layer definitions rather than silently changing them. Mention this conflict to the instructor if the target range is mandatory.",
        "",
        "## Hand-calculated representative FLOPs",
        "Convention used: one multiply-add is counted as 2 FLOPs.",
        "| Layer / operation | Calculation | FLOPs |",
        "|---|---|---:|",
        f"| MLP first Linear | 2 × 784 × 256 | {manual['MLP first Linear (784 -> 256)']:,} |",
        f"| CNN first Conv2d | 2 × 3 × 3 × 1 × 16 × 28 × 28 | {manual['CNN first Conv2d (1 -> 16, 28x28)']:,} |",
        f"| Attention QKV projections | 2 × 16 × 64 × 3 × 64 | {manual['Attention QKV projections (seq=16, d=64)']:,} |",
        f"| Attention score products | 2 × 16² × 64 | {manual['Attention score products (seq=16, d=64)']:,} |",
        "",
        "These are representative operations, not total-model FLOPs. Bias additions, activations, normalization, pooling and other operations are omitted from these hand calculations.",
        "",
        "## THOP comparison",
    ]
    for name, r in results.items():
        t = r["thop"]
        if t.get("available"):
            lines.append(
                f"- **{name}:** THOP reports {t['macs_batch1']:,} MACs for batch 1 "
                f"(approximately {t['approx_flops_batch1_if_1mac_2flops']:,} FLOPs under the 2-FLOPs-per-MAC convention); "
                f"THOP parameter count: {t['params_thop']:,}."
            )
        else:
            lines.append(f"- **{name}:** THOP unavailable — {t.get('reason', 'unknown reason')}.")
    lines += [
        "",
        "THOP commonly reports multiply-accumulate operations (MACs), not FLOPs. To compare consistently, convert using approximately 2 FLOPs per MAC. Differences from hand calculations are expected because the manual calculations cover only selected operations, while profilers may include or omit biases, normalization, pooling, activations, and unsupported operators. Transformer operator coverage can vary by profiler version.",
        "",
        "## Training results",
    ]
    for name, r in results.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| Epoch | Train loss | Validation loss | Validation accuracy |")
        lines.append("|---:|---:|---:|---:|")
        for i, (tl, vl, va) in enumerate(zip(
            r["history"]["train_loss"],
            r["history"]["val_loss"],
            r["history"]["val_accuracy"],
        ), start=1):
            lines.append(f"| {i} | {tl:.4f} | {vl:.4f} | {va:.4%} |")
        lines.append("")
    lines += [
        "## Plots",
        "- `results/accuracy_vs_parameters.png` — test accuracy against parameter count.",
        "- `results/accuracy_vs_latency.png` — test accuracy against batch-1 latency.",
        "- `results/validation_accuracy_by_epoch.png` — validation accuracy across epochs.",
        "",
        "## Discussion: access patterns and cost-effectiveness",
        "The MLP flattens the image, discarding explicit 2D neighborhood structure. Its dense matrix multiplications can be regular and hardware-friendly, but the specified hidden layers lead to a comparatively large parameter count.",
        "The CNN uses local receptive fields and weight sharing, exploiting image locality and translation-related patterns. Its parameter count is comparatively small, although convolution performance depends on implementation, memory layout, and hardware kernels.",
        "The attention model processes sixteen patches and uses pairwise token interactions. Its attention-score work grows quadratically with sequence length, while projections and feed-forward layers also contribute substantial cost. With only sixteen patches, the sequence is short, but Transformer operations may have higher runtime overhead than simpler layers.",
        "Cost-effectiveness should be judged from the measured test accuracy, parameter count, and latency together; the most accurate model is not automatically the fastest or smallest. Use the tables and plots above to state which model gives the best accuracy-cost trade-off on the machine used.",
        "",
        "## Reproducibility notes",
        "Results depend on PyTorch/torchvision versions, CPU/GPU hardware, and runtime conditions. The script records the device and seed. Latency values are local measurements and should not be compared across machines without noting hardware and software versions.",
        "",
        "## Conclusion",
        "Replace this paragraph with a concise conclusion based on the measured values above: identify the strongest test accuracy, smallest parameter count, lowest batch-1 latency, and the best overall trade-off.",
        "",
    ]
    (ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    set_seed()
    print(f"Device: {DEVICE}")
    print("Preparing Fashion-MNIST loaders (first run downloads the dataset)...")
    train_loader, val_loader, test_loader = build_loaders()

    models = {"MLP": MLP(), "CNN": CNN(), "Attention": PatchAttention()}
    results = {}
    for name, model in models.items():
        print(f"\n{'=' * 18} {name} {'=' * 18}")
        params = count_parameters(model)
        print(f"Trainable parameters: {params:,}")
        if summary is not None:
            try:
                print(summary(model, input_size=(1, 1, 28, 28), verbose=0))
            except Exception as exc:
                print(f"torchinfo.summary could not run: {exc}")
        else:
            print("torchinfo is not installed; using manual PyTorch parameter count.")

        model, history = train_one_model(model, train_loader, val_loader, name)
        test_loss, test_accuracy = evaluate(model, test_loader)
        print(f"{name} test accuracy: {test_accuracy:.4%}; test loss: {test_loss:.4f}")

        lat1 = median_latency_ms(model, batch_size=1, runs=100)
        lat128 = median_latency_ms(model, batch_size=128, runs=20)
        print(f"Median latency: batch 1={lat1:.3f} ms; batch 128={lat128:.3f} ms")

        results[name] = {
            "parameters": params,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "latency_batch1_ms": lat1,
            "latency_batch128_ms": lat128,
            "history": history,
            "thop": profile_thop(model),
        }
        torch.save(model.state_dict(), RESULTS / f"{name.lower()}_fashion_mnist.pt")

    manual = manual_flops()
    save_plots(results)
    write_report(results, manual)
    payload = {
        "device": str(DEVICE),
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "manual_flops": manual,
        "models": results,
    }
    (RESULTS / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nFinished. See report.md, results/results.json, and results/*.png")


if __name__ == "__main__":
    main()
