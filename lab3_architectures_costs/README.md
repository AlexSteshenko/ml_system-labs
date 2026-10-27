# Lab 3 — Architectures and their costs

This VS Code project implements the Fashion-MNIST lab with MLP, CNN, and patch-based Transformer attention models.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- VS Code with the Python extension
- Internet access on first run to download Fashion-MNIST
- A CPU works; CUDA-enabled PyTorch is used automatically if available

## 2. Create an environment

Open this folder in VS Code, then use the integrated terminal.

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from Command Prompt instead:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

In VS Code, choose `.venv` as the Python interpreter (Command Palette → “Python: Select Interpreter”).

## 3. Run the lab

```bash
python main.py
```

The first run downloads Fashion-MNIST. Training all three models for five epochs may take a while, especially on CPU.

## 4. Outputs

After successful completion:
- `report.md` is generated with measured parameter counts, test accuracy, latency, hand FLOPs and THOP comparison.
- `results/results.json` stores machine-readable measurements.
- `results/accuracy_vs_parameters.png`
- `results/accuracy_vs_latency.png`
- `results/validation_accuracy_by_epoch.png`
- `results/mlp_fashion_mnist.pt`, `results/cnn_fashion_mnist.pt`, `results/attention_fashion_mnist.pt` contain trained weights.

Do not submit invented measurements. Run the script on your own machine and use its actual outputs.

## 5. Important architecture note

The PDF states a target of 100k–150k parameters while also prescribing exact layer sizes. Those exact definitions do not all fit the target range. This project follows the explicit layer definitions and reports their actual counts; ask the instructor whether the target range or the specified layers take priority.

## 6. Commit with Git

```bash
git init
git add main.py requirements.txt README.md report.md .gitignore
git commit -m "Complete Lab 3 architecture cost comparison"
```

Do not commit downloaded data, virtual environments, or large model weights unless your instructor specifically asks for them. Add result plots and `report.md` if required by your course.
