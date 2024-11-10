# cameraStudy_tool
Started at the CTRL-HACK-DEL Hackathon 2024 at York University, and continued to work on it afterwards. This program uses a camera to scan a human, and a pen. The human points the pen at a particular body part, and it will be dectected by the camera, and bring information about that body part. A good tool for biology students learning Anatomy.



# Setup Commands

## 1.1 First-Time Setup
```bash
# Navigate to project directory
cd /path/to/project

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install PyTorch for CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify installation
python -c "import torch; print(torch.rand(5, 3))"
```

## 1.2 Return Setup
```bash
# Navigate to project directory
cd /path/to/project

# Activate virtual environment
source venv/bin/activate

# Verify PyTorch installation
python -c "import torch; print(torch.rand(5, 3))"
```

# 2.1 To run the code

## Setup
-> Steps 1.1 or 1.2.
```bash
# Execute python code in terminal
python3 cam_detect.py
```
## Testing
-> Use a blue pen and point the top of the pen, at the key points the camera detects. See what happens when you do!

