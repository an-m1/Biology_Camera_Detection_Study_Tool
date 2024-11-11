#  Biology Camera Detection Study Tool (BioCam-DST)
Started at the CTRL-HACK-DEL Hackathon 2024 at York University, and continued to work on it afterwards. This program uses a camera to scan a human, and a pen. The human points the pen at a particular body part, and it will be dectected by the camera, and bring information about that body part. A good tool for biology students learning Anatomy.


Using a common student item, a blue pen, hover over the body's keypoints, and discover what is beneath the skin. Students are to observe, and then try to retain what they observed. This is a more interactive tool, opposed to boring old flashcards.


# 1) Setup Commands

## 1.1 First-Time Setup
```bash
# Navigate to project directory
cd /path/to/project

# Removed the problematic virtual environment (if applicable)
rm -rf venv

# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools

# Install PyTorch for CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install scikit-learn matplotlib opencv-python numpy pandas

pip install mediapipe

which python3


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

# 2) To run the code

## 2.1 Setup
-> Steps 1.1 or 1.2.
```bash
# Execute python code in terminal
python3 cam_detect.py
```
## 2.2 Testing
-> Use a blue pen and point the top of the pen, at the key points the camera detects. See what happens when you do!

