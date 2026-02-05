# Docker GPU Setup and Sionna Installation Guide

## Summary
This guide documents the complete process of setting up native Docker with NVIDIA GPU support and installing Sionna for RF/ML simulations on Ubuntu 22.04.

## Initial Problem
- Docker Desktop was blocking GPU access
- Error: `could not select device driver "" with capabilities: [[gpu]]`
- NVIDIA driver version: 570.172.08 available but not accessible to Docker

### To Check Sionna, Mitsuba and dr. Jit version installed
docker exec magical_margulis python -c "import sionna, drjit, mitsuba; print('Sionna:', sionna.__version__); print('Dr.Jit:', drjit.__version__); print('Mitsuba:', mitsuba.__version__)"


## Solution: Switch from Docker Desktop to Native Docker

### Step 1: Remove Docker Desktop

```bash
# Stop Docker Desktop services
sudo systemctl stop docker docker.socket

# Remove Docker Desktop
sudo apt-get remove docker-desktop

# Clean up Docker Desktop remnants
rm -rf ~/.docker/desktop
```

### Step 2: Install Native Docker Engine

```bash
# Update package index
sudo apt-get update

# Install native Docker Engine
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### Step 3: Configure NVIDIA Container Toolkit

```bash
# Configure NVIDIA Container Toolkit for Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Start Docker service
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Add user to docker group (requires logout/login to take effect)
sudo usermod -aG docker $USER
```

### Step 4: Verify GPU Access

```bash
# Check if NVIDIA runtime is available
sudo docker info | grep -i runtime
# Should show: Runtimes: io.containerd.runc.v2 nvidia runc

# Test GPU access
sudo docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

**Expected Output:**
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX A6000               On  |   00000000:01:00.0  On |                  Off |
| 30%   46C    P8             31W /  300W |    2147MiB /  49140MiB |     26%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

### Step 5: Test TensorFlow GPU Support

```bash
# Test TensorFlow GPU container
docker run --rm --gpus all tensorflow/tensorflow:2.13.0-gpu python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Expected Output:**
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Step 6: Install and Test Sionna

```bash
# Start container with proper user mapping
docker run --rm --gpus all -it --user $(id -u):$(id -g) -e HOME=/tmp -v $(pwd):/workspace -w /workspace tensorflow/tensorflow:2.13.0-gpu bash

# Inside container - install Sionna
pip install sionna

# Test Sionna
python -c "
import sionna as sn
import tensorflow as tf
print('✓ Sionna version:', sn.__version__)
print('✓ TensorFlow version:', tf.__version__)
print('✓ GPUs available:', len(tf.config.list_physical_devices('GPU')))
"
```

**Expected Output:**
```
✓ Sionna version: 0.19.2
✓ TensorFlow version: 2.13.0
✓ GPUs available: 1
```

## Sionna Test Script

Create `simple_sionna_test.py`:

```python
#!/usr/bin/env python3
"""
Simple working Sionna test
"""

import tensorflow as tf
import sionna as sn
import numpy as np

print("🚀 SIONNA WORKING TEST")
print("=" * 40)

# Check setup
print(f"✓ Sionna: {sn.__version__}")
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Simple working example
print("\n📡 Testing Sionna Components:")

try:
    # 1. QAM Mapper
    mapper = sn.mapping.Mapper("qam", num_bits_per_symbol=4)
    demapper = sn.mapping.Demapper("app", "qam", num_bits_per_symbol=4)
    print("✓ QAM Mapper/Demapper created")
    
    # 2. AWGN Channel
    channel = sn.channel.AWGN()
    print("✓ AWGN Channel created")
    
    # 3. Simple simulation
    batch_size = 100
    num_bits = 1000
    
    # Generate random bits
    bits = tf.random.uniform([batch_size, num_bits], 0, 2, dtype=tf.int32)
    
    # Map to symbols
    symbols = mapper(bits)
    
    # Add noise
    noisy_symbols = channel([symbols, 10.0])  # 10 dB SNR
    
    print(f"✓ Processed {symbols.shape} symbols")
    print(f"✓ Symbol power: {tf.reduce_mean(tf.abs(symbols)**2):.3f}")
    
    print("\n🎉 Sionna is ready for RF simulations!")
    
except Exception as e:
    print(f"❌ Error: {e}")
```

## Persistent Sionna Setup Options

### Option 1: Custom Docker Image (Recommended)

Create `Dockerfile.sionna`:

```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

# Install Sionna and common packages
RUN pip install sionna matplotlib jupyter ipykernel scipy numpy

# Set working directory
WORKDIR /workspace

# Optional: Create a non-root user
RUN useradd -m -u 1000 sionna_user
USER sionna_user

# Entry point
CMD ["bash"]
```

Build and use:

```bash
# Build custom image
docker build -f Dockerfile.sionna -t sionna-gpu:latest .

# Use custom image
docker run --rm --gpus all -it -v $(pwd):/workspace sionna-gpu:latest
```

### Option 2: Docker Compose (For Projects)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  sionna:
    build:
      context: .
      dockerfile: Dockerfile.sionna
    volumes:
      - ./:/workspace
    working_dir: /workspace
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

Use:

```bash
docker-compose run --rm sionna
```

### Option 3: Named Volume for Packages

```bash
# Create persistent volume for pip packages
docker volume create sionna-packages

# Use with volume mount
docker run --rm --gpus all -it \
  --user $(id -u):$(id -g) \
  -v sionna-packages:/tmp/.local \
  -v $(pwd):/workspace \
  -w /workspace \
  tensorflow/tensorflow:2.13.0-gpu bash
```

## Quick Start Commands

```bash
# For one-time use (packages lost after exit)
docker run --rm --gpus all -it --user $(id -u):$(id -g) -e HOME=/tmp -v $(pwd):/workspace -w /workspace tensorflow/tensorflow:2.13.0-gpu bash

# With persistent packages volume
docker run --rm --gpus all -it --user $(id -u):$(id -g) -v sionna-packages:/tmp/.local -v $(pwd):/workspace -w /workspace tensorflow/tensorflow:2.13.0-gpu bash

# Using custom image (after building)
docker run --rm --gpus all -it -v $(pwd):/workspace sionna-gpu:latest
```

## Key Achievements

✅ **Native Docker**: Switched from Docker Desktop to native Docker Engine  
✅ **GPU Access**: Full NVIDIA RTX A6000 access with 44GB VRAM  
✅ **CUDA Support**: TensorFlow 2.13.0 with CUDA 12.8  
✅ **Sionna Ready**: Version 0.19.2 with GPU acceleration  
✅ **User Permissions**: Proper file ownership with user mapping  
✅ **Performance**: No VM overhead, direct hardware access  

## System Configuration

- **OS**: Ubuntu 22.04
- **NVIDIA Driver**: 570.172.08
- **Docker**: Native Docker Engine 29.0.2
- **CUDA**: 12.8 (in container)
- **TensorFlow**: 2.13.0-gpu
- **Sionna**: 0.19.2
- **GPU**: NVIDIA RTX A6000 (49GB VRAM)

## Common Issues and Solutions

### Issue: Permission denied for Docker commands
```bash
# Add user to docker group and restart session
sudo usermod -aG docker $USER
# Then logout and login again
```

### Issue: GPU not accessible
```bash
# Check NVIDIA runtime
docker info | grep -i runtime
# Should show "nvidia" in the list
```

### Issue: Container files owned by root
```bash
# Always use user mapping
docker run --user $(id -u):$(id -g) ...
```

---

**Result**: Complete Docker GPU setup with Sionna ready for RF/ML simulations! 🚀
