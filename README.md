# drone_cnn

Python CNN files for drone obstacle avoidance.

## 📖 Wiki
- [Project description](https://github.com/avncalst/drone_cnn/wiki) - 2 approaches for CNN development
- ArduCopter SITL simulation instructions

## 📁 Project Structure
- `avcNet/` - Training CNN models (HDF5/PB binary files included in `rpi3/`)
- `rpi3/` - Files for running CNN on Raspberry Pi 3 with OpenVINO, DNN-OpenCV, TensorFlow, Keras
- `rpi3-rpi4/` - Files for running CNN on Raspberry Pi 3/4 and Coral USB accelerator
- `donkeycar/` - Modified files for ArduPilot Rover-copter compatibility
- `autoencoder/` - Autoencoder implementations
- `transfer_learning/` - Jupyter notebooks for custom object detection/classification
- `jupyter_notebooks/` - Custom classification notebooks
- `rosetta/` - Scripts using rosetta app for DJI drone control

## 🖥️ rpi3-rpi4 Obstacle Avoidance Scripts

### Performance Comparison
- **Intel NCS1 (DNN-MYRIAD):** ~20 FPS with OpenCV 4.1.0
- **DNN-OpenCV:** ~6 FPS on rpi3

---

### DepthAI-based Scripts

| File | Description |
|------|-------------|
| `oak_avoid.py` | OAK-D depth camera obstacle avoidance |
| `oak_avoid_rev3.py` | New avoidance algorithm with person tracking |
| `oak_follow_ip_rev1.0.py` | Compatible with H12Pro remote control |
| `oak_avoid_ip_rev.py` | H12Pro remote control compatibility |

---

### Algorithm Variants

| File | Description |
|------|-------------|
| `avoid_optoflow_rev1.py` | OFLOW-based avoidance algorithm |
| `revised_oak_avoid_ip_rev3.py` | Improved avoidance algorithm |

---

### Other Hardware

| File | Description |
|------|-------------|
| `avcNet/avcNet.h5` | Trained model for inference |
| `avcNet_saved/` | Saved trained model |
| `avcNet_64bit_saved/` | 64-bit model files |
| `avcNet_tiny_saved/` | Optimized model |

---

### Coral USB Accelerator (TFLite)

- **File:** `ai_avoid_rev.py` - TensorFlow Lite inference with Coral USB accelerator
- **Training:** Transfer learning via Jupyter notebook in `transfer_learning/` on Google Colab

---

### RealSense Camera

- **File:** `real_avoid.py` - Intel RealSense camera for obstacle avoidance

---

## 🤖 DJI Drone Integration

- **`rosetta/`** - Python scripts for DJI drone control via rosetta app
- Face tracking app using MobileNet-SSD (see `transfer_learning/`)

---

## 📊 Autoencoder

- test of encoder-decoder structure (see `autoencoder/`)

---

## 🏗️ Architecture

![Variational Autoencoder](https://github.com/avncalst/drone_cnn/blob/master/images/test29.png)

---

## 📝 Additional Notes

- Simple Python obstacle avoidance script included
- `donkeycar/` contains ArduPilot Rover-copter integration files
