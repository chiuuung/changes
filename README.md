# FYP - Human-Cat Interaction Detector

This is the **organized and ready-to-run** version of the project with all essential components clearly labeled.

---

## Project Structure

```
FYP-Codes/
├── README.md                    # You are here
├── AI_Model/                    # AI Model & Training
│   ├── weights/
│   │   └── best.pt             # Trained YOLOv8s model (92.5% mAP@50, 77.1% mAP@50-95)
│   ├── train.py                # Script to train new models
│   ├── data.yaml               # Dataset configuration (4,220 images: 3,736 training / 307 validation / 177 test)
│   ├── runs/                   # Training results and metrics
│   │   └── detect/
│   │       └── weights/best.pt # Best model weights from epoch 100
│   └── README.md               # AI Model documentation
│
├── backend/                     # Backend Server
│   └── streaming_backend_server.py         # Main server (ESP32 camera)
│
├── iOS_App/                     # iOS Application
│   └── PetGuard/                # Xcode project folder
│       ├── PetGuard/            # App source files
│       │   ├── StreamView.swift      # Live stream viewer
│       │   ├── ContentView.swift     # Main app view
│       │   ├── NetworkManager.swift  # API communication
│       │   ├── VideosView.swift      # Video playback
│       │   └── PetGuardApp.swift     # App entry point
│       └── PetGuard.xcodeproj/  # Xcode project
│
├── hardware_part/               # ESP32-S3 Hardware
│   └── esp32_control/          # ESP32-S3 camera + BLE proximity
│       └── esp32s3_camera_stream.ino  # ESP32-S3 firmware
│
├── recorded_videos/             # Auto-recorded videos
│   └── interaction_*.mp4       # Saved interactions (max 10)
│
└── requirements.txt             # Python dependencies
```

---

## Quick Start Guide

### 1. Start the Backend Server (Mac/MacBook)

```bash
cd "c:\Users\james\Desktop\FYP-Codes\backend"

# Run the optimized streaming server
python streaming_backend_server.py
```

**Requirements:**
- Python 3.8+
- Required packages: flask, flask-cors, flask-sock, ultralytics, opencv-python, numpy

**Installation:**
```bash
pip install flask flask-cors flask-sock ultralytics opencv-python numpy
```

**What you should see:**
```
==================================================================
  Optimized Streaming Backend - Human-Pet Interaction Detector
  WebSocket + Binary Transmission + Adaptive FPS
==================================================================

✅ Model loaded
✅ Camera initialized

📱 iOS App Configuration:
   Server URL: http://192.168.1.100:5001
   WebSocket Stream: ws://192.168.1.100:5001/ws/stream
   MJPEG Stream: http://192.168.1.100:5001/stream/mjpeg

📊 Network Optimization:
   Protocol: WebSocket (5-20ms latency)
   Transmission: Binary JPEG (-33% overhead)
   Initial FPS: 15 (adaptive 10-25 FPS)
   Frame Skip: Enabled when queue backs up

🚀 Starting optimized server...
```

Server URL: http://YOUR_IP:5001
Live Stream: http://YOUR_IP:5001/stream/live
```

### 2. Stop the Server

```bash
# Press Ctrl+C in the terminal

# OR force kill:
lsof -ti:5001 | xargs kill -9
```

### 3. Run iOS App

1. Open Xcode
2. Create new iOS project (or use existing)
3. Copy all files from `iOS_App/xcode_project/` to your Xcode project
4. Update `Info.plist`:
   ```xml
   <key>NSAppTransportSecurity</key>
   <dict>
       <key>NSAllowsArbitraryLoads</key>
       <true/>
   </dict>
   ```
5. In app Settings, set server URL: `http://YOUR_MAC_IP:5001`
6. Run on iPhone/iPad

---

## Key Features

### Auto-Recording
- Automatically records when **cat AND human** detected together
- 2-second cooldown between recordings
- Videos saved as: `interaction_YYYYMMDD_HHMMSS.mp4`

### Auto-Storage Management
- Keeps only **10 newest videos** automatically
- Older videos deleted after each recording
- Manual cleanup: `python3 cleanup_videos.py`

### Live Streaming
- Real-time webcam feed to iOS devices
- Shows detection confidence levels
- Recording indicator when active

### iOS Viewer App
- **Live Stream** tab: Watch real-time feed
- **Recordings** tab: Browse and play saved videos
- No iPhone camera needed (viewer only)

---

## Model Performance

- **Model**: YOLOv8s (11.2M parameters)
- **Dataset**: 4,220 annotated images (3,736 train / 307 val / 177 test)
- **Overall mAP@50**: 92.5%
- **Overall mAP@50-95**: 77.1%
- **Cat Detection**: 90.8% mAP@50
- **Human Detection**: 94.2% mAP@50
- **Model Size**: ~22 MB
- **Training Hardware**: NVIDIA RTX 4090, CUDA 12.1
- **Location**: `AI_Model/runs/detect/weights/best.pt`

---

## Configuration

### **Backend Server Settings**
Edit `backend/streaming_backend_server.py`:

```python
# Configuration section
MODEL_PATH = PROJECT_ROOT / "AI_Model" / "runs" / "detect" / "weights" / "best.pt"
VIDEOS_DIR = PROJECT_ROOT / "recorded_videos"
CONFIDENCE_THRESHOLD = 0.25                  # Detection sensitivity
COOLDOWN_SECONDS = 2                         # Recording timeout
MAX_VIDEOS = 10                              # Max stored videos
```

### **iOS App Settings**
In the app's Settings tab:
- **Server URL**: `http://YOUR_MAC_IP:5001`
- Test connection before viewing stream

---


## Requirements

### **For Backend (Mac/Jetson)**
```bash
cd AI_Model
pip3 install -r requirements.txt
```

**Key packages:**
- `ultralytics` (YOLOv8)
- `opencv-python` (Camera & video)
- `flask` (Web server)
- `torch` (Deep learning)

### **For iOS App**
- Xcode 14+
- iOS 15+ device/simulator
- Network connectivity to Mac/Jetson

---

## Workflow

```
1. ESP32-S3 captures camera frames + scans BLE beacons
   ↓
2. Frames sent via HTTP POST to backend server
   ↓
3. YOLOv8s detects humans & cats in real-time
   ↓
4. Auto-records when both detected (async video writing)
   ↓
5. Streams processed frames via WebSocket to iOS app
   ↓
6. BLE proximity alerts forwarded to iOS with zone labels
   ↓
7. View live feed, recordings, & proximity alerts on iPhone
```

---

## System Architecture

The system uses a 3-layer architecture:

```
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   Hardware Layer    │      │   Backend Server     │      │    iOS App       │
│                     │      │                      │      │                  │
│ ESP32-S3 + OV5640   │─────>│ Flask + YOLOv8s      │─────>│ PetGuard App     │
│ Camera (10 FPS)     │ HTTP │ AI Detection         │  WS  │ Live Stream      │
│                     │      │ Video Recording      │      │ Video Playback   │
│ BLE Beacons (×2)    │─────>│ Proximity Tracking   │─────>│ Proximity Alerts │
│ iBeacon Protocol    │ HTTP │ Multi-zone Support   │  WS  │ Multi-zone UI    │
└─────────────────────┘      └──────────────────────┘      └──────────────────┘
```

### Backend API Endpoints
- `POST /esp32/frame` — Receive camera frames from ESP32
- `POST /esp32/distance` — Receive BLE distance data
- `WS /ws/stream` — WebSocket live stream to iOS
- `GET /videos` — List recorded videos
- `GET /video/<uid>` — Get specific video file
- `GET /health` — Health check
- `GET /stream/mjpeg` — MJPEG fallback stream

### BLE Proximity Alert Flow
1. ESP32 scans BLE beacons using iBeacon protocol
2. RSSI path loss formula: `distance = 10^((RSSI_1M - measured_RSSI) / (10 × n))`
3. Distance ≤ 1.0m triggers danger alert
4. Backend forwards alert via WebSocket to iOS
5. iOS shows zone-specific alert with 2-second debouncing

---

## Hardware Setup (ESP32-S3)

### Requirements
- ESP32-S3 DevKit with OV5640 camera module
- 2× BLE iBeacon devices
- USB power bank for portable operation

### Arduino IDE Configuration
1. Board: `ESP32S3 Dev Module`
2. Flash Size: `16MB`
3. Partition Scheme: `Huge APP (3MB No OTA/1MB SPIFFS)`
4. PSRAM: `OPI PSRAM`

### Firmware Configuration
Edit `hardware_part/esp32_control/esp32s3_camera_stream.ino`:
```cpp
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* serverIP = "YOUR_MAC_IP";
const int serverPort = 5001;
```

Configure BLE beacon MAC addresses and UUIDs to match your beacons.

---

## Troubleshooting

### **Server won't start**
```bash
# Check if port 5001 is in use
lsof -i :5001

# Kill existing process
lsof -ti:5001 | xargs kill -9
```

### **Model not found**
```bash
# Verify model exists
ls -lh AI_Model/runs/detect/weights/best.pt

# Update MODEL_PATH in streaming_backend_server.py
```

### **iOS app can't connect**
```bash
# Get your Mac's IP address
ipconfig getifaddr en0

# Update server URL in iOS app settings
# Example: http://10.17.94.27:5001
```

### **ESP32 not connecting**
- Verify WiFi SSID/password in firmware
- Ensure ESP32 and Mac are on the same network
- Check serial monitor for connection status
- Verify server IP address in firmware matches Mac IP

---

## Training New Models

### Environment Setup
```bash
pip install ultralytics opencv-python numpy torch torchvision
```

### Training
```bash
cd AI_Model
python train.py
```

The training script (`AI_Model/train.py`) uses YOLOv8s with:
- **100 epochs**, batch size 16, image size 640
- **AdamW optimizer** with cosine annealing LR schedule
- **Early stopping** patience=20
- **Data augmentation**: mosaic=0.5, flipud=0.1, degrees=15, shear=10, HSV color jitter, scale=0.2, translate=0.1
- **Dataset**: Configured in `AI_Model/data.yaml` (4,220 images across 3 splits)

### Validation
```bash
yolo val model=runs/detect/weights/best.pt data=data.yaml
```

### Resume Training
```bash
yolo train resume model=runs/detect/weights/last.pt
```

---

## Project Status

- AI Model: Trained and tested (92.5% mAP@50, 77.1% mAP@50-95)
- Backend Server: ESP32 streaming + Auto-recording + Async processing
- iOS App: Live stream + Video playback with speed controls + Proximity alerts
- ESP32-S3: Camera streaming + Dual BLE beacon proximity detection + NTP sync
- Storage: Auto-cleanup (MAX_VIDEOS=10) implemented

---

## Support

For detailed setup instructions, refer to:
- `Documentation/STREAMING_SETUP.md` - iOS app setup
- `Documentation/TRAINING_DOCUMENTATION.md` - Model training
- `Documentation/README.md` - Full project details

---

## Project Info

**Title**: Human-Cat Interaction Detector with iOS Monitoring  
**Technology**: YOLOv8, Python, Swift, Flask, OpenCV  
**Platform**: Mac/Jetson Nano (Backend) + iOS (Frontend)  
**Architecture**: Client-Server Streaming Model

---

**Ready to run! 🚀**
