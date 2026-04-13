# CORRECTED Final Presentation — ELEC4848 Senior Design Project 2025-2026

> **Note:** All corrections compared to the original presentation are marked with **<u>bold underline</u>**.
> Suggested improvements (not errors) are marked with **(NEW)**.

---

## Slide 1 — Title Slide

# AI COMPUTER VISION FOR ABNORMAL DETECTION IN HOME PET CARE MONITORING

**ELEC4848 Senior Design Project 2025-2026**

Ng Tsz Chiu James

---

## Slide 2 (01) — INTRODUCTION

### Background:
- Pet care industry growing — pets seen as family [1],[2]
- Professional pet sitters visit homes but owners have NO oversight

### Research gap:
- Existing AI focuses on pet behavior, not staff accountability [3]

### Problem:
- No real-time verification of caregiver conduct
- No detection of unauthorized room entry
- No video evidence for disputes

### Five Objectives:
1. AI model for real-time human–cat detection
2. Wearable hardware with thermal management
3. Backend server for AI inference & streaming
4. iOS app (PetGuard) for live monitoring & alerts
5. Multi-zone BLE proximity detection

> **Speaker notes (~1.5 min):** Introduce the pet care market growth, explain the accountability gap for home visits, state the research gap (existing AI = pet behavior, not staff oversight), list the 3 key problems, then the 5 objectives.

---

## Slide 3 (02) — METHODOLOGY: System Overview

### System Overview:

```
┌──────────────────────────┐    ┌────────────────────────────┐    ┌──────────────────────┐    ┌────────────────────────┐
│ Hardware 1 (BLE Beacons) │    │ Hardware 2 (ESP32-S3)      │    │ Backend Server       │    │ iOS App (PetGuard)     │
│                          │    │                            │    │                      │    │                        │
│ + 2× BLE Beacons        │BLE │ + ESP32-S3  ←── CORRECTED  │HTTP│ - Flask Server       │WS  │ - Live view page       │
│ - Broadcasts RSSI       │───→│ + OV5640 camera            │───→│   (Port 5001)        │───→│ - Recording page       │
│   Signals               │    │ + 5× Cooling fins          │    │ - Trained YOLOv8s    │    │ - Setting page         │
│                          │    │ + 2× 4cm 5V cooling fans   │1. Images (JPEG, 10 FPS)│    │ - Alert banner         │
│                          │    │ + Power Bank               │2. RSSI signal data(1Hz)│    │                        │
│                          │    │ - Scan RSSI Signals via BLE│    │ - Video Storage(Max10)│ 1. Streaming            │
│                          │    │                            │    │ - Recording Logic    │ 2. Recording            │
│                          │    │                            │    │ - RSSI Distance Calc │ 3. Alert Trigger        │
│                          │    │                            │    │ - Alert Trigger Logic│                        │
└──────────────────────────┘    └────────────────────────────┘    └──────────────────────┘    └────────────────────────┘
```

### Data flow of 2 key features:
1. Live Streaming & Object Detection
2. Distance Calculation & Proximity Alert

**<u>CORRECTION: Original slide had "ESP32-S2" — corrected to "ESP32-S3"</u>**

> **Speaker notes (~1 min):** Walk through the 4-component architecture left to right. BLE beacons broadcast RSSI → ESP32-S3 scans + captures video → HTTP POST to backend for AI inference → WebSocket streams to iOS app. Emphasize the two data flows.

---

## Slide 4 (03) — METHODOLOGY: Key Design

### AI Model (YOLOv8s):
- **<u>4,220 images (3,736 train / 307 val / 177 test) on Roboflow [4]</u>**
- 100 epochs on NVIDIA RTX 4090 [5]
- Transfer learning from COCO weights [6]
- **<u>Multi-modal augmentation: geometric, color space, mosaic, random erasing [7]</u>**

**<u>CORRECTION: Original said "contrastive [6],[7]" — "contrastive self-supervised learning" was fabricated. Actual augmentation uses random erasing (10% probability). Reference to CS231n Lecture 12 (Self-Supervised Learning) removed.</u>**

### Hardware:
- *[Figure 2: ESP32-S3 close-up with camera]*
- *[Figure 3: Cooling fins detail]*
- *[Figure 4: Full chest-mount harness worn]*

**(NEW) Dataset screenshot from Roboflow showing 4,220 images with train/val/test split**

> **Speaker notes (~1.5 min):** Explain YOLOv8s choice (single-stage, C2f modules, 11.1M params). Mention batch size reduced from 32→16 due to OOM. Show hardware photos — cooling fins + fans solve thermal shutdown problem. Dataset diversity from Roboflow.

---

## Slide 5 (04) — RESULTS: AI Detection Performance

### AI Detection Performance

| Metric | Value |
|--------|-------|
| mAP@50 Overall | **92.5%** |
| Precision | **94.5%** |
| Recall | **85.4%** |
| mAP@50-95 | **77.1%** |

### Per-Class Breakdown:

| Class | Precision | Recall | mAP@50 | Instances |
|-------|-----------|--------|--------|-----------|
| Cat | 95.9% | 81.4% | 90.8% | 316 |
| Human | 93.1% | 89.4% | 94.2% | 104 |

**(NEW) Per-class table added — shows the precision/recall asymmetry worth discussing**

- *[Figure 1: iOS app — Human detected 79.1%]*
- *[Figure 2: iOS app — Cat detected 79.9%]*

> **Speaker notes (~1.5 min):** Highlight 92.5% mAP@50 exceeds the 90% target. Explain why cat recall (81.4%) < human recall (89.4%) — cats have more morphological diversity. High precision (94.5%) means few false alarms, critical for a system triggering recordings. Show real device screenshots.

---

## Slide 6 (05) — RESULTS: End-to-End Testing

### End-to-End Testing

| Test | Result |
|------|--------|
| Video latency (WebSocket) | < 20 ms |
| Proximity alert delivery | 300–400 ms |
| Detection → recording start | ~100 ms |
| **<u>Recording stop → availability</u>** | **<u>~500 ms</u>** |
| Continuous runtime | 2+ hours |
| **<u>Beacon zone tests (5 scenarios)</u>** | **<u>All Passed</u>** |

**<u>CORRECTIONS:</u>**
- **<u>"recording stop 100-200ms" → corrected to "~500 ms" per technical paper Table 4</u>**
- **<u>"4 test cases" → corrected to "5 scenarios" per technical paper Table 3 (At B1, At B2, At B1 ~2m, Exit B1, At B1 & B2)</u>**

- *[Figure 1: iOS app — Human detected with recording active]*
- *[Figure 2: iOS app — Cat detected with recording active]*

> **Speaker notes (~1.5 min):** WebSocket binary transmission reduced latency from 1-2 seconds (HTTP+base64) to <20ms. Proximity alerts <400ms. All 5 beacon scenarios passed. Continuous runtime >2 hours with dual-layer cooling (was 8-12 min without).

---

## Slide 7 (06) — DIFFICULTIES & MITIGATION

| Difficulty | Impact | Mitigation |
|------------|--------|------------|
| Thermal shutdown in 8–12 min | Device unusable | Cooling system → 2+ hours |
| HTTP+base64: 1–2 s lag | Poor user experience | WebSocket+binary → <20 ms |
| Fail detect in low-light condition | Fail detect | Add light module → detect in low-light |
| Fail detect in small target conditions | Fail detect | Increase datasets → detect target in different condition |

**(NEW) Optional additions from corrected report:**

| Difficulty | Impact | Mitigation |
|------------|--------|------------|
| Batch size 32 → OOM on RTX 4090 | Cannot train | Reduced to batch 16 |
| Main-thread image decoding on iOS | UI freezes | Offloaded to background thread via GCD |
| Beacon MAC instability in RF noise | Missed zone alerts | UUID+Major+Minor fallback matching |

- *[Figure 1: Low-light detection failure example]*
- *[Figure 2: Low-light detection with human at 40.9%]*

> **Speaker notes (~1 min):** The thermal shutdown was the most critical — went from 8-12 min to 2+ hours. HTTP→WebSocket migration gave 100× latency improvement. Low-light and small targets are acknowledged limitations for future work.

---

## Slide 8 (07) — SCHEDULE & STATUS

| Phase | Period | Deliverables | Status |
|-------|--------|-------------|--------|
| Literature review & planning | Sep–Oct 2025 | Problem definition, research gap | ✅ |
| Dataset collection & annotation | Oct–Nov 2025 | 4,220 images on Roboflow | ✅ |
| AI model training & validation | Nov–Dec 2025 | YOLOv8s best.pt (92.5% mAP@50) | ✅ |
| Hardware assembly & thermal design | Dec 2025–Jan 2026 | Wearable prototype with cooling | ✅ |
| Backend server development | Jan–Feb 2026 | Flask + WebSocket pipeline | ✅ |
| PetGuard iOS app development | Feb–Mar 2026 | SwiftUI app with alerts & recording | ✅ |
| System integration & E2E testing | Mar–Apr 2026 | Testing (5 beacon test scenarios) | ✅ |
| Report writing & presentation | Mar–Apr 2026 | Final report + technical paper | ✅ |

**(NEW) Added ✅ status checkmarks — original had empty Status column**

> **Speaker notes (~0.5 min):** Quick overview — all 8 phases completed on schedule over ~7 months. Heaviest work was Jan-Mar with parallel backend + iOS development.

---

## Slide 9 (08) — CONCLUSION & NEXT STEPS

### Objectives Achieved:
- ✅ AI model: **92.5% mAP@50** real-time dual-class detection (exceeded 90% target)
- ✅ Hardware: wearable with **2+ hours** of stable operation (12× improvement over uncooled)
- ✅ Backend: **<20 ms** WebSocket streaming latency
- ✅ PetGuard iOS app: live streaming, **<u>0.5×/1×/2×/3× recording playback</u>**, room-specific alerts
- ✅ BLE proximity: **all 5 test scenarios passed**, alerts within 300–400 ms

**<u>CORRECTION: Added "0.5×/1×/2×/3×" playback speeds (was just "recording review"). Original presentation didn't mention 3× speed which exists in the code.</u>**

### Next Steps:
1. Expand dataset to 10,000+ images (diverse lighting & breeds)
2. Add infrared LED for low-light performance
3. Kalman filtering on RSSI → false alerts 5–8% → 2–3%
4. Adaptive fan control with temp sensor → save power
5. Cloud deployment (AWS/Azure) → remote access

> **Speaker notes (~1 min):** Summarize all 5 objectives met. Highlight key numbers. Next steps prioritized by impact — dataset expansion and infrared are the biggest wins for accuracy.

---

## Slide 10 (09) — REFERENCES

**<u>CORRECTED reference list (removed [7] Self-Supervised Learning, renumbered, added Ultralytics):</u>**

[1] "Growing global pet grooming market," *Vet. Rec.*, vol. 196, no. 2, p. 65, Jan. 2025, doi: 10.1002/vetr.5144.

[2] Y. Brugger, "Private equity challenge – Pets at home commercial assessment of European pet market," M.S. thesis, ProQuest, 2024.

[3] A. Hussain et al., "A deep learning approach for detecting and classifying cat activity," *IEEE Sensors J.*, vol. 24, no. 2, pp. 1996–2008, Jan. 2024.

**<u>[4] G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," Jan. 2023. [Online]. Available: https://docs.ultralytics.com/models/yolov8/</u>**

[5] NVIDIA, "GeForce RTX 40 Series," NVIDIA Newsroom, Sep. 2022. [Online]. Available: https://nvidianews.nvidia.com/

[6] T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. ECCV*, 2014, pp. 740–755, doi: 10.1007/978-3-319-10602-1_48.

[7] Stanford University, "CS231n Lecture 9: Detection, Segmentation," 2025. [Online]. Available: https://cs231n.stanford.edu/slides/2025/lecture_9.pdf

[8] Espressif Systems, "ESP32-S3 Series Datasheet," Version 2.2, 2021. [Online]. Available: https://www.espressif.com/

[9] OmniVision Technologies, "OV5640 Datasheet," Version 2.03, 2011. [Online]. Available: https://cdn.sparkfun.com/datasheets/Sensors/LightImaging/OV5640_datasheet.pdf

**<u>CHANGES: Removed old [7] (CS231n Lecture 12: Self-Supervised Learning) — it was only cited for the fabricated "contrastive" augmentation. Added [4] Ultralytics YOLOv8 (the core framework). Renumbered all subsequent references.</u>**

---

## Slide 11 (10) — THANK YOU

# THANK YOU

---

---

# SUMMARY OF ALL CHANGES

## Must-Fix (6 corrections):

| # | Slide | Original | Corrected |
|---|-------|----------|-----------|
| 1 | 3 (02) | "+ ESP32-S2" in system diagram | **"+ ESP32-S3"** |
| 2 | 4 (03) | "mosaic, contrastive [6],[7]" | **"mosaic, random erasing [7]"** |
| 3 | 6 (05) | "recording stop 100-200ms" | **"~500 ms"** |
| 4 | 6 (05) | "Beacon zone tests (4 test cases)" | **"Beacon zone tests (5 scenarios)"** |
| 5 | 9 (08) | "recording review" (no speeds) | **"0.5×/1×/2×/3× recording playback"** |
| 6 | 10 (09) | Reference [7] = Self-Supervised Learning | **Removed (only cited for fabricated "contrastive"); added Ultralytics YOLOv8 ref** |

## Suggested Improvements (optional):

| # | Slide | Suggestion |
|---|-------|------------|
| A | 5 (04) | Add per-class breakdown table (Cat vs Human precision/recall) |
| B | 7 (06) | Add 1-2 more difficulties (batch OOM, GCD fix, beacon MAC fallback) |
| C | 8 (07) | Fill in Status column with ✅ checkmarks |
| D | 4 (03) | Add dataset size detail (3,736/307/177 split) |

## Timing Guide (10 minutes):

| Slide | Topic | Time |
|-------|-------|------|
| 1 | Title | 0:00 |
| 2 (01) | Introduction | 0:00–1:30 |
| 3 (02) | System Overview | 1:30–2:30 |
| 4 (03) | Key Design | 2:30–4:00 |
| 5 (04) | AI Results | 4:00–5:30 |
| 6 (05) | E2E Testing | 5:30–7:00 |
| 7 (06) | Difficulties | 7:00–8:00 |
| 8 (07) | Schedule | 8:00–8:30 |
| 9 (08) | Conclusion | 8:30–9:30 |
| 10 (09) | References | 9:30 |
| 11 (10) | Thank You / Q&A | 9:30–10:00 |
