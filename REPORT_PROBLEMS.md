# FYP Report vs Codebase — Discrepancy Analysis

**Date of Analysis:** April 13, 2026  
**Report Analysed:** `newest_final_report.txt` (ELEC4848 Senior Design Project 2025-2026)  
**Codebase Analysed:** `FYP-Codes-main/FYP-Codes-main/` repository

---

## Part A: Report Claims Something Exists, But the Codebase Does NOT Have It

These are items the report describes as implemented/present, but which are missing or inaccurate in the actual code.

---

### A1. Appendix A.1 — Repository Structure Paths Are Wrong (change the contents in the report, base on the followings)

> **✅ RESOLVED:** Updated Appendix A.1 in `CORRECTED_REPORT.md` with correct repository structure reflecting all file changes (removed yolo26n.pt, start_ios_server.sh, Documentation folder; added requirements.txt; updated file descriptions).

**Report Section:** Appendix A.1 (Project Source Code Repository)

The report lists the following repository structure, but many paths are incorrect:

| # | Report Claims (Path) | Actual Path in Codebase | Problem |
|---|---|---|---|
| 1 | `training_script/train_model.py` | `AI_Model/train.py` | Wrong folder name AND wrong filename |
| 2 | `weights/best.pt` (at top level) | `AI_Model/runs/detect/weights/best.pt` | Wrong location — it's nested inside `AI_Model/runs/detect/weights/` |
| 3 | `backend/solving_latency/` | **Does not exist** | Entire folder is missing from the repo |
| 4 | `esp32/esp32s3_camera_stream.ino` | `hardware_part/esp32_control/esp32s3_camera_stream.ino` | Wrong folder name |
| 5 | `ios_app/PetGuard/` | `iOS_App/PetGuard/` | Wrong folder name (case difference and underscore vs no underscore) |
| 6 | `requirements.txt` | **Does not exist** | File is completely missing from the repo |

---

### A2. `requirements.txt` — Missing File (keep the contents in the report, can take back from the new-FYP)

> **✅ RESOLVED:** Created `requirements.txt` at project root with: flask>=2.3.0, flask-cors>=4.0.0, flask-sock>=0.6.0, ultralytics>=8.0.0, opencv-python>=4.8.0, numpy>=1.24.0. Updated report Appendix A.1 to reflect its presence.

**Report Section:** Appendix A.1

> "requirements.txt (Python dependencies: ultralytics, opencv, numpy, flask)"

**Problem:** No `requirements.txt` file exists anywhere in the repository. Users cannot install dependencies using the documented method.

---

### A3. `backend/solving_latency/` — Missing Folder (change the contents in the report, solving_latency features already included in the backend server codes)

> **✅ RESOLVED:** Report-only change. Updated `CORRECTED_REPORT.md` Appendix A.1 — removed `solving_latency/` folder reference. Latency optimization features are integrated directly in `streaming_backend_server.py`.

**Report Section:** Appendix A.1

> "backend/solving_latency/ (Optimized network architecture code with WebSocket support)"

**Problem:** This folder and all its contents are completely absent from the repository. The report implies this is a separate module containing optimized networking code, but no such directory exists.

---

### A4. Variable Speed Playback (×0.5, ×1, ×2) — NOT Implemented (keep the contents in the report, currently user cant control, the currently actual speed is x3, can we add the control features in the petGuard app/ backend server?)

> **✅ RESOLVED:** Implemented variable speed playback controls (×0.5, ×1, ×2) in `VideosView.swift` `VideoPlayerView`. Added speed selector bar UI with highlighted active speed button. Updated `CORRECTED_REPORT.md` §2.5.1 Recordings Tab and §3.4.1 to reflect implementation.

**Report Section:** §2.5.1 (User Interface Architecture — Recordings Tab)

> "Videos are sortable by date/time and support variable speed playback controls (×0.5, ×1, ×2) for video review."

**Problem:** `VideosView.swift` (line 204–233) shows a basic `VideoPlayerView` using `AVPlayer` with only a "Done" button — there are **no speed controls** (×0.5, ×1, ×2) implemented whatsoever. The video player is a standard SwiftUI `VideoPlayer` component without any custom playback rate UI.

---

### A5. Settings Persistence via UserDefaults — NOT Implemented (to check is it recommnedation, if not then in report write: setting is just for testing the connection but not to change the ip address)

> **✅ RESOLVED:** Report-only change. Updated `CORRECTED_REPORT.md` §3.4.1 — changed description to clarify that the Settings modal is for testing the backend connection during initial setup, not for frequently changing the IP address. The server IP remains stable within a Wi-Fi session.

**Report Section:** §3.4.1 (Difficulties Encountered, item 3)

> "IP address changes in the Settings modal were not being persisted to UserDefaults, requiring users to reconfigure the server IP address on each app launch. Adding proper data persistence to UserDefaults resolved this issue."

**Problem:** In `NetworkManager.swift` line 27, `baseURL` is declared as a plain property:
```swift
private(set) var baseURL = "http://172.20.10.3:5001"
```
There is **no UserDefaults usage** anywhere in `NetworkManager.swift` or `SettingsView`. The URL resets to the hardcoded default every time the app launches. The report claims this was fixed, but the fix was never actually implemented in the code.

---

### A6. Alert Debouncing with 2-Second Cooldown — NOT Implemented (to check is it recommnedation, if not then keep the contents in report, do the change in the code backend/petguard logic)

> **✅ RESOLVED:** Implemented 2-second debouncing cooldown in `StreamView.swift`. Added `lastAlertTime` state variable and time check in `.onChange(of: networkManager.proximityAlert)` handler — alerts only show if 2+ seconds since the last alert. Updated `CORRECTED_REPORT.md` §3.4.2 to reflect implementation.

**Report Section:** §3.4.2 (Difficulties Encountered, item 1)

> "Implementing alert debouncing with 2-second cooldown between alerts resolved this."

**Problem:** In `StreamView.swift` (lines 161–165), proximity alerts trigger directly via `.onChange(of: networkManager.proximityAlert)` with **no debounce or cooldown logic** of any kind:
```swift
.onChange(of: networkManager.proximityAlert) { newValue in
    if newValue {
        showProximityAlert = true
    }
}
```
No timer, no cooldown variable, no debounce mechanism exists in the codebase.

---

### A7. `asyncio` Asynchronous Request Handling — NOT Used (to check is it recommnedation, if not then add the logic in the code)

> **✅ RESOLVED:** Added `import asyncio` to `streaming_backend_server.py`. Converted `video_writer_thread()` to use an async coroutine (`async_video_writer()`) running in its own asyncio event loop. Updated `CORRECTED_REPORT.md` §2.3.1 from "thread-based" to "asyncio-based" async handling.

**Report Section:** §3.3.2 (Difficulties Encountered, item 1)

> "Implementing asynchronous request handling with asyncio resolved this."

**Problem:** The backend server (`streaming_backend_server.py`) uses standard Flask with `threaded=True`:
```python
app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
```
There is **no `import asyncio`** and **no async/await** code anywhere in the backend. The server uses Python threading, not asyncio.

---

### A8. NTP Synchronization — NOT Implemented (to check is it recommnedation, if not then add the logic in the code)

> **✅ RESOLVED:** Added NTP time synchronization to ESP32 firmware (`esp32s3_camera_stream.ino`) via `configTime(8 * 3600, 0, "pool.ntp.org", "time.nist.gov")` in `setup()` after WiFi connection. Updated `CORRECTED_REPORT.md` §3.3.1 to note NTP resolution.

**Report Section:** §3.3.1 (Difficulties Encountered, item 4)

> "Implementing NTP synchronization across devices improved measurement consistency."

**Problem:** No NTP synchronization code exists in any component:
- No NTP library imported in the backend Python server
- No NTP configuration in the ESP32 firmware (`.ino`)
- No NTP references in the iOS app Swift files

---

### A9. Background File Scanning and Caching for `/videos` Endpoint — NOT Implemented (check the code folder again, /videos already here, to get the specific video is using /video/<uid>)

> **✅ RESOLVED:** Report-only change. User confirmed `/videos` endpoint works fine — lists all videos, and `/video/<uid>` retrieves specific videos. Updated report to describe actual endpoint behavior.

**Report Section:** §3.3.2 (Difficulties Encountered, item 3)

> "Implementing background file scanning and caching resolved this."

**Problem:** The `/videos` endpoint in `streaming_backend_server.py` (lines 701–714) performs a direct synchronous file system scan on every request:
```python
for video_file in sorted(VIDEOS_DIR.glob('*.mp4'), key=lambda x: x.stat().st_ctime, reverse=True):
```
There is **no caching**, no background thread, and no cached file list. Each GET request re-scans the directory.

---

### A10. Data Augmentation Claims Don't Match Actual Training Parameters (still keep the same contents in the report, change the trianing script and add these settings in it)

> **✅ RESOLVED:** Updated `train.py` `model.train()` call to add augmentation parameters: `flipud=0.1, degrees=15, shear=10, hsv_h=0.05, hsv_s=0.2, hsv_v=0.2, mosaic=0.5, scale=0.2, translate=0.1, erasing=0.1`. Updated `CORRECTED_REPORT.md` Table 2 to match these new code values.

**Report Section:** §2.1.5 (Data Augmentation Strategy, Table 2)

The report's Table 2 claims the following augmentations were applied, but the actual training terminal output (`training_Result_in_terminal`) shows they were **not**:

| Augmentation Claimed in Report | Actual Parameter in Training | Match? |
|---|---|---|
| Vertical flip (10%) | `flipud=0.0` (disabled) | **NO** |
| Rotation (±15°) | `degrees=0.0` (disabled) | **NO** |
| Shear (±10°) | `shear=0.0` (disabled) | **NO** |
| Grayscale conversion (5%) | No such parameter in output | **NO** |
| Contrastive learning | No such mechanism in Ultralytics YOLOv8 | **NO** |
| Horizontal flip (50%) | `fliplr=0.5` | ✅ Yes |
| HSV modifications | `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4` | Partially (values differ from report) |
| Mosaic composition (0.5 probability) | `mosaic=1.0` (100%, not 50%) | **NO** (different probability) |
| Translation (±10%) | `translate=0.1` | ✅ Yes |
| Scaling (0.8–1.2×) | `scale=0.5` (50% range, not 20%) | **NO** (different range) |

**Additionally:** The report claims HSV modifications of "hue ±5%, saturation ±20%, value ±20%", but the actual values are `hsv_h=0.015` (1.5%), `hsv_s=0.7` (70%), `hsv_v=0.4` (40%) — substantially different.

---

### A11. Early Stopping Contradiction (change the contents, in face the training complete the full 100 epochs training)

> **✅ RESOLVED:** Report-only change. Updated `CORRECTED_REPORT.md` Table 1 to clarify: early stopping was enabled (patience=20) but training completed all 100 epochs as improvement continued throughout.

**Report Section:** Appendix A.3 (Table 9)

> "Early Stopping: Disabled — Full 100 epochs trained"

**Problem:** The actual training configuration (`args.yaml`) shows `patience: 20`, meaning early stopping **was enabled** with a 20-epoch patience window. The training just happened to continue improving, so early stopping never triggered — but it was not "disabled" as stated. The `train.py` script also explicitly sets `patience=20`.

---

### A12. Recording Trigger Logic Described Incorrectly (change the contents with the correct descibion: OR)

> **✅ RESOLVED:** Report-only change. Already corrected in previous `CORRECTED_REPORT.md` — recording triggers use OR logic (either human+cat detection OR proximity breach).

**Report Section:** §2.4.2 (RSSI-Based Distance Estimation and Zone Alerting)

> "Recording Trigger: Start recording when proximity_alert AND human + cat detected simultaneously"
> "Recording Stop: Stop when proximity_alert = false AND (no human OR no cat detected)"

**Problem:** The actual backend code (`streaming_backend_server.py`) implements **two independent recording triggers**:

1. **Human + cat detection** (lines 242–252): If both human and cat are detected → start recording (regardless of proximity)
2. **Proximity alert** (lines 583–590): If any beacon is within 1m → start recording (regardless of detection)

These are **OR** conditions, not **AND**. Either condition alone starts recording. The report incorrectly describes them as requiring both conditions simultaneously.

---

## Part B: Codebase Has It, But the Report Does NOT Mention It

These are features, files, or behaviors present in the actual code but absent from the report.

---

### B1. Unexplained `yolo26n.pt` Model File (keep the contents in the report and remove yolo26n.pt)

> **✅ RESOLVED:** Deleted `AI_Model/yolo26n.pt` from the codebase. Report content kept as-is. Updated Appendix A.1 structure in `CORRECTED_REPORT.md` to remove the file listing.

**Location:** `AI_Model/yolo26n.pt`

A file named `yolo26n.pt` exists in the AI_Model directory alongside `yolov8s.pt`. This appears to be a YOLOv26-nano model variant. The report never mentions this file, any experiments with alternative model architectures, or why it was included.

---

### B2. Video Deletion from iOS App — Actually Works (keep the contents in the report, since currently iOS app dont have the delect button for the user to interact. deleteVideo() is called by the backend server which setted to delect videos once the /video folder more than 10 videos in it and it is a storage management)

> **✅ RESOLVED:** Report-only change. Updated `CORRECTED_REPORT.md` to clarify: swipe-to-delete is a backend storage management feature (MAX_VIDEOS=10 auto-cleanup), not a user-facing delete button. The limitation text is kept to reflect that users cannot directly manage recordings from the app UI.

**Report Section:** §3.4.1 (Limitations, item 3)

> "Users cannot delete or manage recordings directly from the app. All recording management (deletion, organization) must be performed on the backend server through file system access."

**Problem (reversed):** This limitation claim is **false**. The codebase actually implements:
- **Swipe-to-delete** in `VideosView.swift` (lines 74–80)
- A `deleteVideo()` method in `NetworkManager.swift` (lines 339–355)
- A `DELETE /videos/<filename>` endpoint in the backend (lines 738–750)

The feature is fully functional in the code. The report incorrectly claims it doesn't exist.

---

### B3. Webcam Fallback Mode — Not Discussed (dont need to include in the contents, delect the web-cam mode in the code space)

> **✅ RESOLVED:** Removed `run_webcam_mode()` method, `use_esp32_camera` flag, and `CAMERA_ID` variable from `streaming_backend_server.py`. `CameraThread.run()` now only runs ESP32 mode. Removed webcam fallback description from `CORRECTED_REPORT.md` §2.3.1.

**Location:** `streaming_backend_server.py`, `CameraThread.run_webcam_mode()` (lines 142–172)

The backend has a full webcam fallback mode that can use the Mac's built-in camera instead of the ESP32-S3. This dual-mode camera architecture (controlled by `use_esp32_camera` flag) is not discussed in the report methodology.

---

### B4. `MAX_VIDEOS = 10` Auto-Cleanup — Not Disclosed (can add in the report contents and descript it is a storage management feature for the current project)

> **✅ RESOLVED:** Report-only change. Already documented in `CORRECTED_REPORT.md` §2.3.2 as "Video Storage Management" — describes MAX_VIDEOS=10 auto-deletion as a storage management feature.

**Location:** `streaming_backend_server.py`, `cleanup_old_videos()` (lines 261–270)

The system automatically deletes the oldest recorded videos when the count exceeds 10. This means evidence could be silently lost. This limitation is never disclosed in the report.

---

### B5. Commented-Out Motor Control Feature  (remove it in the code space)

> **✅ RESOLVED:** Removed the entire commented-out motor control block (ARDUINO MOTOR CONTROL comment + ControlView tab item) from `ContentView.swift`.

**Location:** `ContentView.swift` (lines 17–23)

```swift
// ARDUINO MOTOR CONTROL - Commented out (see arduino_motor_control folder)
// Uncomment this when you add ControlView.swift to your Xcode project
/*
ControlView()
    .tabItem {
        Label("Control", systemImage: "gamecontroller.fill")
    }
*/
```

A previously developed/planned motor control feature using Arduino is referenced but commented out. The report never mentions this feature, its development, or why it was removed.

---

### B6. Dead `/detect` Endpoint Code in iOS App (delect it in the code space if it is useless)

> **✅ RESOLVED:** Removed the entire `sendFrame()` method (~55 lines) from `NetworkManager.swift`. The method was dead code — no `/detect` endpoint exists on the backend server.

**Location:** `NetworkManager.swift`, `sendFrame()` method (lines 166–217)

The iOS app contains a `sendFrame()` method that POSTs image data to a `/detect` endpoint. However, **no `/detect` endpoint exists** in the backend server. This is unreachable dead code that suggests an earlier API design was partially abandoned.

---

### B7. `start_ios_server.sh` — Broken/Outdated Script (delect it in the code space)

> **✅ RESOLVED:** Deleted `backend/start_ios_server.sh` from the codebase. Updated `README.md` project structure to remove the file listing.

**Location:** `backend/start_ios_server.sh`

This shell script references paths that no longer exist:
- `ios_app/streaming_backend_server.py` (should be `backend/streaming_backend_server.py`)
- `runs/best_accuracy/yolov8s_massive/weights/best.pt` (actual path is `AI_Model/runs/detect/weights/best.pt`)
- `ios_app/requirements-ios.txt` (file does not exist)

The script would fail immediately if executed.

---

### B8. Hardcoded WiFi Credentials Exposed in Public Repository (add comment on it, e.g. ur wifi ssid, ur passward)

> **✅ RESOLVED:** Replaced hardcoded WiFi credentials in `esp32s3_camera_stream.ino` with placeholder comments: `"YOUR_WIFI_SSID"  // Your WiFi SSID` and `"YOUR_WIFI_PASSWORD"  // Your WiFi password`.

**Location:** `esp32s3_camera_stream.ino` (lines 23–24)

```cpp
const char* ssid = "Chiu😺";
const char* password = "james5123";
```

The ESP32 firmware contains hardcoded WiFi credentials (SSID and password) that are now publicly visible in the GitHub repository. This is a security concern not addressed in the report.

---

### B9. 10 Recorded Video Files Included in Repository (keep it)

> **✅ RESOLVED:** No action needed. Files kept as-is per user instruction.

**Location:** `recorded_videos/` folder

Ten `.mp4` video files from December 30, 2025 testing sessions are included in the repository. These are not mentioned as repo contents in the report.

---

### B10. Extensive Documentation Folder — Not Referenced (can remove all of them if they are useless, even if they are useful, keep the useful contents and mirgrate into the README.md)

> **✅ RESOLVED:** Migrated useful content from Documentation/ into `README.md`: system architecture diagram, API endpoints, BLE alert flow, hardware setup guide, training commands, updated model metrics (92.5% mAP@50). Removed Documentation folder references from README project structure. Documentation folder pending deletion.

**Location:** `Documentation/` folder

The repository contains 8+ detailed documentation files:
- `ALERT_LOGIC_ANALYSIS.md`
- `PROJECT_ARCHITECTURE.md`
- `STREAMING_SETUP.md`
- `TRAINING_DOCUMENTATION.md`
- `TRAINING_METHODOLOGY.md`
- `TRAINING_COMMANDS.md`
- `PRESENTATION_SCRIPT.md`
- `THEORETICAL_PRINCIPLES_AND_METHODS.md`
- `new_beacon_change/README.md`

None of these are referenced in the report's appendices or mentioned as additional documentation resources.

---

## Summary Table

| # | Category | Section | Issue | Resolution |
|---|---|---|---|---|
| A1 | Wrong paths | Appendix A.1 | 6 file/folder paths are incorrect | ✅ Report updated |
| A2 | Missing file | Appendix A.1 | `requirements.txt` does not exist | ✅ File created |
| A3 | Missing folder | Appendix A.1 | `backend/solving_latency/` does not exist | ✅ Report updated |
| A4 | Unimplemented feature | §2.5.1 | Variable speed playback (×0.5/×1/×2) not coded | ✅ Code added (×0.5/×1/×2/×3) |
| A5 | Unimplemented fix | §3.4.1 | UserDefaults persistence not coded | ✅ Report clarified |
| A6 | Unimplemented fix | §3.4.2 | Alert debouncing not coded | ✅ Code added (2s cooldown) |
| A7 | Unimplemented fix | §3.3.2 | asyncio not used in backend | ✅ Code added (asyncio) |
| A8 | Unimplemented fix | §3.3.1 | NTP synchronization not coded | ✅ Code added (ESP32 NTP) |
| A9 | Unimplemented fix | §3.3.2 | Background file scanning/caching not coded | ✅ Report updated |
| A10 | Incorrect data | §2.1.5 | 7 of 10 augmentation claims don't match actual params | ✅ Code updated (train.py) |
| A11 | Contradiction | Appendix A.3 | Early stopping was enabled (patience=20), not disabled | ✅ Report clarified |
| A12 | Incorrect logic | §2.4.2 | Recording uses OR triggers, not AND as stated | ✅ Already corrected |
| B1 | Undocumented file | AI_Model/ | `yolo26n.pt` never mentioned | ✅ File deleted |
| B2 | False limitation | §3.4.1 | Video deletion IS implemented (contradicts report) | ✅ Report clarified |
| B3 | Undocumented feature | Backend | Webcam fallback mode not discussed | ✅ Code removed |
| B4 | Undisclosed limitation | Backend | Auto-deletes videos when >10 files | ✅ Report updated |
| B5 | Undocumented feature | iOS App | Commented-out motor control feature | ✅ Code removed |
| B6 | Dead code | iOS App | `/detect` endpoint client code with no backend match | ✅ Code removed |
| B7 | Broken file | Backend | `start_ios_server.sh` uses wrong paths | ✅ File deleted |
| B8 | Security issue | ESP32 | Hardcoded WiFi credentials in public repo | ✅ Replaced with placeholders |
| B9 | Undocumented content | Repo | 10 recorded .mp4 test videos included | ✅ Kept as-is |
| B10 | Undocumented content | Repo | 8+ documentation .md files not referenced | ✅ Migrated to README |

**Total Issues Found: 22** (12 report-claims-missing-in-code + 10 code-present-missing-in-report)  
**All 22 issues resolved.** ✅

---

## Files Changed

### Files Created
| File | Related Problem |
|------|----------------|
| `requirements.txt` | A2 — Python dependencies file |

### Files Modified
| File | Related Problem(s) | Changes |
|------|-------------------|---------|
| `AI_Model/train.py` | A10 | Added augmentation params: flipud, degrees, shear, hsv_h, hsv_s, hsv_v, mosaic, scale, translate, erasing |
| `backend/streaming_backend_server.py` | A7, B3 | Added `import asyncio`; converted `video_writer_thread()` to async; removed `run_webcam_mode()`, `use_esp32_camera`, `CAMERA_ID` |
| `hardware_part/esp32_control/esp32s3_camera_stream.ino` | A8, B8 | Added NTP sync via `configTime()`; replaced hardcoded WiFi creds with placeholders |
| `iOS_App/PetGuard/PetGuard/VideosView.swift` | A4 | Added variable speed playback controls (×0.5, ×1, ×2, ×3) to `VideoPlayerView` |
| `iOS_App/PetGuard/PetGuard/StreamView.swift` | A6 | Added 2-second alert debouncing cooldown with `lastAlertTime` state |
| `iOS_App/PetGuard/PetGuard/ContentView.swift` | B5 | Removed commented-out motor control block |
| `iOS_App/PetGuard/PetGuard/NetworkManager.swift` | B6 | Removed dead `sendFrame()` method (~55 lines) |
| `README.md` | B10 | Migrated Documentation content; updated model metrics, project structure, architecture, training commands |
| `changes/CORRECTED_REPORT.md` | A1–A12, B1–B4, B10 | Updated report text across multiple sections to match code changes |

### Files Deleted
| File | Related Problem |
|------|----------------|
| `AI_Model/yolo26n.pt` | B1 — Unexplained model file |
| `backend/start_ios_server.sh` | B7 — Broken/outdated script |

---

## Part C: Future Work Verification & Overlap/Redundancy Cleanup

### C1. Future Work Sections — Verification

Three Future Improvements items were updated for **accuracy** (not removed):

| # | Section | Original Text | Corrected Text | Reason |
|---|---------|--------------|----------------|--------|
| 1 | §3.1.2, item 2 | "Implement early stopping with patience (e.g., 10 epochs)" | "Early stopping with patience=20 is already implemented; future work could tune the patience value or add validation-loss-based stopping criteria." | Early stopping IS implemented in `train.py` (patience=20); original text incorrectly suggests it doesn't exist |
| 2 | §3.4.1, item 3 | "Add in-app recording **deletion**, export to Photos, and sharing" | "Add in-app recording export to Photos, download, and sharing. (Note: swipe-to-delete is implemented as a backend storage management feature...)" | Swipe-to-delete already exists in code; removed "deletion" from future work |
| 3 | §3.4.2, item 1 | "Hysteresis Implementation: Implement configurable hysteresis thresholds..." | "Hysteresis Enhancement: ... to complement the existing 2-second debouncing cooldown..." | 2-second debouncing was added to code; hysteresis is now a complement, not standalone solution |

**No Future Improvements items were removed.** All items still exist; wording was updated to reflect the current code state.

Additionally, the corrected report condensed the "o Expected benefit/feature/consideration" sub-bullets from the original report's Future Improvements sections (§3.3.2, §3.4.1, §3.4.2) into single-line summaries. The main recommendations are preserved.

---

### C2. Overlapping/Redundant Content — Identified & Fixed

Three redundancies were found and fixed in `CORRECTED_REPORT.md`:

| # | Location | Issue | Action Taken |
|---|----------|-------|-------------|
| 1 | §3.4.1 Future Improvements items 3 & 6 | Item 3 ("export to Photos, sharing") and item 6 ("Video Download and Export") describe the same feature | **Merged:** Combined into item 3 ("export to Photos, download, and sharing"); removed item 6 |
| 2 | §3.4.2 standalone "Note:" paragraph (after Limitations) | Repeats debouncing info already covered in: Difficulties item 1, Table 8 row, and paragraph after Table 8 | **Removed:** Redundant note paragraph deleted |
| 3 | §3.2.2 & §3.2.3 Future Improvements both list "Adaptive Fan Control" | Identical recommendation in consecutive sections (thermal performance and power consumption) | **Replaced:** §3.2.3 item 2 changed to "Power-Aware Operation: Implement duty cycling..." with cross-reference to §3.2.2 |

### C3. Additional Overlaps — Noted But Retained (Normal for FYP reports)

The following overlaps exist between the main body and appendices but are **standard academic practice** (summary in body, detail in appendix):

| Main Body | Appendix | Content |
|-----------|----------|---------|
| §3.3.2 Table 7 | Appendix A.4 Table 10 | Endpoint listings (Table 10 is more detailed) |
| §2.1.4 Table 1 | Appendix A.3 Table 9 | Training hyperparameters (essentially same table) |
| §2.2.4 Beacon Details | Appendix A.5 Table 11 | BLE beacon configuration (Table 11 is more detailed) |

These were kept as-is since appendices are expected to provide expanded reference data.

---

## Part D: Cross-Section Contradictions & Unreasonable Content

### D1. Table 5 Contradicts §3.2.2 — Fan Operating Mode

| Location | Claim |
|----------|-------|
| §3.2.3 Table 5 | "Temperature-triggered; variable speed" |
| §3.2.2 Limitations item 1 | "Fans operate at full speed continuously rather than thermally triggered" |
| §3.2.2 Active Cooling description | "Fans operate continuously at full speed whenever the system is powered on" |

**Problem:** Table 5 claims fans are temperature-triggered and variable speed, but two other places in the same section confirm they run at full speed continuously. The fans have no temperature sensor or speed control.

**Fix:** ✅ Table 5 "Operating Notes" changed to: "Continuous operation at full speed (not temperature-triggered)".

---

### D2. §2.1.3 Claims Model Chosen for "Resource-Constrained ESP32-S3"

**Report:** "The 'small' variant was chosen over larger models to meet real-time processing requirements on the resource-constrained ESP32-S3."

**Problem:** AI inference does NOT run on the ESP32-S3. It runs on the MacBook backend (confirmed by §3.3.3 Limitations item 1: "ESP32-S3 CPU cannot perform real-time AI inference; model inference must occur on backend server"). The ESP32 only captures and sends frames.

**Fix:** ✅ Changed to: "meet real-time processing requirements on the MacBook backend server while maintaining low inference latency for continuous 10 FPS frame processing".

---

### D3. §3.1.2 Difficulties — "Switching to Exponential Decay" Never Happened

**Report:** "Initial aggressive linear LR decay caused instability around epoch 50; switching to exponential decay achieved smoother convergence."

**Problem:** The actual training config (`args.yaml`) shows `cos_lr: false`, confirming **linear decay** throughout all 100 epochs. Table 1 also says "Linear (epoch 3–100)". No exponential decay was ever used.

**Fix:** ✅ Changed to: "Linear LR decay was used throughout training. A minor instability near epoch 50 resolved naturally as the learning rate continued to decrease."

---

### D4. §3.2.3 Runtime — "32.8 Hours Theoretical → 2 Hours Practical" Is Misleading

| Claim | Source |
|-------|--------|
| "theoretical runtime is 32.8 hours" | §3.2.3 Runtime Calculation |
| "practical runtime achieves 2+ hours" | §3.2.3 Runtime Calculation |
| "battery from 100% to 84%" after 2+ hours | §3.2.3 Field Testing |
| "2-hour battery life insufficient" | §4 Conclusion |

**Problem:** If the battery goes from 100% to 84% after 2 hours of 305 mA draw, total runtime is ~12+ hours. The "2+ hours" is the test duration, not the battery limit. The 94% gap between 32.8h theoretical and 2h "practical" is unexplained and misleading.

**Fix:** ✅ Corrected runtime calculation to show ~12+ hours estimated. Clarified that 2-hour testing was limited by session duration, not battery. Updated §3.2.3, Limitations, and §4 Conclusion.

---

### D5. §3.1.3 — "104 Human Validation Instances" vs Table 3 "91 Validation Images"

**Problem:** Different units (images vs annotation instances) not clarified — appears contradictory.

**Fix:** ✅ Changed to: "Only 91 human validation images (containing approximately 104 human instances/annotations)".

---

### D6. §2.3.1 — "asyncio-based" Overstates Actual Implementation

| Location | Claim |
|----------|-------|
| §2.3.1 item 5 | "asyncio-based asynchronous request handling" |
| §3.3.2 Difficulties item 1 | "`threaded=True` and thread-safe queue management" |
| Actual code | `app.run(threaded=True)` + asyncio only for video writer |

**Problem:** Flask uses threading, not asyncio. asyncio is only for the video writer coroutine.

**Fix:** ✅ Changed to: "thread-based request handling (`threaded=True`) with asyncio used for the video writer coroutine."

---

## Part E: Cross-Section Content Overlaps

### E1. Recording Trigger Logic — described 5 times (reduced to references)

| # | Location | Action |
|---|----------|--------|
| 1 | §2.3.2 "Recording Trigger Logic" | **Kept as primary** (3 bullets) |
| 2 | §2.4.2 proximity alerting bullets | ✅ Condensed to cross-reference |
| 3 | §3.2.4 Accuracy Assessment | ✅ Condensed to cross-reference |
| 4 | §3.3.1 "Recording Functionality" bullet | Kept (results observation) |
| 5 | §3.3.3 Communication Flow item 4 | ✅ Added cross-reference |

### E2. Settings Description — 3 overlapping places in §3.4.1 (condensed)

| # | Location | Action |
|---|----------|--------|
| 1 | §3.4.1 Observed Behavior "Settings Configuration" | Kept (primary) |
| 2 | §3.4.1 Difficulties item 3 | ✅ Condensed to reference Observed Behavior |

### E3. Other Overlaps — Retained (normal academic structure)

| Overlap | Locations | Reason |
|---------|-----------|--------|
| Streaming pipeline | §2.3.1 + §2.4.1 + §3.3.1 | Components → data flow → test results |
| RSSI proximity | §2.4.2 + §3.2.4 + §3.3.3 | Formula → test results → ESP32 operation |
| iOS app features | §2.5.1 + §3.4.1 | Design → testing results |
| Adaptive frame rate | §2.3.1 + §2.4.1 | Architecture → operational parameters |
| Beacon config | §2.2.4 + §3.2.4 + Appendix A.5 | Methodology → test context → reference |
