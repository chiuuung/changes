from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from flask_sock import Sock
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
from datetime import datetime
import threading
import queue
import time
import json
import struct
import asyncio

app = Flask(__name__)
CORS(app)
sock = Sock(app)

# Configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent.absolute()  # Go up 3 levels to reach workspace root
MODEL_PATH = PROJECT_ROOT / "AI_Model" / "runs" / "detect" / "weights" / "best.pt"
VIDEOS_DIR = PROJECT_ROOT / "recorded_videos"

VIDEOS_DIR.mkdir(exist_ok=True)

# Video write queue (for async writing)
video_write_queue = queue.Queue(maxsize=3)

# Global variables
model = None
esp32_frame_queue = queue.Queue(maxsize=1)
# Camera source: ESP32-S3 only

# Latency optimization settings
INITIAL_FPS = 15
MIN_FPS = 10
MAX_FPS = 25
FRAME_SKIP_THRESHOLD = 0.5  # Skip frames if queue exceeds this depth

# Network performance tracking
network_stats = {
    "queue_depth": 0,
    "current_fps": INITIAL_FPS,
    "latency_ms": 0,
    "frame_size_bytes": 0,
    "websocket_clients": 0,
    "last_adjustment": time.time()
}

recording_state = {
    "is_recording": False,
    "video_writer": None,
    "current_filename": None,
    "last_detection_time": time.time(),
    "both_detected": False,
    "latest_frame": None,
    "latest_annotated_frame": None,
    "detections": []
}

proximity_state = {
    "distance": 999.0,
    "rssi": -100,
    "is_close": False,
    "last_update": time.time(),
    "beacon_mac": None,
    "beacon_id": None,
    "danger_beacon_ids": [],
    "danger_beacon_labels": [],
    "danger_beacons": {},
    "beacons": {},
    "rssi_at_1m": -59,
    "path_loss_exponent": 2.5,
    "proximity_recording": False,
    "proximity_alert_active": False
}

BEACON_LABELS = {
    "beacon_1": "Room 1",
    "beacon_2": "Room 2"
}

CONFIDENCE_THRESHOLD = 0.25
COOLDOWN_SECONDS = 2
MAX_VIDEOS = 10

# WebSocket client tracking
websocket_clients = set()
ws_lock = threading.Lock()


class AdaptiveFrameRateController:
    """Dynamically adjust FPS based on network conditions"""
    
    def __init__(self, initial_fps=INITIAL_FPS):
        self.target_fps = initial_fps
        self.adjustment_interval = 2.0  # Check every 2 seconds
        self.last_adjustment = time.time()
        self.frame_times = []
    
    def should_skip_frame(self):
        """Skip frame if queue is backing up"""
        queue_depth = esp32_frame_queue.qsize()
        if queue_depth > FRAME_SKIP_THRESHOLD * esp32_frame_queue.maxsize:
            return True
        return False
    
    def adjust_fps(self, queue_depth):
        """Adjust FPS based on queue depth and latency"""
        current_time = time.time()
        
        if current_time - self.last_adjustment < self.adjustment_interval:
            return self.target_fps
        
        self.last_adjustment = current_time
        
        # If queue is backing up, reduce FPS
        if queue_depth > 0.75 * esp32_frame_queue.maxsize:
            self.target_fps = max(MIN_FPS, self.target_fps - 2)
        # If queue is healthy and we have headroom, increase FPS
        elif queue_depth < 0.25 * esp32_frame_queue.maxsize and self.target_fps < MAX_FPS:
            self.target_fps = min(MAX_FPS, self.target_fps + 1)
        
        network_stats["current_fps"] = self.target_fps
        return self.target_fps
    
    def get_frame_delay(self):
        """Get delay in seconds for current FPS"""
        return 1.0 / self.target_fps


frame_rate_controller = AdaptiveFrameRateController(INITIAL_FPS)


class CameraThread(threading.Thread):
    """Optimized camera capture and detection thread"""
    
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.camera = None
        self.frame_count = 0
    
    def run(self):
        """Main camera loop - ESP32 mode"""
        self.running = True
        self.run_esp32_mode()
    
    def run_esp32_mode(self):
        """Use ESP32-S3 camera frames with optimized processing"""
        last_frame_time = time.time()
        
        while self.running:
            try:
                # Get frame with timeout
                frame = esp32_frame_queue.get(timeout=1.0)
                self.frame_count += 1
                
                # Skip frame if network is congested
                if frame_rate_controller.should_skip_frame():
                    continue
                
                # Adjust FPS based on queue depth
                queue_depth = esp32_frame_queue.qsize()
                frame_rate_controller.adjust_fps(queue_depth)
                
                # Run detection
                results = model.predict(
                    source=frame,
                    conf=CONFIDENCE_THRESHOLD,
                    iou=0.45,
                    imgsz=640,
                    half=False,
                    verbose=False
                )
                
                self.process_detections(frame, results[0])
                
                # Adaptive delay
                frame_delay = frame_rate_controller.get_frame_delay()
                elapsed = time.time() - last_frame_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
                last_frame_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                time.sleep(0.1)
    
    def process_detections(self, frame, result):
        """Process YOLOv8 detections (null class filtered out)"""
        boxes = result.boxes
        detections = []
        has_human = False
        has_cat = False
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # AI_Model_v3 classes: 0=cat, 1=human
            if cls == 0:
                class_name = "cat"
            elif cls == 1:
                class_name = "human"
            else:
                continue
            
            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": box.xyxy[0].tolist()
            })
            
            if class_name == "human":
                has_human = True
            elif class_name == "cat":
                has_cat = True
        
        # Create annotated frame
        annotated_frame = result.plot()
        
        # Log detections
        if len(detections) > 0:
            detection_summary = ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in detections])
            print(f"🎯 Detections: {detection_summary}")
        
        # Update recording state
        recording_state["latest_frame"] = frame
        recording_state["latest_annotated_frame"] = annotated_frame
        recording_state["detections"] = detections
        
        both_present = has_human and has_cat
        
        if both_present:
            recording_state["last_detection_time"] = time.time()
            recording_state["both_detected"] = True
            
            if not recording_state["is_recording"] and not proximity_state["proximity_recording"]:
                recording_state["is_recording"] = True
                start_recording(frame.shape)
        
        if recording_state["is_recording"]:
            try:
                video_write_queue.put_nowait(frame.copy())
            except queue.Full:
                pass
        
        if recording_state["is_recording"] and not proximity_state["proximity_recording"]:
            check_recording_timeout()
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False


async def async_video_writer():
    """Async coroutine for writing video frames"""
    frames_written = 0
    
    while True:
        try:
            frame = video_write_queue.get_nowait()
            
            if recording_state["video_writer"]:
                recording_state["video_writer"].write(frame)
                frames_written += 1
            
        except queue.Empty:
            if frames_written > 0:
                frames_written = 0
            await asyncio.sleep(0.01)
            continue
        except Exception as e:
            await asyncio.sleep(0.1)


def video_writer_thread():
    """Background thread running async video writer"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_video_writer())


def cleanup_old_videos():
    """Delete old videos, keep only MAX_VIDEOS newest"""
    try:
        videos = sorted(VIDEOS_DIR.glob("*.mp4"), key=lambda p: p.stat().st_ctime, reverse=True)
        
        if len(videos) > MAX_VIDEOS:
            videos_to_delete = videos[MAX_VIDEOS:]
            for video in videos_to_delete:
                video.unlink()
    except Exception as e:
        pass


def start_recording(frame_shape):
    """Start a new video recording"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interaction_{timestamp}.mp4"
    filepath = VIDEOS_DIR / filename
    
    height, width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30.0
    
    recording_state["video_writer"] = cv2.VideoWriter(
        str(filepath), fourcc, fps, (width, height)
    )
    recording_state["current_filename"] = filename


def stop_recording():
    """Stop and save the current video"""
    if recording_state["video_writer"]:
        time.sleep(0.2)
        recording_state["video_writer"].release()
        recording_state["video_writer"] = None
        recording_state["current_filename"] = None
        
        while not video_write_queue.empty():
            try:
                video_write_queue.get_nowait()
            except queue.Empty:
                break
        
        cleanup_old_videos()


def check_recording_timeout():
    """Check if we should stop recording due to timeout"""
    if recording_state["is_recording"]:
        time_since_detection = time.time() - recording_state["last_detection_time"]
        if time_since_detection > COOLDOWN_SECONDS:
            recording_state["is_recording"] = False
            recording_state["both_detected"] = False
            stop_recording()


def load_model():
    """Load YOLOv8 model"""
    global model
    model = YOLO(str(MODEL_PATH))
    
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_frame, verbose=False)


def frame_to_binary_jpeg(frame, quality=80):
    """Convert frame to binary JPEG (more efficient than base64)"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


@sock.route('/ws/stream')
def websocket_stream(ws):
    """WebSocket endpoint for low-latency frame streaming"""
    
    with ws_lock:
        websocket_clients.add(ws)
        network_stats["websocket_clients"] = len(websocket_clients)
    
    try:
        frame_count = 0
        start_time = time.time()
        
        while True:
            if recording_state["latest_annotated_frame"] is None:
                time.sleep(0.05)
                continue
            
            # Send binary JPEG (33% smaller than base64)
            frame = recording_state["latest_annotated_frame"]
            jpeg_data = frame_to_binary_jpeg(frame, quality=80)
            
            # Send frame size + data
            frame_size = len(jpeg_data)
            header = struct.pack('<I', frame_size)  # 4-byte size header
            
            try:
                ws.send(header + jpeg_data, binary=True)
                frame_count += 1
                
                # Update network stats
                network_stats["frame_size_bytes"] = frame_size
                network_stats["latency_ms"] = (1000.0 / network_stats["current_fps"]) * 1.2
            
            except Exception as e:
                break
            
            # Adaptive delay
            time.sleep(0.001)  # Small delay to allow frame updates
    
    except Exception as e:
        pass
    
    finally:
        with ws_lock:
            websocket_clients.discard(ws)
            network_stats["websocket_clients"] = len(websocket_clients)


@app.route('/esp32/frame', methods=['POST'])
def receive_esp32_frame():
    """Receive JPEG frame from ESP32-S3 (binary optimized)"""
    try:
        jpeg_data = request.data
        
        if len(jpeg_data) == 0:
            return jsonify({"error": "No image data"}), 400
        
        # Decode JPEG
        nparr = np.frombuffer(jpeg_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # Add to queue (non-blocking)
        try:
            esp32_frame_queue.put_nowait(frame)
        except queue.Full:
            pass
        
        return jsonify({
            "success": True,
            "frame_received": True,
            "frame_size": frame.shape,
            "queue_depth": esp32_frame_queue.qsize(),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/esp32/distance', methods=['POST'])
def receive_distance_data():
    """Receive BLE beacon distance from ESP32-S3"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        rssi_at_1m = int(data.get('rssi_at_1m', -59))
        path_loss_exponent = float(data.get('path_loss_exponent', 2.5))

        raw_beacons = data.get('beacons', [])
        normalized_beacons = []

        if not raw_beacons and ('distance' in data or 'rssi' in data or 'beacon_mac' in data):
            normalized_beacons.append({
                'beacon_id': data.get('beacon_id', 'beacon_1'),
                'beacon_mac': data.get('beacon_mac', 'unknown'),
                'distance': float(data.get('distance', 999.0)),
                'rssi': int(data.get('rssi', -100))
            })
        else:
            for beacon in raw_beacons:
                normalized_beacons.append({
                    'beacon_id': beacon.get('beacon_id', 'unknown'),
                    'beacon_mac': beacon.get('beacon_mac', 'unknown'),
                    'distance': float(beacon.get('distance', 999.0)),
                    'rssi': int(beacon.get('rssi', -100))
                })

        if len(normalized_beacons) == 0:
            return jsonify({"error": "No beacon data in payload"}), 400

        beacon_state_map = {}
        danger_beacon_ids = []
        danger_beacons = {}

        for beacon in normalized_beacons:
            beacon_id = beacon['beacon_id']
            beacon_distance = beacon['distance']
            beacon_rssi = beacon['rssi']
            beacon_mac = beacon['beacon_mac']
            beacon_is_close = beacon_distance <= 1.0

            beacon_state_map[beacon_id] = {
                "beacon_id": beacon_id,
                "label": BEACON_LABELS.get(beacon_id, beacon_id.replace('_', ' ').title()),
                "beacon_mac": beacon_mac,
                "distance": beacon_distance,
                "rssi": beacon_rssi,
                "is_close": beacon_is_close,
                "updated_at": time.time()
            }

            if beacon_is_close:
                danger_beacon_ids.append(beacon_id)
                danger_beacons[beacon_id] = beacon_state_map[beacon_id]

        nearest_beacon = min(normalized_beacons, key=lambda item: item['distance'])
        nearest_beacon_id = nearest_beacon['beacon_id']
        nearest_label = BEACON_LABELS.get(nearest_beacon_id, nearest_beacon_id.replace('_', ' ').title())

        # Update proximity state
        proximity_state['distance'] = nearest_beacon['distance']
        proximity_state['rssi'] = nearest_beacon['rssi']
        proximity_state['beacon_mac'] = nearest_beacon['beacon_mac']
        proximity_state['beacon_id'] = nearest_beacon_id
        proximity_state['beacons'] = beacon_state_map
        proximity_state['danger_beacons'] = danger_beacons
        proximity_state['danger_beacon_ids'] = danger_beacon_ids
        proximity_state['danger_beacon_labels'] = [
            BEACON_LABELS.get(beacon_id, beacon_id.replace('_', ' ').title())
            for beacon_id in danger_beacon_ids
        ]
        proximity_state['rssi_at_1m'] = rssi_at_1m
        proximity_state['path_loss_exponent'] = path_loss_exponent
        proximity_state['last_update'] = time.time()

        is_now_close = len(danger_beacon_ids) > 0
        proximity_state['is_close'] = is_now_close

        danger_str = ', '.join(proximity_state['danger_beacon_labels']) if proximity_state['danger_beacon_labels'] else "None"
        print(
            f"📏 Distance: Primary {nearest_label}: RSSI {nearest_beacon['rssi']} dBm, "
            f"Distance {nearest_beacon['distance']:.2f}m | Danger zones: {danger_str}"
        )
        
        # Handle proximity detection
        if is_now_close:
            proximity_state['proximity_alert_active'] = True
            
            if not proximity_state['proximity_recording']:
                if recording_state['latest_frame'] is not None:
                    proximity_state['proximity_recording'] = True
                    recording_state['is_recording'] = True
                    start_recording(recording_state['latest_frame'].shape)
                    print(f"🚨 PROXIMITY ALERT! Recording started - Danger zones: {proximity_state['danger_beacon_labels']}")
        else:
            if proximity_state['proximity_alert_active']:
                proximity_state['proximity_alert_active'] = False
            
            if proximity_state['proximity_recording']:
                proximity_state['proximity_recording'] = False
                recording_state['is_recording'] = False
                stop_recording()
        
        return jsonify({
            "success": True,
            "distance": proximity_state['distance'],
            "rssi": proximity_state['rssi'],
            "is_close": is_now_close,
            "alert_active": proximity_state['proximity_alert_active'],
            "recording": proximity_state['proximity_recording'],
            "beacon_id": proximity_state['beacon_id'],
            "beacon_label": nearest_label,
            "danger_beacon_ids": proximity_state['danger_beacon_ids'],
            "danger_beacon_labels": proximity_state['danger_beacon_labels'],
            "beacons": proximity_state['beacons'],
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check with network stats"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "recording": recording_state["is_recording"],
        "camera_active": camera_thread.running if camera_thread else False,
        "camera_source": "ESP32-S3",
        "network_stats": network_stats,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/stream/live', methods=['GET'])
def stream_live():
    """Legacy HTTP endpoint for live frame + detections"""
    if recording_state["latest_annotated_frame"] is None:
        return jsonify({"error": "No frame available"}), 404
    
    # Encode as binary and send
    frame = recording_state["latest_annotated_frame"]
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        "frame": frame_base64,
        "detections": recording_state["detections"],
        "is_recording": recording_state["is_recording"],
        "current_video": recording_state["current_filename"],
        "proximity_alert": proximity_state["proximity_alert_active"],
        "beacon_distance": proximity_state["distance"],
        "beacon_id": proximity_state.get("beacon_id"),
        "danger_beacon_ids": proximity_state.get("danger_beacon_ids", []),
        "danger_beacon_labels": proximity_state.get("danger_beacon_labels", []),
        "network_stats": network_stats,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/proximity/status', methods=['GET'])
def get_proximity_status():
    """Get current proximity status"""
    time_since_update = time.time() - proximity_state["last_update"]
    return jsonify({
        "distance": proximity_state["distance"],
        "rssi": proximity_state["rssi"],
        "is_close": proximity_state["is_close"],
        "alert_active": proximity_state["proximity_alert_active"],
        "recording": proximity_state["proximity_recording"],
        "beacon_id": proximity_state.get("beacon_id"),
        "beacon_mac": proximity_state["beacon_mac"],
        "danger_beacon_ids": proximity_state.get("danger_beacon_ids", []),
        "danger_beacon_labels": proximity_state.get("danger_beacon_labels", []),
        "danger_beacons": proximity_state.get("danger_beacons", {}),
        "beacons": proximity_state.get("beacons", {}),
        "rssi_at_1m": proximity_state.get("rssi_at_1m", -59),
        "path_loss_exponent": proximity_state.get("path_loss_exponent", 2.5),
        "calculation_method": "path_loss_formula",
        "last_update": proximity_state["last_update"],
        "time_since_update": time_since_update,
        "has_frame": recording_state["latest_frame"] is not None,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/stream/mjpeg', methods=['GET'])
def stream_mjpeg():
    """MJPEG stream (for browser compatibility)"""
    def generate():
        while True:
            if recording_state["latest_annotated_frame"] is not None:
                _, buffer = cv2.imencode('.jpg', recording_state["latest_annotated_frame"])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status', methods=['GET'])
def get_status():
    """Get current detection and recording status"""
    return jsonify({
        "is_recording": recording_state["is_recording"],
        "both_detected": recording_state["both_detected"],
        "current_video": recording_state["current_filename"],
        "detections": recording_state["detections"],
        "network_stats": network_stats,
        "timestamp": datetime.now().isoformat()
    })


@app.route('/videos', methods=['GET'])
def get_videos():
    """Get list of recorded videos"""
    videos = []
    for video_file in sorted(VIDEOS_DIR.glob('*.mp4'), key=lambda x: x.stat().st_ctime, reverse=True):
        videos.append({
            'filename': video_file.name,
            'size': video_file.stat().st_size,
            'created': datetime.fromtimestamp(video_file.stat().st_ctime).isoformat(),
            'url': f'/videos/{video_file.name}'
        })
    
    return jsonify({'videos': videos, 'count': len(videos)})


@app.route('/videos/<filename>', methods=['GET'])
def get_video(filename):
    """Download a specific video file"""
    try:
        video_path = VIDEOS_DIR / filename
        
        if not video_path.exists():
            return jsonify({"error": "Video not found"}), 404
        
        return send_file(video_path, mimetype='video/mp4', as_attachment=False, download_name=filename)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/videos/<filename>', methods=['DELETE'])
def delete_video(filename):
    """Delete a specific video file"""
    try:
        video_path = VIDEOS_DIR / filename
        
        if not video_path.exists():
            return jsonify({"error": "Video not found"}), 404
        
        video_path.unlink()
        return jsonify({"message": "Video deleted successfully"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Global camera thread
camera_thread = None


def main():
    """Main entry point"""
    global camera_thread
    
    print("\n" + "="*70)
    print("  Optimized Streaming Backend - Hand-Pet Interaction Detector")
    print("  WebSocket + Binary Transmission + Adaptive FPS")
    print("="*70)
    
    load_model()
    
    writer_thread = threading.Thread(target=video_writer_thread, daemon=True)
    writer_thread.start()
    
    camera_thread = CameraThread()
    camera_thread.start()
    
    time.sleep(2)
    
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print("\n📱 iOS App Configuration:")
    print(f"   Server URL: http://{local_ip}:5001")
    print(f"   WebSocket Stream: ws://{local_ip}:5001/ws/stream  (⚡ LOW LATENCY)")
    print(f"   MJPEG Stream: http://{local_ip}:5001/stream/mjpeg")
    print(f"\n📊 Network Optimization:")
    print(f"   Protocol: WebSocket (5-20ms latency)")
    print(f"   Transmission: Binary JPEG (-33% overhead)")
    print(f"   Initial FPS: {INITIAL_FPS} (adaptive 10-25 FPS)")
    print(f"   Frame Skip: Enabled when queue backs up")
    print(f"\n📂 Paths:")
    print(f"   Model: {MODEL_PATH}")
    print(f"   Videos: {VIDEOS_DIR.absolute()}")
    print(f"\n🎯 Settings:")
    print(f"   Confidence: {CONFIDENCE_THRESHOLD}")
    print(f"   Cooldown: {COOLDOWN_SECONDS}s")
    print(f"   Camera: ESP32-S3")
    print("\n🚀 Starting optimized server...")
    print("="*70 + "\n")
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        if camera_thread:
            camera_thread.stop()
        stop_recording()
        print("\n👋 Server stopped")


if __name__ == "__main__":
    main()
