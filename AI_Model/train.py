"""
Train YOLOv8 model on FYP.v9i.yolov8 dataset
Output to AI_Model_v3
"""

import os
import sys
import json
import signal
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import yaml

class TrainingManager:
    def __init__(self, project_dir, data_yaml_path):
        self.project_dir = Path(project_dir)
        self.data_yaml = data_yaml_path
        self.checkpoint_file = self.project_dir / "training_checkpoint.json"
        self.model = None
        self.training_interrupted = False
        
        self.project_dir.mkdir(parents=True, exist_ok=True)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n⚠️  Training interrupted by user...")
        self.training_interrupted = True
        sys.exit(0)
    
    def save_checkpoint(self, last_epoch, total_epochs):
        """Save training checkpoint"""
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "last_epoch": last_epoch,
            "total_epochs": total_epochs,
            "status": "paused"
        }
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"\n✓ Checkpoint saved at epoch {last_epoch}")
    
    def load_checkpoint(self):
        """Load previous training checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            return checkpoint
        return None
    
    def train(self, epochs=100, batch_size=16, imgsz=640, device=0, resume=False):
        """Train YOLOv8s model"""
        
        if not Path(self.data_yaml).exists():
            print(f"✗ Error: data.yaml not found at {self.data_yaml}")
            return None
        
        if resume:
            checkpoint = self.load_checkpoint()
            if checkpoint:
                print(f"\n✓ Found checkpoint from {checkpoint['timestamp']}")
                runs_dir = self.project_dir / "runs" / "detect" / "train" / "weights"
                best_pt = runs_dir / "best.pt"
                if best_pt.exists():
                    self.model = YOLO(str(best_pt))
                    print(f"✓ Loaded model from {best_pt}")
                else:
                    self.model = YOLO("yolov8s.pt")
            else:
                self.model = YOLO("yolov8s.pt")
        else:
            self.model = YOLO("yolov8s.pt")
        
        print("\n" + "="*60)
        print("Starting YOLOv8s Training")
        print("="*60)
        print(f"Dataset: {self.data_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {imgsz}")
        print(f"Device: GPU {device}")
        print("="*60 + "\n")
        
        try:
            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                patience=20,
                project=str(self.project_dir / "runs"),
                name="detect",
                exist_ok=True,
                verbose=True,
                amp=True,
                workers=0,  # Windows fix: use 0 workers
                # Data augmentation settings
                flipud=0.1,
                degrees=15,
                shear=10,
                hsv_h=0.05,
                hsv_s=0.2,
                hsv_v=0.2,
                mosaic=0.5,
                scale=0.2,
                translate=0.1,
                erasing=0.1,
            )
            
            print("\n" + "="*60)
            print("✓ Training Completed Successfully!")
            print("="*60)
            
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            
            return results
        
        except Exception as e:
            print(f"\n✗ Training error: {e}")
            self.save_checkpoint(0, epochs)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLOv8s for cat and human detection")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()
    PROJECT_DIR = Path(__file__).parent.absolute()
    data_yaml = PROJECT_DIR / "data.yaml"
    
    # Create data.yaml if not exists
    if not data_yaml.exists():
        data_config = {
            'path': str((PROJECT_ROOT / "FYP.v9i.yolov8").absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'nc': 2,
            'names': ['cat', 'human']
        }
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)
        print(f"✅ Created data.yaml: {data_yaml}")
    
    manager = TrainingManager(str(PROJECT_DIR), str(data_yaml))
    manager.train(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume
    )

if __name__ == "__main__":
    main()