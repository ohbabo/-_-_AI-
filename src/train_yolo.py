from ultralytics import YOLO
import os
import shutil
import yaml
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def setup_yolo_data(dataset_root, split_ratio=0.8):
    """
    Organizes images and labels into train/val structure for YOLO.
    dataset_root/
      images/ -> all images
      labels/ -> all labels
    
    Transforms to:
    dataset_root/
      train/
        images/
        labels/
      val/
        images/
        labels/
    """
    root = Path(dataset_root)
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print("Error: 'images' or 'labels' directory missing in dataset root.")
        return False
        
    # Check if already split
    if (root / 'train').exists():
        print("Dataset seems to be already split. Skipping organization.")
        return True

    print("Organizing dataset into Train/Val splits...")
    
    # Get all image files
    image_files = list(images_dir.rglob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    train_files, val_files = train_test_split(image_files, train_size=split_ratio, random_state=42)
    
    for split_name, files in [('train', train_files), ('val', val_files)]:
        split_dir = root / split_name
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        for img_path in files:
            # Move image
            # We flatten structure here for YOLO simplicity or keep it? 
            # YOLO doesn't care about structure inside train/images as long as labels match.
            # But labels must be in .../labels/image_name.txt
            
            # Find corresponding label
            # We need to find the relative path of the image from 'images_dir' to find the label in 'labels_dir'
            rel_path = img_path.relative_to(images_dir)
            label_path = labels_dir / rel_path.with_suffix('.txt')
            
            if not label_path.exists():
                # print(f"Warning: Label not found for {img_path.name}")
                continue
                
            # Copy/Move to split dir
            # We flatten here to avoid deep nesting issues in YOLO
            shutil.copy2(img_path, split_dir / 'images' / img_path.name)
            shutil.copy2(label_path, split_dir / 'labels' / label_path.name)
            
    print(f"Split complete. Train: {len(train_files)}, Val: {len(val_files)}")
    return True

def create_data_yaml(dataset_root):
    root = Path(dataset_root).absolute()
    
    data = {
        'path': str(root),
        'train': 'train/images',
        'val': 'val/images',
        'names': {
            0: 'Fire',
            1: 'Smoke'
        }
    }
    
    yaml_path = root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    return yaml_path

def train_yolo(dataset_root, epochs=10, batch_size=8, model_size='n'):
    """
    Train YOLOv8 model.
    """
    if not setup_yolo_data(dataset_root):
        return

    yaml_path = create_data_yaml(dataset_root)
    
    print(f"Starting training with YOLOv8{model_size}...")
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=0, # Use GPU 0
        workers=4,
        project='fire_detection_yolo',
        name=f'yolov8{model_size}_run'
    )
    
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to sampled dataset root (containing images/ and labels/)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8, help="Batch size (keep low for GTX 1050)")
    parser.add_argument("--model", default='n', choices=['n', 's', 'm', 'l', 'x'], help="YOLO model size (n=nano, s=small)")
    
    args = parser.parse_args()
    
    train_yolo(args.data, args.epochs, args.batch, args.model)
