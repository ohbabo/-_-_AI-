import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

class FireDataset(Dataset):
    def __init__(self, image_paths, label_paths, processor):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.processor = processor
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        # Determine label from YOLO txt file
        # 0: Fire, 1: Smoke -> We map both to 1 (Fire/Danger) for binary classification?
        # Or 0: Normal, 1: Fire, 2: Smoke?
        # Let's do Binary: 0=Normal, 1=Fire/Smoke
        
        label = 0 # Default Normal
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    cls_id = int(line.split()[0])
                    if cls_id in [0, 1]: # Fire or Smoke
                        label = 1
                        break
        
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        return {"pixel_values": pixel_values, "labels": torch.tensor(label)}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def train_vit(dataset_root, epochs=3, batch_size=8, model_name='google/vit-base-patch16-224'):
    root = Path(dataset_root)
    images_dir = root / 'images'
    labels_dir = root / 'labels'
    
    if not images_dir.exists():
        print("Error: images directory not found.")
        return

    # Gather all images
    all_images = list(images_dir.rglob('*.jpg')) + list(images_dir.rglob('*.png'))
    
    # Create corresponding label paths
    # Assuming flat or relative structure matches. 
    # If sample_dataset.py flattened it, we need to be careful.
    # But train_yolo.py might have reorganized it into train/val.
    # Let's check if 'train' folder exists.
    
    if (root / 'train').exists():
        print("Detected train/val split. Loading from there...")
        train_images = list((root / 'train' / 'images').rglob('*'))
        val_images = list((root / 'val' / 'images').rglob('*'))
        
        # Helper to find labels
        def get_labels(img_list, split_name):
            lbls = []
            for img in img_list:
                rel = img.relative_to(root / split_name / 'images')
                lbl = root / split_name / 'labels' / rel.with_suffix('.txt')
                lbls.append(lbl)
            return lbls
            
        train_labels = get_labels(train_images, 'train')
        val_labels = get_labels(val_images, 'val')
        
    else:
        print("No split detected. Using random split on 'images' folder...")
        # Fallback to simple split
        train_imgs, val_imgs = train_test_split(all_images, test_size=0.2, random_state=42)
        
        def get_labels_flat(img_list):
            lbls = []
            for img in img_list:
                # Try to find label in labels_dir with same relative path from images_dir
                rel = img.relative_to(images_dir)
                lbl = labels_dir / rel.with_suffix('.txt')
                lbls.append(lbl)
            return lbls

        train_images = train_imgs
        val_images = val_imgs
        train_labels = get_labels_flat(train_images)
        val_labels = get_labels_flat(val_images)

    print(f"Train size: {len(train_images)}, Val size: {len(val_images)}")
    
    # Processor
    processor = ViTImageProcessor.from_pretrained(model_name)
    
    train_dataset = FireDataset(train_images, train_labels, processor)
    val_dataset = FireDataset(val_images, val_labels, processor)
    
    # Model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: 'Normal', 1: 'Fire'},
        label2id={'Normal': 0, 'Fire': 1},
        ignore_mismatched_sizes=True
    )
    
    # Training Args
    training_args = TrainingArguments(
        output_dir='./vit_results',
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=None, # Default is fine
    )
    
    print("Starting ViT training...")
    trainer.train()
    print("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to dataset root")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    
    args = parser.parse_args()
    
    train_vit(args.data, args.epochs, args.batch)
