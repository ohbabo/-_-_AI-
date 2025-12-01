import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

# Class mapping based on user info
CLASS_MAP = {
    "FL": 0, # Fire
    "SM": 1  # Smoke
}

def convert_bbox(size, box):
    """
    Convert AI Hub bbox [xmin, ymin, width, height] to YOLO [x_center, y_center, width, height] normalized.
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    
    xmin = box[0]
    ymin = box[1]
    w = box[2]
    h = box[3]
    
    x_center = xmin + w / 2.0
    y_center = ymin + h / 2.0
    
    x_center = x_center * dw
    w = w * dw
    y_center = y_center * dh
    h = h * dh
    
    return (x_center, y_center, w, h)

def convert_labels(image_dir, label_source_root, output_dir):
    """
    Walks through the sampled image directory, finds corresponding JSONs in label_source_root,
    and generates YOLO .txt labels.
    
    Args:
        image_dir (str): Path to sampled images (e.g., ai_project/data/sampled/images)
        label_source_root (str): Path to the root of ORIGINAL label data (e.g., /mnt/d/AIHub/Labels)
        output_dir (str): Path to save YOLO .txt labels (e.g., ai_project/data/sampled/labels)
    """
    image_path_root = Path(image_dir)
    label_source_path = Path(label_source_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning images in {image_path_root}...")
    image_files = list(image_path_root.rglob('*'))
    image_files = [f for f in image_files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Found {len(image_files)} images. Starting conversion...")
    
    converted_count = 0
    missing_count = 0
    
    for img_path in tqdm(image_files):
        # Determine relative path to find the corresponding JSON
        # Example: 
        # Image: sampled/images/Part1/Scene01/frame_001.jpg
        # Label: source_labels/Part1/Scene01/frame_001.json
        
        rel_path = img_path.relative_to(image_path_root)
        json_rel_path = rel_path.with_suffix('.json')
        
        # Construct potential JSON path
        # Note: AI Hub sometimes has slightly different folder structures for labels.
        # We assume strict parallelism here.
        json_file = label_source_path / json_rel_path
        
        if not json_file.exists():
            # Try searching? No, too slow. Just log warning.
            # Sometimes labels are in a flat folder?
            missing_count += 1
            continue
            
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Prepare YOLO label content
            yolo_lines = []
            
            # AI Hub structure varies. Based on user image:
            # Root object has "image" info and "annotations" array?
            # User image shows: "4. 어노테이션" -> "annotations" array.
            # "4-5. 바운딩박스" -> "bbox" array inside annotation.
            # "2-1. 클래스" -> "class" string.
            
            # Let's try to adapt to the structure shown in the user's uploaded image
            # The example JSON snippet in the image shows a flat structure?
            # Wait, the example at the bottom of the image shows:
            # { "video":..., "class": "FL", ... "scene": ... }
            # This looks like metadata for a VIDEO or a single frame?
            # If it's a JSON for a single image, it might have 'annotations' key.
            
            # Common AI Hub Image format:
            # { "image": {...}, "annotations": [ { "class": "FL", "bbox": [...] } ] }
            
            img_width = data.get('image', {}).get('width') or data.get('width') or 1920
            img_height = data.get('image', {}).get('height') or data.get('height') or 1080
            
            annotations = data.get('annotations', [])
            
            # If annotations is empty, check if the root object itself is the annotation (unlikely for detection)
            # But the user image example shows "class": "FL" at root. This might be classification label?
            # If it's detection data, there MUST be coordinates.
            
            if not annotations:
                # Check for root level bbox?
                pass
                
            for ann in annotations:
                cls_str = ann.get('class') or ann.get('category_name') # "FL" or "SM"
                bbox = ann.get('bbox') # [xmin, ymin, w, h]
                
                if cls_str in CLASS_MAP and bbox:
                    cls_id = CLASS_MAP[cls_str]
                    yolo_box = convert_bbox((img_width, img_height), bbox)
                    yolo_lines.append(f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
            
            # Save .txt file
            # Maintain relative structure in output_dir
            out_file = output_path / rel_path.with_suffix('.txt')
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            if yolo_lines:
                with open(out_file, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                converted_count += 1
            else:
                # Create empty file for negative samples (no fire)
                with open(out_file, 'w') as f:
                    pass
                converted_count += 1
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            
    print(f"Conversion complete. Converted: {converted_count}, Missing JSON: {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="Path to sampled images root")
    parser.add_argument("--labels", required=True, help="Path to original labels root")
    parser.add_argument("--output", required=True, help="Path to save YOLO labels")
    args = parser.parse_args()
    
    convert_labels(args.images, args.labels, args.output)
