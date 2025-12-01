import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def sample_dataset(source_dir, target_dir, sample_size=5000, seed=42):
    """
    Samples a subset of images from source_dir and copies them to target_dir.
    Maintains the directory structure or flattens it? 
    For YOLO, it's often easier to have images/ and labels/ folders.
    
    Args:
        source_dir (str): Path to the root of the extracted dataset (e.g., /mnt/d/AIHub_Fire/Training/01.원천데이터)
        target_dir (str): Path to save the sampled dataset.
        sample_size (int): Number of images to sample.
    """
    random.seed(seed)
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directories
    images_dir = target_path / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning for images in {source_path}...")
    # Extensions to look for
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.mp4'] # AI Hub data might be mp4 clips or jpg frames
    
    all_files = []
    for ext in extensions:
        all_files.extend(list(source_path.rglob(ext)))
    
    print(f"Found {len(all_files)} files.")
    
    if len(all_files) == 0:
        print("No files found! Please check the source directory.")
        return

    # Sample files
    if len(all_files) > sample_size:
        sampled_files = random.sample(all_files, sample_size)
    else:
        sampled_files = all_files
        print(f"Dataset smaller than sample size. Using all {len(all_files)} files.")

    print(f"Copying {len(sampled_files)} files to {images_dir}...")
    
    for file_path in tqdm(sampled_files):
        # Calculate relative path from source_dir
        try:
            rel_path = file_path.relative_to(source_path)
        except ValueError:
            # Fallback if file is not relative to source (shouldn't happen with rglob)
            rel_path = file_path.name

        dest_file = images_dir / rel_path
        
        # Create parent directories in target
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(file_path, dest_file)

    print("Sampling complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a subset of the dataset.")
    parser.add_argument("--source", type=str, required=True, help="Path to source images")
    parser.add_argument("--target", type=str, required=True, help="Path to target dataset folder")
    parser.add_argument("--size", type=int, default=5000, help="Number of images to sample")
    
    args = parser.parse_args()
    
    sample_dataset(args.source, args.target, args.size)
