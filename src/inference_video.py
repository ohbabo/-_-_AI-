import cv2
from ultralytics import YOLO
import argparse
from pathlib import Path

def process_video(video_path, model_path, output_path=None):
    """
    Runs YOLO inference on a video file.
    Demonstrates how 'Image-based' training is applied to 'Video' data.
    """
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing video: {video_path}...")
    
    frame_count = 0
    fire_detected_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Inference
        results = model(frame, verbose=False)
        
        # Visualize
        annotated_frame = results[0].plot()
        
        # Temporal Logic Example:
        # Check if fire is detected in this frame
        # results[0].boxes.cls contains class IDs. (Assuming 0=Fire)
        detected_classes = results[0].boxes.cls.cpu().numpy()
        if 0 in detected_classes: # If Fire is present
            fire_detected_frames += 1
        else:
            fire_detected_frames = max(0, fire_detected_frames - 1) # Decay
            
        # Simple Alarm Logic: If Fire seen in last 10 consecutive frames
        if fire_detected_frames > 10:
            cv2.putText(annotated_frame, "WARNING: FIRE CONFIRMED!", (50, 50), 
                        cv2.VideoWriter_fourcc(*'mp4v'), 1, (0, 0, 255), 2)
        
        if out:
            out.write(annotated_frame)
            
        # Optional: Display (might not work well in WSL without GUI setup)
        # cv2.imshow('Fire Detection', annotated_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
            
    cap.release()
    if out:
        out.release()
    print("Processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", required=True, help="Path to trained .pt model")
    parser.add_argument("--output", default="output.mp4", help="Path to save output video")
    
    args = parser.parse_args()
    
    process_video(args.video, args.model, args.output)
