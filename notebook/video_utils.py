import cv2
import os
from tqdm import tqdm
from pathlib import Path


def create_mp4(output_path: Path, frames_folder, fps=30):
    """
    output_path: folder / video_name
    frames_folder: folder containing images with name XX_{index}
    """
    output_path.parent.mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = sorted(
        [os.path.join(frames_folder, img) for img in os.listdir(frames_folder) if img.endswith(".jpg")],
        key=lambda x: int(Path(x).stem.split("_")[-1])
    )
    
    # Read the first image to get the frame size
    first_frame = cv2.imread(image_files[0])
    frame_size = (first_frame.shape[1], first_frame.shape[0])
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    
    for img_path in image_files:
        frame = cv2.imread(img_path)
        out.write(frame)
        
    out.release()