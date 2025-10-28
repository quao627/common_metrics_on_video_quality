import torch
import cv2
import numpy as np
import os
import json
from calculate_fvd import calculate_fvd
# from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
# from calculate_lpips import calculate_lpips

def load_video(video_path, max_frames=121):
    """Load video and return tensor in format [T, C, H, W] normalized to [0, 1]"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0  # Normalize to [0, 1]
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None
    
    # Convert to tensor: [T, H, W, C] -> [T, C, H, W]
    video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2)
    return video_tensor

def load_videos_from_folder(folder_path, max_frames=121):
    """Load all videos from folder and return as batch tensor [B, T, C, H, W]"""
    videos = []
    video_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
    print(video_files)
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        video_tensor = load_video(video_path, max_frames)
        if video_tensor is not None:
            videos.append(video_tensor)
    
    if not videos:
        return None
    
    # Stack to create batch dimension [B, T, C, H, W]
    return torch.stack(videos)

print("Starting...")

# Load videos from data folders
real_videos = load_videos_from_folder('data/real')
print("Finished loading real videos...")
generated_videos = load_videos_from_folder('data/generated')
print("Finished loading generated videos...")

if real_videos is None or generated_videos is None:
    print("Error: Could not load videos from data folders")
    exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Real videos shape: {real_videos.shape}")
print(f"Generated videos shape: {generated_videos.shape}")

# Calculate metrics
result = {}
only_final = True

print("Calculating FVD...")
result['fvd'] = calculate_fvd(real_videos, generated_videos, device, method='styleganv', only_final=only_final)

print("Calculating SSIM...")
result['ssim'] = calculate_ssim(real_videos, generated_videos, only_final=only_final)

# print("Calculating PSNR...")
# result['psnr'] = calculate_psnr(real_videos, generated_videos, only_final=only_final)

# print("Calculating LPIPS...")
# result['lpips'] = calculate_lpips(real_videos, generated_videos, device, only_final=only_final)

print("\nResults:")
print(json.dumps(result, indent=4))
