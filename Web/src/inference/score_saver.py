import os
import torch
import torch.nn.functional as F
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
import sys
from tqdm import tqdm


from .model import generate_model
from .opts import parse_opts
from .learner import Learner


class ToTensor(object):
    """Convert a PIL.Image or numpy.ndarray to tensor."""
    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float().div(self.norm_value)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.as_tensor(np.array(pic))
            
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
            
        img = img.view(pic.size[1], pic.size[0], nchannel)
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        
        if img.dtype == torch.uint8:
            return img.float().div(self.norm_value)
        else:
            return img

def extract_frames_ffmpeg(video_path, output_dir):
    """Extract frames from video using FFmpeg"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract frames using FFmpeg
    cmd = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'scale=240:240',  # Resize to 240x240
        '-q:v', '1',             # High quality
        '-vsync', 'vfr',  # Giữ nguyên tốc độ khung hình
        f'{output_dir}/frame_%05d.jpg'
    ]
    
    subprocess.run(cmd, check=True)
    
    # Return sorted list of frame paths
    frames = sorted(Path(output_dir).glob('frame_*.jpg'))
    return frames


def run_vad_model(video_path: str, total_frames, scores_dir):
    print(video_path)
    """Run Video Anomaly Detection model on video and save anomaly scores"""
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model weights
    pretrain_path = Path(__file__).resolve().parent.parent.parent / "weights" / "RGB_Kinetics_16f.pth"
    
    # Set up model parameters
    opts = parse_opts()
    opts.pretrain_path = str(pretrain_path)
    opts.arch = "resnext-101"
    opts.sample_size = 240
    opts.sample_duration = 15
    opts.output_layers = ["avgpool"]

    # Initialize classifier
    classifier = Learner().to(device)
    classifier.eval()
    
    # Initialize feature extractor
    model = generate_model(opts)
    model = model.to(device)
    
    # Create temporary directory for extracted frames
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting frames from {video_path} to {temp_dir}...")
        frame_paths = extract_frames_ffmpeg(video_path, temp_dir)
        
        if not frame_paths:
            raise ValueError(f"Failed to extract frames from {video_path}")
        
        # Create buffer for 16 frames (sliding window approach)
        window_size = 16
        inputs = torch.zeros(1, 3, window_size, 240, 240)
        
        # Initialize anomaly scores array with default scores
        anomaly_scores = np.full(total_frames, 0.2)
        
        # Process frames
        num_frames = len(frame_paths)
        print(f"Processing {num_frames} extracted frames")
        
        # Mapping extracted frames to original video frames
        frame_ratio = total_frames / num_frames
        
        for idx, frame_path in enumerate(frame_paths):
            # Calculate corresponding original frame index
            orig_frame_idx = min(int(idx * frame_ratio), total_frames - 1)
            
            # Read and process frame
            pil_img = Image.open(frame_path)
            
            if idx < window_size:
                # Fill buffer with initial frames
                inputs[:, :, idx, :, :] = ToTensor(1)(pil_img)
            else:
                # Shift buffer left by one frame
                inputs[:, :, :window_size-1, :, :] = inputs[:, :, 1:, :, :]
                # Add new frame to buffer
                inputs[:, :, window_size-1, :, :] = ToTensor(1)(pil_img)
                
                # Move to device
                inputs_device = inputs.to(device)
                
                # Forward pass through model
                with torch.no_grad():
                    _, feature = model(inputs_device)
                    feature = F.normalize(feature, p=2, dim=1)
                    out = classifier(feature)
                    
                    # Store prediction
                    anomaly_scores[orig_frame_idx] = out.item()
            
            if idx % 10 == 0:
                print(f"Processed {idx}/{num_frames} frames")
    
    # Save scores to file
    scores_file = Path(scores_dir) / f"{Path(video_path).stem}_scores.npy"
    np.save(scores_file, anomaly_scores)
    print(f"Saved anomaly scores to {scores_file}")
    
    return str(scores_file)