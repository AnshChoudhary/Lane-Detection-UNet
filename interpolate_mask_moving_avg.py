import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from model import UNet  # Assuming the UNet class is in a file named model.py

def moving_average_2d(data, window_size):
    ret = np.cumsum(data, axis=0, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def process_frame(frame, model, transform):
    # Convert frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply transformations
    input_tensor = transform(pil_image).unsqueeze(0)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert output to numpy array
    mask = output.squeeze().cpu().numpy()
    
    # Resize mask to match frame size
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Invert the mask
    inverted_mask = 1 - mask
    
    return inverted_mask

def overlay_mask_on_frame(frame, mask):
    # Create green overlay
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255  # Set green channel to 255
    
    # Apply inverted mask to green overlay
    mask_3d = np.stack([mask, mask, mask], axis=2)
    green_mask = (mask_3d * green_overlay).astype(np.uint8)
    
    # Blend original frame with green mask
    alpha = 0.9  # Adjust transparency
    result = cv2.addWeighted(frame, 1, green_mask, alpha, 0)
    
    return result

def process_video(input_path, output_path, model, transform, window_size=31):
    print("Opening video file...")
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height} at {fps} fps, {total_frames} frames")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Reading and processing frames...")
    frames = []
    masks = []
    for _ in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        mask = process_frame(frame, model, transform)
        masks.append(mask)
    
    print(f"Loaded and processed {len(frames)} frames")
    
    print("Applying moving average on masks...")
    masks = np.array(masks)
    smoothed_masks = moving_average_2d(masks, window_size)
    
    # Check for discrepancy in frame and mask count
    if len(smoothed_masks) != len(frames):
        print(f"Discrepancy detected: {len(smoothed_masks)} smoothed masks for {len(frames)} frames")
        min_length = min(len(smoothed_masks), len(frames))
        frames = frames[:min_length]
        smoothed_masks = smoothed_masks[:min_length]
    
    print("Overlaying masks on frames and writing to output video...")
    for frame, mask in tqdm(zip(frames, smoothed_masks), desc="Writing frames", total=len(frames)):
        result_frame = overlay_mask_on_frame(frame, mask)
        out.write(result_frame.astype(np.uint8))
    
    cap.release()
    out.release()
    print("Processing complete!")

def main():
    # Load the model
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('new-unet/unet_lane_detection_epoch_5.pth', map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Process the video
    process_video('/Users/anshchoudhary/Downloads/u-net-torch/night_video.mp4',
                  '/Users/anshchoudhary/Downloads/u-net-torch/new-unet/output_interpolated_night.mp4',
                  model, transform, window_size=31)

if __name__ == '__main__':
    main()

