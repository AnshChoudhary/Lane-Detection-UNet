import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet  # Assuming the UNet class is in a file named model.py

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
    
    # Create green overlay
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255  # Set green channel to 255
    
    # Apply inverted mask to green overlay
    mask_3d = np.stack([inverted_mask, inverted_mask, inverted_mask], axis=2)
    green_mask = (mask_3d * green_overlay).astype(np.uint8)
    
    # Blend original frame with green mask
    alpha = 0.9  # Adjust transparency
    result = cv2.addWeighted(frame, 1, green_mask, alpha, 0)
    
    return result

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

    # Open input video
    input_video = cv2.VideoCapture('/Users/anshchoudhary/Downloads/u-net-torch/input3.mp4')
    
    # Get video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/Users/anshchoudhary/Downloads/u-net-torch/new-unet/output_input3.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = input_video.read()
        if not ret:
            break
        
        # Process frame
        result_frame = process_frame(frame, model, transform)
        
        # Write frame to output video
        output_video.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('Lane Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()