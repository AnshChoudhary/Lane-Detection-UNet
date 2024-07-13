import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet  # Assuming the UNet class is in a file named model.py
from ultralytics import YOLO

def get_traffic_light_color(frame, x1, y1, x2, y2):
    # Extract the traffic light region
    light = frame[y1:y2, x1:x2]
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
    
    # Define color ranges
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Create masks for each color
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    # Count pixels for each color
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    
    # Determine the dominant color
    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    if max_pixels == red_pixels:
        return "Red"
    elif max_pixels == yellow_pixels:
        return "Yellow"
    elif max_pixels == green_pixels:
        return "Green"
    else:
        return "Unknown"

previous_inverted_mask = None

def process_frame(frame, lane_model, yolo_model, transform):
    global previous_inverted_mask
    
    # Lane detection processing
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        output = lane_model(input_tensor)
    
    mask = output.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    inverted_mask = 1 - mask
    
    # Apply interpolation on the inverted mask
    if previous_inverted_mask is not None:
        interpolation_factor = 0.7  # Adjust this value between 0 and 1
        inverted_mask = cv2.addWeighted(previous_inverted_mask, 1 - interpolation_factor, inverted_mask, interpolation_factor, 0)
    
    previous_inverted_mask = inverted_mask.copy()
    
    # Apply Gaussian blur to smooth the inverted mask
    inverted_mask = cv2.GaussianBlur(inverted_mask, (15, 15), 0)
    
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255
    
    mask_3d = np.stack([inverted_mask, inverted_mask, inverted_mask], axis=2)
    green_mask = (mask_3d * green_overlay).astype(np.uint8)
    
    alpha = 0.9
    result = cv2.addWeighted(frame, 1, green_mask, alpha, 0)
    
    # YOLO object detection (unchanged)
    yolo_results = yolo_model(frame)
    
    # Create a separate overlay for YOLO detections
    yolo_overlay = np.zeros_like(frame, dtype=np.uint8)
    
    # Draw translucent bounding boxes and labels
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            if conf > 0.5:  # Confidence threshold
                class_name = yolo_model.names[cls]
                color = (0, 255, 0)  # Default color (green)
                
                if class_name == "traffic light":
                    # Detect traffic light color
                    light_color = get_traffic_light_color(frame, x1, y1, x2, y2)
                    label = f'Traffic Light ({light_color}) {conf:.2f}'
                    
                    # Set color based on traffic light state
                    if light_color == "Red":
                        color = (0, 0, 255)  # Red
                    elif light_color == "Yellow":
                        color = (0, 255, 255)  # Yellow
                    elif light_color == "Green":
                        color = (0, 255, 0)  # Green
                else:
                    label = f'{class_name} {conf:.2f}'
                
                # Draw rectangle on the overlay
                cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), color, 2)
                
                # Draw text on the overlay
                cv2.putText(yolo_overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Blend the YOLO overlay with the result
    yolo_alpha = 0.5  # Adjust this value to change the transparency (0.0 - 1.0)
    cv2.addWeighted(result, 1, yolo_overlay, yolo_alpha, 0, result)
    
    return result

def main():
    # Load the lane detection model
    lane_model = UNet(in_channels=3, out_channels=1)
    lane_model.load_state_dict(torch.load('new-unet/unet_lane_detection_epoch_5.pth', map_location=torch.device('cpu'))['model_state_dict'])
    lane_model.eval()

    # Load the YOLO model
    yolo_model = YOLO('yolov8n.pt')  # You can choose a different YOLO model if needed

    # Define transform for lane detection
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Open input video
    input_video = cv2.VideoCapture('/Users/anshchoudhary/Downloads/u-net-torch/seoul_traffic.mp4')
    
    # Get video properties
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('/Users/anshchoudhary/Downloads/u-net-torch/new-unet/output_seoultraffic_yolo.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = input_video.read()
        if not ret:
            break
        
        # Process frame
        result_frame = process_frame(frame, lane_model, yolo_model, transform)
        
        # Write frame to output video
        output_video.write(result_frame)
        
        # Display frame (optional)
        cv2.imshow('Lane Detection and Object Detection', result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()