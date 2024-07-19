import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet  # Ensure this is in the same directory
from ultralytics import YOLO
import time
import os
import tempfile
import subprocess

# Initialize session state variables
if 'previous_inverted_mask' not in st.session_state:
    st.session_state.previous_inverted_mask = None
if 'previous_center' not in st.session_state:
    st.session_state.previous_center = None
if 'previous_time' not in st.session_state:
    st.session_state.previous_time = None
if 'speeds' not in st.session_state:
    st.session_state.speeds = []
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'total_distance' not in st.session_state:
    st.session_state.total_distance = 0

# Load models
@st.cache_resource
def load_models():
    lane_model = UNet(in_channels=3, out_channels=1)
    lane_model.load_state_dict(torch.load('quantized_unet_lane_detection.pth', map_location=torch.device('cpu')))
    lane_model.eval()
    
    yolo_model = YOLO('yolov8n.pt')
    
    return lane_model, yolo_model

lane_model, yolo_model = load_models()

def moving_average_2d(data, window_size):
    ret = np.cumsum(data, axis=0, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    return ret[window_size - 1:] / window_size

def get_traffic_light_color(frame, x1, y1, x2, y2):
    light = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    green_pixels = cv2.countNonZero(mask_green)
    
    max_pixels = max(red_pixels, yellow_pixels, green_pixels)
    if max_pixels == red_pixels:
        return "Red"
    elif max_pixels == yellow_pixels:
        return "Yellow"
    elif max_pixels == green_pixels:
        return "Green"
    else:
        return "Unknown"

def calculate_lane_center(inverted_mask):
    height, width = inverted_mask.shape
    bottom_quarter = inverted_mask[3*height//4:, :]
    
    left_boundary = np.argmax(bottom_quarter < 0.5, axis=1)
    right_boundary = width - np.argmax(np.fliplr(bottom_quarter) < 0.5, axis=1)
    
    valid_left = left_boundary[left_boundary > 0]
    valid_right = right_boundary[right_boundary < width]
    
    if len(valid_left) > 0 and len(valid_right) > 0:
        lane_center = (np.mean(valid_left) + np.mean(valid_right)) / 2
    else:
        lane_center = width / 2
    
    return lane_center

def calculate_speed(current_center, previous_center, time_diff, pixel_to_meter):
    if previous_center is None or time_diff == 0:
        return 0
    
    distance = abs(current_center - previous_center) * pixel_to_meter
    speed = (distance / time_diff) * 3.6
    
    return speed

def dynamic_window_size_adjustment(mask, base_window_size, min_window_size, max_window_size):
    detected_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    proportion = detected_pixels / total_pixels
    
    # Larger window size when fewer lane pixels are detected
    window_size = int(max_window_size * (1 - proportion) + min_window_size * proportion)
    
    return max(min_window_size, min(max_window_size, window_size))

def process_frame(frame, lane_model, yolo_model, transform, yolo_conf, detection_alpha, interpolation_factor, pixel_to_meter, base_window_size, min_window_size, max_window_size):
    current_time = time.time()
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        output = lane_model(input_tensor)
    
    mask = output.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    inverted_mask = 1 - mask
    
    if st.session_state.previous_inverted_mask is not None:
        inverted_mask = cv2.addWeighted(st.session_state.previous_inverted_mask, 1 - interpolation_factor, inverted_mask, interpolation_factor, 0)
    
    st.session_state.previous_inverted_mask = inverted_mask.copy()
    
    window_size = dynamic_window_size_adjustment(mask, base_window_size, min_window_size, max_window_size)
    if window_size > 1:
        inverted_mask = moving_average_2d(inverted_mask, window_size)
    
    lane_center = calculate_lane_center(inverted_mask)
    frame_center = frame.shape[1] // 2
    distance_from_center = (lane_center - frame_center) * pixel_to_meter
    
    speed = 0
    if st.session_state.previous_time is not None and st.session_state.previous_center is not None:
        time_diff = current_time - st.session_state.previous_time
        speed = calculate_speed(lane_center, st.session_state.previous_center, time_diff, pixel_to_meter)
        st.session_state.speeds.append(speed)
        st.session_state.total_distance += abs(lane_center - st.session_state.previous_center) * pixel_to_meter
    
    avg_speed = np.mean(st.session_state.speeds) if st.session_state.speeds else 0
    
    st.session_state.previous_center = lane_center
    st.session_state.previous_time = current_time
    st.session_state.frame_count += 1
    
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255
    mask_3d = np.stack([inverted_mask] * 3, axis=2)
    mask_3d = cv2.resize(mask_3d, (frame.shape[1], frame.shape[0]))  # Ensure the mask is resized to match the frame
    green_mask = (mask_3d * green_overlay).astype(np.uint8)
    result = cv2.addWeighted(frame, 1, green_mask, 0.9, 0)  # Fixed blending alpha
    
    yolo_results = yolo_model(frame, conf=yolo_conf)
    yolo_overlay = np.zeros_like(frame, dtype=np.uint8)
    
    for r in yolo_results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            if conf > yolo_conf:
                class_name = yolo_model.names[cls]
                color = (0, 255, 0)
                
                if class_name == "traffic light":
                    light_color = get_traffic_light_color(frame, x1, y1, x2, y2)
                    label = f'Traffic Light ({light_color}) {conf:.2f}'
                    if light_color == "Red":
                        color = (0, 0, 255)
                    elif light_color == "Yellow":
                        color = (255, 255, 0)
                    elif light_color == "Green":
                        color = (0, 255, 0)
                else:
                    label = f'{class_name} {conf:.2f}'
                
                cv2.rectangle(yolo_overlay, (x1, y1), (x2, y2), color, 2)
                cv2.putText(yolo_overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    cv2.addWeighted(result, 1, yolo_overlay, detection_alpha, 0, result)
    
    return result

def convert_video_to_web_friendly(input_path, output_path):
    try:
        command = [
            'ffmpeg',
            '-i', input_path,
            '-vcodec', 'libx264',
            '-crf', '23',
            '-preset', 'medium',
            '-acodec', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            output_path
        ]
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during video conversion: {e}")
        return False
    
def main():
    st.title("Lane Detection and Object Recognition App")

    st.sidebar.title("Controls")
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    use_webcam = st.sidebar.checkbox("Use Webcam")

    if use_webcam or uploaded_file is not None:
        yolo_conf = st.sidebar.slider('YOLO Confidence Threshold', 0.0, 1.0, 0.5, 0.05)
        detection_alpha = st.sidebar.slider('Detection Alpha', 0.0, 1.0, 0.5, 0.05)
        interpolation_factor = st.sidebar.slider('Interpolation Factor', 0.0, 1.0, 0.5, 0.05)
        pixel_to_meter = st.sidebar.slider('Pixel to Meter Ratio', 0.0, 1.0, 0.1, 0.01)
        base_window_size = st.sidebar.slider('Base Window Size', 1, 100, 30, 1)
        min_window_size = st.sidebar.slider('Min Window Size', 1, 50, 10, 1)
        max_window_size = st.sidebar.slider('Max Window Size', 1, 200, 50, 1)

        if st.sidebar.button("Process Video" if uploaded_file else "Start Webcam"):
            st.session_state.previous_inverted_mask = None
            st.session_state.previous_center = None
            st.session_state.previous_time = None
            st.session_state.speeds = []
            st.session_state.frame_count = 0
            st.session_state.total_distance = 0

            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmpfile:
                    tmpfile.write(uploaded_file.getbuffer())
                    temp_file_name = tmpfile.name

                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

                cap = cv2.VideoCapture(temp_file_name)
            else:
                cap = cv2.VideoCapture(0)
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                ])

            if uploaded_file:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                os.makedirs("outputs", exist_ok=True)
                output_file = "outputs/processed_video_raw.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

                progress_bar = st.progress(0)

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = process_frame(frame, lane_model, yolo_model, transform, yolo_conf, detection_alpha, interpolation_factor, pixel_to_meter, base_window_size, min_window_size, max_window_size)
                    out.write(processed_frame)

                    progress = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) / frame_count
                    progress_bar.progress(progress)

                cap.release()
                out.release()
                os.unlink(temp_file_name)

                st.success("Video processing complete!")
                
                web_friendly_output = "outputs/processed_video_web.mp4"
                st.info("Converting video to web-friendly format...")
                if convert_video_to_web_friendly(output_file, web_friendly_output):
                    st.success("Video conversion complete!")
                    video_file = open(web_friendly_output, 'rb')
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                else:
                    st.error("Failed to convert the video. Please check the logs.")
            else:
                stframe = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    processed_frame = process_frame(frame, lane_model, yolo_model, transform, yolo_conf, detection_alpha, interpolation_factor, pixel_to_meter, base_window_size, min_window_size, max_window_size)
                    stframe.image(processed_frame, channels="BGR", use_column_width=True)

                cap.release()

if __name__ == "__main__":
    main()
