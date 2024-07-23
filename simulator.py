import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import UNet  # Assuming the UNet class is in a file named model.py
import pyautogui
import time
from mss import mss
from datetime import datetime

# Debugging variables
steering_history = []
key_press_count = {"left": 0, "right": 0}

def capture_screen():
    sct = mss()
    monitor = {"top": 345, "left": 500, "width": 440, "height": 210}
    screenshot = np.array(sct.grab(monitor))
    return cv2.cvtColor(screenshot, cv2.COLOR_RGBA2RGB)

def process_frame(frame, model, transform):
    pil_image = Image.fromarray(frame)
    input_tensor = transform(pil_image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    mask = output.squeeze().cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    inverted_mask = 1 - mask
    
    green_overlay = np.zeros_like(frame)
    green_overlay[:, :, 1] = 255
    green_mask = (inverted_mask[:, :, np.newaxis] * green_overlay).astype(np.uint8)
    
    alpha = 0.3
    result = cv2.addWeighted(frame, 1, green_mask, alpha, 0)
    
    return result, inverted_mask

def calculate_steering(mask, image_center):
    height, width = mask.shape
    lower_half = mask[height//2:, :]
    lane_pixels = np.where(lower_half > 0.5)
    
    if len(lane_pixels[1]) == 0:
        return 0
    
    lane_center = np.mean(lane_pixels[1])
    steering = lane_center - image_center
    return steering

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0
    
    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

def steer_vehicle(steering, pid_controller, prev_steering, dt):
    global key_press_count
    steering_threshold = 10
    center_threshold = 5
    
    # Update the steering value using PID controller
    steering = pid_controller.update(steering, dt)
    
    # Combine previous steering with the new steering for smoothing
    steering = 0.7 * steering + 0.3 * prev_steering
    
    if abs(steering) < center_threshold:
        return steering, "CENTER"
    
    # Determine key press duration based on the steering value
    key_hold_duration = min(0.1, max(0.01, abs(steering) / 100))
    
    if steering < -steering_threshold:
        pyautogui.keyDown('left')
        time.sleep(key_hold_duration)
        pyautogui.keyUp('left')
        key_press_count["left"] += 1
        return steering, "LEFT"
    elif steering > steering_threshold:
        pyautogui.keyDown('right')
        time.sleep(key_hold_duration)
        pyautogui.keyUp('right')
        key_press_count["right"] += 1
        return steering, "RIGHT"
    else:
        return steering, "CENTER"

def draw_debug_info(image, steering, mask, direction):
    global steering_history
    steering_history.append(steering)
    if len(steering_history) > 50:
        steering_history.pop(0)
    
    debug_image = image.copy()
    
    mask_overlay = np.zeros_like(debug_image)
    mask_overlay[:, :, 1] = (mask * 255).astype(np.uint8)
    debug_image = cv2.addWeighted(debug_image, 1, mask_overlay, 0.5, 0)
    
    cv2.putText(debug_image, f"Steering: {steering:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(debug_image, f"Direction: {direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    graph_width = 200
    graph_height = 100
    graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)
    for i, steer in enumerate(steering_history):
        y = int(graph_height / 2 - steer)
        cv2.circle(graph, (i * 4, y), 1, (0, 255, 0), -1)
    
    debug_image[50:50+graph_height, 10:10+graph_width] = cv2.addWeighted(
        debug_image[50:50+graph_height, 10:10+graph_width],
        0.5,
        graph,
        0.5,
        0
    )
    
    cv2.putText(debug_image, f"Left: {key_press_count['left']}", (10, debug_image.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(debug_image, f"Right: {key_press_count['right']}", (10, debug_image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return debug_image

def main():
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load('new-unet/unet_lane_detection_epoch_5.pth', map_location=torch.device('cpu'))['model_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    pid_controller = PIDController(Kp=0.1, Ki=0.01, Kd=0.05)
    prev_steering = 0

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'lane_detection_run_{timestamp}.avi'
    out = None  # We'll initialize this later

    print("Starting in 3 seconds. Switch to the game window!")
    time.sleep(3)

    try:
        while True:
            start_time = time.time()
            screen = capture_screen()
            
            result_frame, inverted_mask = process_frame(screen, model, transform)
            
            image_center = screen.shape[1] / 2
            steering = calculate_steering(inverted_mask, image_center)
            dt = time.time() - start_time
            steering, direction = steer_vehicle(steering, pid_controller, prev_steering, dt)
            prev_steering = steering
            
            debug_image = draw_debug_info(result_frame, steering, inverted_mask, direction)
            
            # Initialize video writer with the correct frame size
            if out is None:
                frame_height, frame_width = debug_image.shape[:2]
                out = cv2.VideoWriter(output_filename, fourcc, 20.0, (frame_width, frame_height))
            
            # Write frame to video
            out.write(debug_image)
            
            cv2.imshow('Lane Detection Debug', debug_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        print(f"Video saved as {output_filename}")

if __name__ == '__main__':
    main()
