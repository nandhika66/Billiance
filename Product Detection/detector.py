from yolo_nas_onnx.models import load_net
from yolo_nas_onnx.processing import Preprocessing, Postprocessing
from yolo_nas_onnx.draw import draw_box
from yolo_nas_onnx.utils import Labels
import numpy as np
import cv2
from PIL import Image
import argparse
import os
import time

def detect(net, source, pre_process, post_process, labels, fps=0):
    """Run detection on a single frame"""
    net_input = source.copy()
    input_, prep_meta = pre_process(net_input)
    outputs = net.forward(input_)

    boxes, scores, classes = post_process(outputs, prep_meta)
    selected = cv2.dnn.NMSBoxes(boxes, scores, post_process.score_thres, post_process.iou_thres)

    # Count the number of products detected
    product_count = len(selected)

    # Draw products in red and add count in blue
    for i in selected:
        box = boxes[i, :].astype(np.int32).flatten()
        score = float(scores[i]) * 100
        label, _ = labels(classes[i], use_bgr=True)
        draw_box(source, box, label, score, (0, 0, 255), hide_percentage=True)

    # Add product count text in green color (BGR format)
    cv2.putText(source, f"Total Products: {product_count}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Add FPS counter in green color
    cv2.putText(source, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return source, product_count

use_gpu = False  # Changed to False since GPU is not available
use_opencv_dnn_runtime = True  # Using OpenCV DNN runtime for better CPU performance
model_path = r"E:\Nandhika\Billiance\Product_detection\Product_detection\average_model.onnx"

net = load_net(model_path, use_gpu, use_opencv_dnn_runtime)
net.assert_input_shape([1,3,640,640])
net.warmup()

prep_steps = [
    {"DetLongMaxRescale": None},
    {"BotRightPad": {"pad_value": 114}}
]

iou_thres = 0.5
score_thres = 0.3
labels = ["0"]

_, _, input_height, input_width = net.input_shape  

pre_process = Preprocessing(
    prep_steps, (input_height, input_width)
)

post_process = Postprocessing(
    prep_steps,
    iou_thres,
    score_thres,
)

labels = Labels(labels)

def process_video(video_source):
    """Process video with optimized performance and improved display"""
    # Initialize video capture
    if video_source.isdigit():
        cap = cv2.VideoCapture(int(video_source))  # For webcam
    else:
        cap = cv2.VideoCapture(video_source)  # For video file
    
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_source}")
        return

    # Set buffer size to minimize delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video properties: {frame_width}x{frame_height} @ {fps}fps")
    
    # Determine optimal processing size (smaller for speed)
    max_dimension = 640
    scale_factor = min(max_dimension / frame_width, max_dimension / frame_height)
    if scale_factor < 1.0:
        proc_width = int(frame_width * scale_factor)
        proc_height = int(frame_height * scale_factor)
        print(f"Resizing for processing: {proc_width}x{proc_height}")
    else:
        proc_width, proc_height = frame_width, frame_height
        
    # Frame skipping for faster processing
    frame_skip = 1  # Process every nth frame, adjust as needed
    
    # For display, determine if we need to resize for viewing
    max_display_width = 1280
    display_scale = min(1.0, max_display_width / frame_width)
    display_width = int(frame_width * display_scale)
    display_height = int(frame_height * display_scale)
    need_resize_for_display = (display_width != frame_width)
    
    # Create named window first to set properties
    cv2.namedWindow('Product Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Product Detection', display_width, display_height)

    # Initialize variables for FPS calculation
    prev_time = time.time()
    current_fps = 0
    fps_alpha = 0.1  # Smoothing factor for FPS calculation

    # Initialize variables for product count tracking
    prev_count = 0
    count_change_time = time.time()
    
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
                
            # Skip frames for faster processing if needed
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # Resize frame for faster processing if needed
            if scale_factor < 1.0:
                process_frame = cv2.resize(frame, (proc_width, proc_height))
            else:
                process_frame = frame
            
            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - prev_time
            prev_time = current_time
            
            if elapsed_time > 0:
                current_fps = current_fps * (1 - fps_alpha) + (1.0 / elapsed_time) * fps_alpha

            # Process frame
            processed_frame, current_count = detect(net, process_frame, pre_process, post_process, labels, current_fps)
            
            # If we processed a resized frame, resize the output back for display
            if scale_factor < 1.0 and need_resize_for_display:
                display_frame = cv2.resize(processed_frame, (display_width, display_height))
            elif need_resize_for_display:
                display_frame = cv2.resize(frame, (display_width, display_height))
            else:
                display_frame = processed_frame

            # Check for product count changes
            if current_count != prev_count:
                count_diff = current_count - prev_count
                change_type = "increased" if count_diff > 0 else "decreased"
                print(f"Product count {change_type} by {abs(count_diff)} (Total: {current_count})")
                prev_count = current_count
                count_change_time = current_time

            # Add change notification if recent
            if current_time - count_change_time < 2.0:  # Show notification for 2 seconds
                cv2.putText(display_frame, "Count Updated!", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display the frame
            cv2.imshow('Product Detection', display_frame)
            
            # Ensure window is refreshed
            cv2.waitKey(1)

            # Break loop on 'q' press
            key = cv2.waitKey(15) & 0xFF  # Wait longer (15ms) to ensure display updates
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error during processing: {str(e)}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        # Add extra waitKey calls to ensure windows are properly closed
        for _ in range(5):
            cv2.waitKey(1)

def main():
    # Hardcoded video path
    video_path = r"E:\Nandhika\Billiance\Product_detection\Product_detection\1101381565-preview.mp4"
    
    # Process video stream
    process_video(video_path)

if __name__ == "__main__":
    main()
