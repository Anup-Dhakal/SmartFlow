import streamlit as st
import cv2
import tempfile
from collections import defaultdict
from ultralytics import YOLO
import torch
import os
from pathlib import Path
import math
import time
from PIL import Image, ImageDraw

# Suppress OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda:0" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
print(f"Using device: {device}")

# Load YOLO model with half-precision for faster inference
model = YOLO(r"trained_model\100best.pt")
model.fp16 = True  # Enable mixed precision
model.to(device)

# Global variables for vehicle tracking and counting
vehicle_tracker = defaultdict(lambda: {
    'positions': [],
    'counted': False,
    'lane': None,
    'direction': None,
    'speed': 0.0,
    'overspeeding': False
})
lane_outgoing = {'two_wheeler': 0, 'four_wheeler': 0}  # For vehicles on the right (Outgoing)
lane_incoming = {'two_wheeler': 0, 'four_wheeler': 0}  # For vehicles on the left (Incoming)
counting_line_y = 350  # Adjust this based on your video
speed_limit = 50  # Speed limit in km/h
overspeeding_vehicles = {}  # Track IDs and speeds of overspeeding vehicles
overspeeding_images = {}  # Store images of overspeeding vehicles
displayed_overspeeding_ids = set()  # Track IDs of displayed overspeeding vehicles

# Traffic light control variables
traffic_light_states = {
    'incoming': {'color': 'green', 'last_change': time.time(), 'count': 0},
    'outgoing': {'color': 'red', 'last_change': time.time(), 'count': 0}
}

# Thresholds and timers for traffic signal control
LOW_DENSITY_THRESHOLD = 5  # Minimum vehicle count for low traffic
HIGH_DENSITY_THRESHOLD = 20  # Maximum vehicle count for high traffic
RED_THRESHOLD = 10  # Vehicle count that triggers a red light change
T_MIN = 10  # Minimum green light duration (seconds)
T_MAX = 30  # Maximum green light duration (seconds)
T_RED_MAX = 20  # Maximum red light duration (seconds)
YELLOW_TIME = 5  # Fixed yellow light duration (seconds)
RED_BUFFER_TIME = 3  # Buffer time before switching from red to green (seconds)

# Color definitions for lanes and overspeeding
COLORS = {
    'incoming': (255, 255, 0),  # Cyan for incoming lane
    'outgoing': (0, 255, 0),    # Green for outgoing lane
    'overspeed': (0, 0, 255)    # Red for overspeeding
}

# Constants for speed calculation
MOVING_AVERAGE_FRAMES = 3  # Number of frames to consider for moving average
CONVERSION_FACTOR =  0.10091  # Meters per pixel (calibrate this for your video)
MIN_SPEED_THRESHOLD = 5  # Minimum speed threshold (km/h)
MAX_SPEED_THRESHOLD = 150  # Maximum speed threshold (km/h)

# Helper function to draw text with a white background
def put_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.6, text_color=(0, 255, 0),
                             bg_color=(255, 255, 255), thickness=2, padding=5):
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x - padding, y - text_height - padding),
                  (x + text_width + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness)

# Function to create compact horizontal traffic lights
def create_traffic_light(color, label):
    light_size = (200, 80)
    base_image = Image.new("RGB", light_size, "black")
    draw = ImageDraw.Draw(base_image)
    light_positions = {
        'red': (30, 30, 70, 70),
        'yellow': (90, 30, 130, 70),
        'green': (150, 30, 190, 70)
    }
    for light, pos in light_positions.items():
        fill_color = color if light == color else "#333333"
        draw.ellipse(pos, fill=fill_color)
    draw.text((100, 50), label, fill="white", anchor="mm")
    return base_image

# Function to update traffic lights based on vehicle density
def update_traffic_lights():
    current_time = time.time()
    
    for direction in ['incoming', 'outgoing']:
        state = traffic_light_states[direction]
        time_elapsed = current_time - state['last_change']
        vehicle_count = state['count']

        # Green Light Phase
        if state['color'] == 'green':
            if vehicle_count < LOW_DENSITY_THRESHOLD:
                # Low density: set green light for minimum time
                if time_elapsed >= T_MIN:
                    state['color'] = 'yellow'
                    state['last_change'] = current_time
            elif vehicle_count > HIGH_DENSITY_THRESHOLD:
                # High density: set green light for maximum time
                if time_elapsed >= T_MAX:
                    state['color'] = 'yellow'
                    state['last_change'] = current_time
            else:
                # Medium density: proportional green light duration
                green_time = T_MIN + (T_MAX - T_MIN) * (vehicle_count - LOW_DENSITY_THRESHOLD) / (HIGH_DENSITY_THRESHOLD - LOW_DENSITY_THRESHOLD)
                if time_elapsed >= green_time:
                    state['color'] = 'yellow'
                    state['last_change'] = current_time

        # Yellow Light Phase
        elif state['color'] == 'yellow':
            if time_elapsed >= YELLOW_TIME:
                state['color'] = 'red'
                state['last_change'] = current_time

        # Red Light Phase
        elif state['color'] == 'red':
            if time_elapsed >= T_RED_MAX:
                # Maximum red light duration reached: switch to green
                state['color'] = 'green'
                state['last_change'] = current_time
            elif vehicle_count > RED_THRESHOLD:
                # Check if buffer time has passed
                if time_elapsed >= RED_BUFFER_TIME:
                    state['color'] = 'green'
                    state['last_change'] = current_time

# Function to calculate speed using moving average
def calculate_speed(vehicle, frame_time):
    if len(vehicle['positions']) >= MOVING_AVERAGE_FRAMES:
        # Get the position from MOVING_AVERAGE_FRAMES ago
        prev_x, prev_y = vehicle['positions'][-MOVING_AVERAGE_FRAMES]
        current_x, current_y = vehicle['positions'][-1]

        # Calculate distance traveled in pixels
        distance_px = math.hypot(current_x - prev_x, current_y - prev_y)

        # Calculate speed in pixels per second
        speed_px_per_sec = distance_px / (MOVING_AVERAGE_FRAMES * frame_time)

        # Convert speed to km/h using the conversion factor
        speed_kmh = speed_px_per_sec * CONVERSION_FACTOR * 3.6

        # Filter unrealistic speeds
        if MIN_SPEED_THRESHOLD <= speed_kmh <= MAX_SPEED_THRESHOLD:
            return speed_kmh
        else:
            return 0  # Ignore unrealistic speeds
    else:
        return 0  # Not enough data to calculate speed

# Streamlit UI
st.header("Speed Estimation and Traffic Signal Simulation")
st.sidebar.header("Video Source Options")
option = st.sidebar.radio("Choose a video source:", ("Upload Video", "Use Personal Camera", "Use External Camera"))

# Add sliders for confidence and IoU thresholds
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.4, 0.05)

uploaded_file = None
camera_url = None
camera_index = 0

if option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
elif option == "Use Personal Camera":
    camera_index = st.sidebar.selectbox("Select Camera Index", [0, 1, 2])
elif option == "Use External Camera":
    camera_url = st.sidebar.text_input("Enter Camera URL or IP Address (e.g., http://192.168.x.x:4747/video)")

# Initialize video capture based on sidebar input
if option == "Upload Video" and uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
elif option == "Use Webcam":
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Error: Could not open webcam. Ensure Iriun Webcam is running.")
        cap = None
elif option == "Use External Camera" and camera_url:
    cap = cv2.VideoCapture(camera_url)
else:
    cap = None

if cap is not None and cap.isOpened():
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 FPS if not available
    st.write(f"Video FPS: {video_fps}")

    # Create a two-column layout: video on the left, traffic lights on the right
    col1, col2 = st.columns([3, 1])

    with col1:
        video_placeholder = st.empty()

    with col2:
        st.markdown("### Traffic Lights")
        light_col1, light_col2 = st.columns(2)
        with light_col1:
            st.markdown("**Incoming**")
            incoming_light = st.empty()
        with light_col2:
            st.markdown("**Outgoing**")
            outgoing_light = st.empty()

    # Create two columns for incoming and outgoing counts below the video
    st.markdown("### Vehicle Counts")
    count_col1, count_col2 = st.columns(2)
    with count_col1:
        st.markdown("🚗 **Incoming Vehicles**")
        incoming_two_wheeler = st.empty()
        incoming_four_wheeler = st.empty()
        incoming_total = st.empty()

    with count_col2:
        st.markdown("🚗 **Outgoing Vehicles**")
        outgoing_two_wheeler = st.empty()
        outgoing_four_wheeler = st.empty()
        outgoing_total = st.empty()

    # Add a section for overspeeding vehicles
    st.markdown(f"🚨 **Overspeeding Vehicles ID ({speed_limit}> km/h)**")
    overspeeding_placeholder = st.empty()

    # Define the tracker configuration file path (update as needed)
    TRACKER_CONFIGS = Path(os.getenv("YOLO_TRACKER_DIR", "trackers"))
    tracking_config = TRACKER_CONFIGS / "botsort.yaml"

    # Placeholder for overspeeding vehicle images
    st.subheader("Overspeeding Vehicle Snapshots")
    overspeeding_images_placeholder = st.empty()

    # Main loop to process video frames
    prev_time = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate real-time FPS and frame time
        current_time = time.time()
        frame_time = current_time - prev_time
        prev_time = current_time
        processing_fps = 1 / frame_time if frame_time != 0 else 0

        # Perform tracking on the current frame with confidence and IoU thresholds
        results = model.track(
            frame,
            persist=True,
            tracker=str(tracking_config),
            conf=confidence_threshold,
            iou=iou_threshold,
            verbose=False  # Disable logging for better performance
        )

        # Draw the counting line on the frame
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (0, 0, 255), 2)

        # Reset traffic light counters for this frame
        traffic_light_states['incoming']['count'] = 0
        traffic_light_states['outgoing']['count'] = 0

        # Process each detected object
        for box in results[0].boxes:
            if box.id is None:
                continue

            track_id = int(box.id.item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            cls = int(box.cls.item())  # Class ID (0: Four Wheeler, 1: Two Wheeler)

            # Update vehicle tracking history
            vehicle = vehicle_tracker[track_id]
            vehicle['positions'].append((x_center, y_center))
            if len(vehicle['positions']) > MOVING_AVERAGE_FRAMES:
                vehicle['positions'].pop(0)

            # Calculate speed using moving average
            speed_kmh = calculate_speed(vehicle, frame_time)
            vehicle['speed'] = speed_kmh

            # Check for overspeeding
            if speed_kmh > speed_limit:
                vehicle['overspeeding'] = True
                overspeeding_vehicles[track_id] = speed_kmh

                # Capture the image of the overspeeding vehicle
                if track_id not in overspeeding_images:
                    vehicle_img = frame[y1:y2, x1:x2]
                    if vehicle_img.size != 0:
                        resized_img = cv2.resize(vehicle_img, (100, 70))
                        overspeeding_images[track_id] = resized_img
            else:
                vehicle['overspeeding'] = False

            # Determine lane based on horizontal position
            lane = "incoming" if x_center < width / 2 else "outgoing"
            vehicle['lane'] = lane

            # Determine direction based on previous position
            if len(vehicle['positions']) >= 2:
                prev_y = vehicle['positions'][-2][1]
                direction = "incoming" if y_center < prev_y else "outgoing"
                vehicle['direction'] = direction

                # Update traffic light counters
                traffic_light_states[direction]['count'] += 1

                # Check if the vehicle crosses the counting line
                if (prev_y > counting_line_y and y_center <= counting_line_y) or \
                   (prev_y < counting_line_y and y_center >= counting_line_y):
                    if not vehicle['counted']:
                        if lane == "outgoing":
                            if cls == 0:
                                lane_outgoing['four_wheeler'] += 1
                            elif cls == 1:
                                lane_outgoing['two_wheeler'] += 1
                        else:
                            if cls == 0:
                                lane_incoming['four_wheeler'] += 1
                            elif cls == 1:
                                lane_incoming['two_wheeler'] += 1
                        vehicle['counted'] = True

            # Determine box color (red for overspeeding, lane color otherwise)
            if vehicle['overspeeding']:
                box_color = COLORS['overspeed']  # Red for overspeeding
            else:
                box_color = COLORS[lane]  # Lane-specific color

            # Draw bounding box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            class_name = "four_wheeler" if cls == 0 else "two_wheeler"
            text_color = (0, 0, 0) if vehicle['overspeeding'] else (255, 255, 255)
            put_text_with_background(frame, f"ID: {track_id} ", 
                                   (x1, y1 - 10), text_color=text_color, 
                                   bg_color=box_color)
            put_text_with_background(frame, f"{vehicle['speed']:.1f} km/h",
                                   (x1, y2 + 20), text_color=text_color,
                                   bg_color=box_color)

        # Update traffic lights every second
        if time.time() - traffic_light_states['incoming']['last_change'] > 1:
            update_traffic_lights()

        # Display compact traffic lights
        incoming_img = create_traffic_light(traffic_light_states['incoming']['color'], "IN")
        outgoing_img = create_traffic_light(traffic_light_states['outgoing']['color'], "OUT")
        incoming_light.image(incoming_img, use_container_width=True)
        outgoing_light.image(outgoing_img, use_container_width=True)

        # Update counts display
        incoming_two_wheeler.metric("Two Wheelers", lane_incoming['two_wheeler'])
        incoming_four_wheeler.metric("Four Wheelers", lane_incoming['four_wheeler'])
        incoming_total.metric("Total", lane_incoming['two_wheeler'] + lane_incoming['four_wheeler'])
        
        outgoing_two_wheeler.metric("Two Wheelers", lane_outgoing['two_wheeler'])
        outgoing_four_wheeler.metric("Four Wheelers", lane_outgoing['four_wheeler'])
        outgoing_total.metric("Total", lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler'])

        # Convert frame from BGR to RGB and display
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        # Update overspeeding vehicles list and total count
        overspeeding_placeholder.markdown(
            f"{', '.join([f'ID: {k}, Speed: {v:.2f} km/h' for k, v in overspeeding_vehicles.items()]) if overspeeding_vehicles else 'None'}", 
            unsafe_allow_html=True)

        # Display overspeeding vehicle images in a grid layout
        if overspeeding_images:
            # Create a container for the grid
            grid_container = overspeeding_images_placeholder.container()
            
            # Split images into groups of 4 for grid rows
            cols_per_row = 4
            image_items = list(overspeeding_images.items())
            
            for i in range(0, len(image_items), cols_per_row):
                row_images = image_items[i:i+cols_per_row]
                cols = grid_container.columns(cols_per_row)
                
                for col_idx, (track_id, img) in enumerate(row_images):
                    with cols[col_idx]:
                        st.image(
                            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            caption=f"ID: {track_id} | Speed: {overspeeding_vehicles[track_id]:.1f} km/h",
                            use_container_width=True
                        )
        else:
            # Clear the container if no overspeeding vehicles
            overspeeding_images_placeholder.empty()

    # Release video capture when finished
    cap.release()

    # Display final counts after processing ends
    st.markdown("", unsafe_allow_html=True)
    st.subheader("Final Counts")
    st.write(f"Total Incoming Vehicles: {lane_incoming['two_wheeler'] + lane_incoming['four_wheeler']}")
    st.write(f"Total Outgoing Vehicles: {lane_outgoing['two_wheeler'] + lane_outgoing['four_wheeler']}")
    st.write(f"Overspeeding Vehicles: {', '.join([f'ID: {k}, Speed: {v:.2f} km/h' for k, v in overspeeding_vehicles.items()]) if overspeeding_vehicles else 'None'}")
    st.write(f"Total Overspeeding Vehicles: {len(overspeeding_vehicles)}")
else:
    st.error(" Please select a valid video source from the sidebar.")