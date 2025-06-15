from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load YOLOv8-pose model
model = YOLO("yolov8n-pose.pt")

# Open webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Warm color palette (BGR) for lava drops
warm_colors = [
    (0, 0, 255),      # Red
    (0, 165, 255),    # Orange
    (0, 255, 255),    # Yellow
    (10, 100, 255),   # Light orange
    (20, 200, 255)    # Pale orange
]

# Initialize lava drops
num_drops = 500
lava_drops = []
for i in range(num_drops):
    drop = {
        'x': random.randint(0, frame_width),
        'y': random.randint(0, frame_height),
        'speed': random.randint(1, 5),
        'color': random.choice(warm_colors)
    }
    lava_drops.append(drop)

# Skeleton connection pairs (without facial details)
skeleton_pairs = [
    # Left arm
    (5, 7), (7, 9),
    # Right arm
    (6, 8), (8, 10),
    # Torso
    (5, 6), (5, 11), (6, 12), (11, 12),
    # Left leg
    (11, 13), (13, 15),
    # Right leg
    (12, 14), (14, 16)
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the model to get keypoints
    results = model(frame)

    # Create a black background
    black_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    boxes = []  # Used for lava drop collision detection
    keypoints = []
    if len(results) > 0:
        res = results[0]
        # Extract bounding boxes (only for collision check)
        if hasattr(res, "boxes") and res.boxes is not None:
            boxes_array = (
                res.boxes.xyxy.cpu().numpy() 
                if hasattr(res.boxes.xyxy, "cpu") 
                else res.boxes.xyxy
            )
            boxes = boxes_array.tolist()
        # Extract keypoints
        if hasattr(res, "keypoints") and res.keypoints is not None:
            kpts = (
                res.keypoints.xy.cpu().numpy() 
                if hasattr(res.keypoints.xy, "cpu") 
                else res.keypoints.xy
            )
            keypoints = kpts.tolist()

    # Draw skeletons
    for person in keypoints:
        # Draw lines between keypoint pairs
        for (p1, p2) in skeleton_pairs:
            if p1 < len(person) and p2 < len(person):
                x1, y1 = person[p1]
                x2, y2 = person[p2]
                cv2.line(black_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                         (255, 255, 255), 2)
        # Draw head using the nose point if available
        if len(person) > 0:
            nose = person[0]
            cv2.circle(black_frame, (int(nose[0]), int(nose[1])), 10, (255, 255, 255), 2)

    # Update and draw lava drops
    for drop in lava_drops:
        drop['y'] += drop['speed']
        if drop['y'] > frame_height:
            drop['y'] = 0
            drop['x'] = random.randint(0, frame_width)
        
        # Check collision with detected humans using bounding boxes
        inside_human = False
        for b in boxes:
            x_min, y_min, x_max, y_max = b
            if x_min <= drop['x'] <= x_max and y_min <= drop['y'] <= y_max:
                inside_human = True
                break
        
        if not inside_human:
            cv2.circle(black_frame, (drop['x'], drop['y']), 5, drop['color'], -1)

    # Show the final frame (skeleton + lava drops over black background)
    cv2.imshow("Skeleton Without Tracking Box", black_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
