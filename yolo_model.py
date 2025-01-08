import cv2
from ultralytics import YOLO
import torch

# check if cuda or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# load YOLO and move to device
model = YOLO('yolov8l.pt')  # yolo Large
model.to(device)

# start cam and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # height

# check resolution of webcam
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam resolution set to: {actual_width}x{actual_height}")

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # yolo infer
    results = model.predict(frame, conf=0.5, device=device)  # Set confidence threshold to 50%

    # annotate frames
    for detection in results[0].boxes:
        # get bbox and class IDs
        class_id = int(detection.cls.item())  # Convert tensor to int
        confidence = float(detection.conf.item())  # Convert tensor to float
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())  # tensor to list then int

        # coco has dog as class 16
        if class_id == 16:
            # draw bbox and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Dog: {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show frame
    cv2.imshow("Dog Tracker", frame)

    # exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close
cap.release()
cv2.destroyAllWindows()
