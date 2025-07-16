from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("best.pt")  # put your correct path to best.pt if needed

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run prediction
    results = model(frame)

    # Draw the results on the frame
    annotated_frame = results[0].plot()

    # Show the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
