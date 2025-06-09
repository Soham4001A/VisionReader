import cv2
import torch
import mediapipe as mp
from torchvision import models, transforms
import numpy as np
import torch.nn as nn

# --- Configuration ---
MODEL_PATH = "models/finger_classifier.pth"
NUM_CLASSES = 6 # 0-5 fingers
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load the trained model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        print("Please run '5_classifier_trainer.py' first.")
        return

    model = models.resnet18(weights=None) # No pre-trained weights needed here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Classifier model loaded.")

    # --- Image pre-processing for the model ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- MediaPipe Hand Detection ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- OpenCV Webcam Feed ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    print("\n🚀 Starting Real-Time Demo... Press 'q' to quit.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for visualization
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Bounding Box Calculation ---
                h, w, c = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if x > x_max: x_max = x
                    if y < y_min: y_min = y
                    if y > y_max: y_max = y
                
                # Add padding to the bounding box
                padding = 30
                x_min -= padding
                y_min -= padding
                x_max += padding
                y_max += padding
                
                # Ensure box is within frame bounds
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)
                
                # --- Prediction ---
                if x_max > x_min and y_max > y_min:
                    hand_crop = frame[y_min:y_max, x_min:x_max]
                    hand_tensor = transform(hand_crop).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(hand_tensor)
                        _, predicted = torch.max(outputs.data, 1)
                        finger_count = predicted.item()

                    # Draw bounding box and prediction
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    label_text = f"Fingers: {finger_count}"
                    cv2.putText(frame, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('VAE-Augmented Hand Pose Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == '__main__':
    main()