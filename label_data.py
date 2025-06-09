import cv2
import mediapipe as mp
import os
import shutil
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split

# --- Configuration ---
SOURCE_DIRS = ["raw_data", "synthetic_data"]
PROCESSED_DIR = "processed_data"
TRAIN_RATIO = 0.85
IMAGE_SIZE = (224, 224) # Size for the classifier

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def count_fingers(hand_landmarks):
    """
    A simple heuristic to count extended fingers based on landmark positions.
    This is a basic implementation and can be improved.
    """
    if hand_landmarks is None:
        return -1 # Indicate no hand found

    # Finger landmark indices
    THUMB_TIP = 4
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_TIP = 16
    PINKY_TIP = 20
    
    # MCP (Metacarpophalangeal) joint indices for reference
    INDEX_FINGER_MCP = 5
    PINKY_MCP = 17

    landmarks = hand_landmarks.landmark
    finger_count = 0

    # Heuristic: A finger is "up" if its tip is above its MCP joint.
    # This is a simplification. A more robust method would compare distances.

    # Y-coordinates decrease as you go up the image.
    # Index Finger
    if landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_MCP].y:
        finger_count += 1
    # Middle Finger
    if landmarks[MIDDLE_FINGER_TIP].y < landmarks[MIDDLE_FINGER_TIP - 2].y:
        finger_count += 1
    # Ring Finger
    if landmarks[RING_FINGER_TIP].y < landmarks[RING_FINGER_TIP - 2].y:
        finger_count += 1
    # Pinky Finger
    if landmarks[PINKY_TIP].y < landmarks[PINKY_MCP].y:
        finger_count += 1
    
    # Thumb: Check if it's horizontally away from the index finger base
    # This is tricky; a simple X-coord check is used here.
    if landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x: # For a right hand
         finger_count += 1
    
    # Note: A left hand would need a different thumb rule (landmarks[THUMB_TIP].x > landmarks[INDEX_FINGER_MCP].x)
    # This simplified model assumes a right hand shown to the camera.

    return finger_count

def process_and_label_images():
    print("--- Starting Data Labeling and Processing ---")
    
    all_files = []
    for source_dir in SOURCE_DIRS:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory '{source_dir}' not found. Skipping.")
            continue
        for filename in os.listdir(source_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_files.append(os.path.join(source_dir, filename))

    if not all_files:
        print("Error: No image files found in source directories. Please run previous steps.")
        return

    labeled_files = {} # {label: [file_path, ...]}

    print(f"Processing {len(all_files)} images...")
    progress_bar = tqdm(all_files, desc="Labeling Images")
    for img_path in progress_bar:
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            finger_count = count_fingers(results.multi_hand_landmarks[0])
            if 0 <= finger_count <= 5:
                if finger_count not in labeled_files:
                    labeled_files[finger_count] = []
                labeled_files[finger_count].append(img_path)
    
    print("\n--- Splitting and Saving Data ---")
    # Clear and create directories
    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    
    for split in ['train', 'val']:
        for i in range(6): # 0-5 fingers
            os.makedirs(os.path.join(PROCESSED_DIR, split, str(i)), exist_ok=True)
            
    for label, files in labeled_files.items():
        if len(files) < 2:
            continue # Not enough data to split
            
        train_files, val_files = train_test_split(files, train_size=TRAIN_RATIO, random_state=42)
        
        # Copy training files
        for file_path in train_files:
            shutil.copy(file_path, os.path.join(PROCESSED_DIR, 'train', str(label)))
            
        # Copy validation files
        for file_path in val_files:
            shutil.copy(file_path, os.path.join(PROCESSED_DIR, 'val', str(label)))

    print("✅ Data processing and labeling complete.")
    print(f"Data saved to '{PROCESSED_DIR}'")


if __name__ == "__main__":
    process_and_label_images()
    hands.close()