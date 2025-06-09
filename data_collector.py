import cv2
import os
import time

# --- Configuration ---
SAVE_PATH = "raw_data"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
IMAGE_COUNT_GOAL = 200 # Collect 200 images

def main():
    """
    Captures and saves images from the webcam to build the initial dataset.
    Instructions:
    - Show your hand to the camera in various poses (0-5 fingers).
    - Vary the lighting and background.
    - Press 's' to save an image.
    - Press 'q' to quit.
    """
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print(f"Created directory: {SAVE_PATH}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    count = len(os.listdir(SAVE_PATH))
    print("Starting data collection. Press 's' to save, 'q' to quit.")

    while count < IMAGE_COUNT_GOAL:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame. Exiting ...")
            break
        
        # Flip the frame horizontally for a more intuitive selfie-view
        frame = cv2.flip(frame, 1)

        # Display instructions and current count
        text = f"Images Saved: {count}/{IMAGE_COUNT_GOAL}. Press 's' to save, 'q' to quit."
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Data Collector', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_name = os.path.join(SAVE_PATH, f"hand_{int(time.time())}_{count}.png")
            # Save a cropped, resized version for training
            # You can adjust cropping later, for now save the whole frame
            cv2.imwrite(img_name, frame)
            print(f"💾 Saved {img_name}")
            count += 1
        elif key == ord('q'):
            break

    print(f"\nData collection finished. Total images saved: {count}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()