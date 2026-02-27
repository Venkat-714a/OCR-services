import cv2
import pytesseract
import re
import os
import warnings
import sys
from datetime import datetime
from pymongo import MongoClient

# Suppress macOS system warnings about Continuity Cameras
warnings.filterwarnings('ignore')
os.environ['OPENCV_AVFOUNDATION_SKIP_AUTH'] = '1'

# 🔹 If using Apple Silicon Mac, uncomment this:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# 🔥 MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["ocr_database"]
collection = db["scanned_texts"]


def clean_text(text):
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def speak_text(text):
    # Limit length to avoid very long speech
    short_text = text[:1000]
    os.system(f'say "{short_text}"')


def ocr_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=20)
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    custom_config = r'--oem 3 --psm 3 -l eng'

    text = pytesseract.image_to_string(thresh, config=custom_config)

    return clean_text(text)


def save_to_mongodb(text):
    document = {
        "text": text,
        "length": len(text),
        "timestamp": datetime.now()
    }

    result = collection.insert_one(document)
    print(f"✅ Saved to MongoDB with ID: {result.inserted_id}")


def start_camera_ocr():
    try:
        cap = cv2.VideoCapture(0)
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        
        # Give camera time to initialize
        import time
        time.sleep(1)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            print("Make sure camera permissions are granted in System Preferences.")
            return

        print("Press 's' to scan text")
        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            h, w, _ = frame.shape

            start_x = int(w * 0.15)
            end_x = int(w * 0.85)
            start_y = int(h * 0.15)
            end_y = int(h * 0.85)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            cv2.imshow("Camera - OCR Scanner", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("\n🔍 Scanning...")

                crop = frame[start_y:end_y, start_x:end_x]

                text = ocr_from_frame(crop)

                print("\n📄 Extracted Text:\n")
                print(text)
                print("-" * 60)

                if len(text) > 20:
                    save_to_mongodb(text)

                    print("🔊 Reading text aloud...")
                    speak_text(text)

                else:
                    print("⚠️ Text too short / unclear. Not saving.")

            elif key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n\n⛔ Interrupted by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera closed.")


if __name__ == "__main__":
    start_camera_ocr()