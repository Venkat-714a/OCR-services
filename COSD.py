import cv2
import pytesseract
from datetime import datetime
from pymongo import MongoClient

# 🔹 If using Apple Silicon Mac, uncomment this:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


# 🔥 MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["ocr_database"]
collection = db["scanned_texts"]


def ocr_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        10
    )

    custom_config = r'--oem 3 --psm 4'

    text = pytesseract.image_to_string(thresh, config=custom_config)

    return text


def save_to_mongodb(text):
    document = {
        "text": text,
        "timestamp": datetime.now()
    }

    result = collection.insert_one(document)
    print(f"Saved to MongoDB with ID: {result.inserted_id}")


def start_camera_ocr():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 's' to scan text")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        start_x = int(w * 0.2)
        end_x = int(w * 0.8)
        start_y = int(h * 0.2)
        end_y = int(h * 0.8)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.imshow("Camera - OCR Scanner", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("\nScanning...")

            crop = frame[start_y:end_y, start_x:end_x]

            text = ocr_from_frame(crop)

            print("\nExtracted Text:\n")
            print(text)
            print("-" * 50)

            if text.strip():
                save_to_mongodb(text)
            else:
                print("No text detected. Not saving.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera_ocr()
    