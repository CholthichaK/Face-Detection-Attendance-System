import cv2
import os
import numpy as np
from deepface import DeepFace
import pickle
from datetime import datetime
import csv

DB_FILE = "database.pkl"
FACE_FOLDER = "faces"
ATTENDANCE_FILE = "attendance.csv"

os.makedirs(FACE_FOLDER, exist_ok=True)

# ---------------- BLUR DETECTION ---------------- #
def is_blurry(image, threshold=120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

# ---------------- CROP FACE REGION ---------------- #
def crop_face_region(frame, x, y, w, h):
    expand_top = int(h * 0.25)
    expand_bottom = int(h * 0.20)

    new_y = max(y - expand_top, 0)
    new_h = h + expand_top + expand_bottom
    new_h = min(new_h, frame.shape[0] - new_y)

    return frame[new_y:new_y + new_h, x:x + w]

# ---------------- SAVE EMBEDDINGS ---------------- #
def save_embeddings():
    embeddings, labels = [], []

    for person in os.listdir(FACE_FOLDER):
        person_path = os.path.join(FACE_FOLDER, person)
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            try:
                emb = DeepFace.represent(
                    img_path,
                    model_name="Facenet",
                    detector_backend="opencv"
                )[0]["embedding"]
                embeddings.append(emb)
                labels.append(person)
            except:
                pass

    with open(DB_FILE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels}, f)

# ---------------- ATTENDANCE ---------------- #
def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])

    with open(ATTENDANCE_FILE, "r") as f:
        for row in csv.reader(f):
            if row and row[0] == name and row[1] == today:
                return

    with open(ATTENDANCE_FILE, "a", newline="") as f:
        csv.writer(f).writerow([name, today, time_now])

    print(f"📝 Attendance marked: {name}")

# ---------------- ADD NEW FACE ---------------- #
def add_new_face():
    name = input("Enter person's name: ").strip()

    if os.path.exists(os.path.join(FACE_FOLDER, name)):
        print("⚠ Person already exists.")
        return

    cap = cv2.VideoCapture(0)
    os.makedirs(os.path.join(FACE_FOLDER, name))

    count = 0
    print("📸 Capturing 15 images...")

    while count < 15:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            faces = DeepFace.extract_faces(frame, detector_backend="opencv")
        except:
            continue

        if not faces:
            cv2.putText(frame, "FACE NOT CLEAR", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Add Face", frame)
            if cv2.waitKey(1) == ord('q'):
                break
            continue

        area = faces[0]["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]
        cropped = crop_face_region(frame, x, y, w, h)

        if is_blurry(cropped):
            cv2.putText(frame, "BLUR DETECTED", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Add Face", frame)
            cv2.waitKey(1)
            continue

        cv2.imwrite(os.path.join(FACE_FOLDER, name, f"{count}.jpg"), cropped)
        count += 1

        cv2.putText(frame, f"{count}/15 Captured", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Add Face", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    save_embeddings()
    print("✅ Face saved")

# ---------------- DETECT MULTIPLE FACES ---------------- #
def detect_face():
    if not os.path.exists(DB_FILE):
        print("⚠ No database found.")
        return

    with open(DB_FILE, "rb") as f:
        data = pickle.load(f)

    embeddings = np.array(data["embeddings"])
    labels = data["labels"]

    cap = cv2.VideoCapture(0)
    print("🔍 Multi-face recognition started (press q to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            faces = DeepFace.extract_faces(
                frame,
                detector_backend="opencv",
                enforce_detection=False
            )
        except:
            continue

        for face in faces:
            area = face["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]
            cropped = crop_face_region(frame, x, y, w, h)

            try:
                emb = DeepFace.represent(
                    cropped,
                    model_name="Facenet",
                    detector_backend="opencv",
                    enforce_detection=False
                )[0]["embedding"]
            except:
                continue

            distances = np.linalg.norm(embeddings - emb, axis=1)
            idx = np.argmin(distances)

            if distances[idx] < 10:
                name = labels[idx]
                color = (0,255,0)
                mark_attendance(name)
            else:
                name = "Unknown"
                color = (0,0,255)

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Multi Face Recognition", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN MENU ---------------- #
while True:
    print("\n===== FACE ATTENDANCE SYSTEM =====")
    print("1. Add New Face")
    print("2. Detect Faces + Attendance")
    print("3. Exit")

    choice = input("Select option (1/2/3): ")

    if choice == "1":
        add_new_face()
    elif choice == "2":
        detect_face()
    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid option")
