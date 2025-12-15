# Face-Detection-Attendance-System

# Face Recognition Attendance System

This project is a real-time face recognition–based attendance system built using Python, OpenCV, and DeepFace.
It detects multiple faces simultaneously and records attendance automatically.

---

## Features
- Real-time face detection using webcam
- Multi-face recognition at once
- Face recognition using DeepFace (FaceNet)
- Automatic attendance marking
- No duplicate attendance per person per day
- Blur detection for better accuracy

---

## Technologies Used
- Python 3.11
- OpenCV
- DeepFace
- TensorFlow 2.15.1
- NumPy



---

## Installation

### 1. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

### 2. Install Dependencies
pip install tensorflow==2.15.1
pip install keras==2.15.0
pip install deepface
pip install opencv-python mtcnn retina-face numpy pandas

---

## Run the Project
python main.py

---

## Usage
1. Add New Face
2. Detect Faces + Attendance
3. Exit

Press 'q' to close camera window.

---

## Attendance
- Stored in attendance.csv
- Format: Name | Date | Time
- Each person is recorded once per day

---

## Future Improvements
- Excel attendance export
- GUI interface
- Database integration

---

## Author
CSC369 Computer Vision Project
