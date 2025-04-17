# Attendance Management System 🎓📷

This is a Python-based Attendance Management System that uses facial recognition to automate attendance logging. Built using Flask for the web framework and a deep learning model trained to recognize faces.

## 🚀 Features

- 🔒 Secure facial recognition for marking attendance.
- 📊 Real-time tracking and attendance logging.
- 📁 Attendance records saved in Excel format.
- 🌐 Web-based interface using Flask.
- 🧠 Pre-trained deep learning model (`model.h5`) for face recognition.

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- Flask
- NumPy
- Pandas
- HTML/CSS (via Flask templates)

## 📂 Project Structure
📁 attendence management system 178 ├── app.py ├── app1.py ├── app2.py ├── attendance1.xlsx ├── class_new.npy ├── model.h5 ├── requirements.txt ├── face_recognition_dataset/ ├── static/ ├── templates/ └── studentattendence/


## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/attendance-management-system.git
   cd attendance-management-system
   
2. Install dependencies:
   pip install -r requirements.txt
   
3. Run the application:
   python app.py
   
4. Open http://127.0.0.1:5000 in your browser.

📸 Face Dataset
Store student face images under the face_recognition_dataset/ folder, organized by individual student names.

📈 Attendance Records
Attendance is stored and updated in attendance1.xlsx for tracking and future reference.

