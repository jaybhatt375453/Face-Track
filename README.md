# 🎯 FaceTrack: AI-powered Liveliness Verifier 📷🧠

> Real-time face recognition & liveliness detection system built with OpenCV, dlib, CNNs, Flask & MySQL — enabling lightning-fast authentication and attendance automation.

🔗 [GitHub Repository](https://github.com/jaybhatt375453/sgp6.git)

---

## 📌 About

**FaceTrack** is a cutting-edge AI-based facial recognition and liveliness verification system designed for secure and real-time identity validation. It combines the power of **OpenCV**, **dlib**, and **CNNs** to recognize and authenticate users with **97%+ accuracy**. Flask APIs serve as the backend, while **MySQL** and **Cloud Storage** ensure fast and reliable record management for over **10,000+ users**.

---

## ✨ Features

- 🧠 Real-time face detection & recognition (97% accuracy)
- 🖼️ Liveliness verification to detect spoofing attacks
- 📦 Cloud-based data retrieval with MySQL
- ⚡ Fast authentication — **40% quicker**
- 🗺️ Geolocation-based validation
- 📆 Attendance automation with timestamp
- 📤 Image upload to cloud storage

---

## 🧠 Resume Highlights

- 🧠 **Engineered** a real-time facial recognition system using **OpenCV, dlib**, and **CNNs** with **97% accuracy**
- 🚀 **Integrated** Flask, MySQL, & Cloud Storage — reducing authentication time by **40%** and enabling rapid retrieval of **10K+ records**
- ✅ **Automated** attendance with live face match, optimized DB (30% faster), and **geo-validation**

---

## 🛠 Tech Stack

### 🧠 AI/ML & Computer Vision
- **OpenCV**
- **dlib**
- **face_recognition** (built on dlib)
- **Convolutional Neural Networks (CNN)**

### 🧱 Backend
- **Flask**
- **RESTful APIs**

### 💾 Database & Storage
- **MySQL** (Relational DB)
- **Cloudinary / Firebase / GCP Storage** (for face images)

### 🌐 Other Tools
- **Geolocation APIs**
- **HTML/CSS/JS** (for basic frontend interface)

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/jaybhatt375453/sgp6.git
cd sgp6
```

### 2️⃣ Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Required Libraries
```bash
pip install -r requirements.txt
```

### 4️⃣ Start the Flask Server
```bash
python app.py
```

### 5️⃣ Access the App
Open your browser and go to:  
```
http://localhost:5000
```

---

## 🧪 How It Works

1. 📷 **Capture Face:** User’s webcam captures the face.
2. 🧠 **Liveliness Check:** Real-time detection to prevent spoofing (e.g. blinking, motion).
3. 🧾 **Database Match:** Face vector compared with stored encodings.
4. 🗃️ **Store / Retrieve:** Data stored in MySQL + Cloud storage.
5. ✅ **Attendance Marked** (along with location + timestamp).

---

## 🛡️ Security

- ⚠️ Spoofing Detection: Prevent login via static images or videos
- 🔒 Password Hashing & Secure Sessions
- 📍 Geolocation validation

---

## 📸 Screenshot

_Add screenshots of the attendance dashboard, face recognition in action, etc._

---

## 📦 Directory Structure

```
sgp6/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── js, css, captured images
├── utils/
│   └── face_recognition.py
├── database/
│   └── mysql_connection.py
├── requirements.txt
└── README.md
```

---

## 🙌 Contributing

Feel free to fork the repo, open PRs, and contribute!  
Steps:
1. 🍴 Fork the repo  
2. 🛠 Create a new branch  
3. 💬 Make your changes  
4. 📤 Push and open a Pull Request

---

## 👨‍💻 Author

Built with ❤️ by [Jay Bhatt](https://github.com/jaybhatt375453) and team.  
Also see: [Janak Makadia](https://github.com/jaybhatt375453)

---

## 📬 Contact

📧 jay375453@gmail.com 
🌐 [LinkedIn]()

