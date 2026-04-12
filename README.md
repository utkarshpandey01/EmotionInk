
# 🎨 EmotionInk – AI-Powered Air Drawing with Emotion Detection

EmotionInk is a real-time computer vision application that enables users to draw in the air using hand gestures while dynamically changing the drawing color based on detected facial emotions.

This project combines **hand tracking**, **gesture recognition**, and **facial landmark analysis** to create a seamless, touchless human-computer interaction experience.

---

# 🚀 Features

### ✍️ Air Drawing

* Draw on a virtual canvas using your **index finger**
* No physical contact required

### 🧽 Gesture-Based Eraser

* Raise **index + middle finger** to activate eraser mode
* Natural and intuitive interaction

### 😊 Emotion Detection

* Detects **Happy / Sad / Neutral** emotions using facial landmarks
* Uses mouth geometry for lightweight real-time inference

### 🎨 Emotion-Based Brush Color

* 😄 Happy → Green
* 😢 Sad → Red
* 😐 Neutral → Default

### 🧠 Smooth Drawing (Anti-Shake)

* Implemented using **deque-based averaging**
* Eliminates jitter from hand movement

### ⚡ Real-Time Performance

* Optimized using:

  * Bitwise masking
  * Efficient frame processing
  * Minimal redraw operations

### 🎯 Clean UI

* Custom hand skeleton (no clutter)
* Highlighted fingertip
* Eraser icon overlay

---

# 🛠️ Tech Stack

* **Python**
* **OpenCV** – Image processing & rendering
* **MediaPipe** – Hand & face landmark detection
* **NumPy** – Numerical computations
* **Deque (collections)** – Smoothing algorithm

---

# 🧩 How It Works

### 1. Hand Tracking

* MediaPipe detects **21 hand landmarks**
* Index finger tip (landmark 8) is used as drawing pointer

### 2. Gesture Recognition

* Finger positions are compared:

  * Index up → Draw
  * Index + Middle up → Erase

### 3. Smoothing Algorithm

* Last few points stored in a **deque**
* Average position used → smooth lines

### 4. Emotion Detection

* Face Mesh detects facial landmarks
* Key metrics:

  * Lip distance (open/close)
  * Mouth width
* Emotion classified as Happy / Sad / Neutral

### 5. Canvas Rendering

* Drawing happens on a separate canvas
* Combined with webcam feed using:

  * `bitwise_and`
  * masking techniques

---

# 🎮 Controls

| Gesture                     | Action           |
| --------------------------- | ---------------- |
| ☝️ Index Finger Up          | Draw             |
| ✌️ Index + Middle Finger Up | Erase            |
| ❌ Press 'X'                 | Exit Application |

---

# ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EmotionInk.git
cd EmotionInk
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Project

```bash
python main.py
```

# ⚡ Challenges Faced

* Stabilizing shaky hand movements
* Real-time emotion detection without heavy ML models
* Efficiently merging canvas with live video
* Handling gesture misclassification

---

# 🔮 Future Improvements

* 🎨 Add multiple color selection gestures
* 😠 Add more emotions (Angry, Surprise)
* 🟦 Shape drawing (Circle, Rectangle)
* 💾 Save drawing feature
* 🖱️ Virtual mouse integration
* 🧩 GUI buttons and UI panel

---

# 📚 Key Learnings

* Real-time computer vision pipeline design
* MediaPipe landmark system
* Gesture recognition logic
* Performance optimization in OpenCV
* Human-computer interaction design

---

# 👨‍💻 Author

**Utkarsh Pandey**

Computer Vision | AI | OpenCV | MediaPipe

---

# 💡 Final Note

EmotionInk is more than just a drawing app — it's a step toward **touchless interfaces**, where humans interact with machines naturally using gestures and expressions.

---

