

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Load eraser icon
eraser_icon = cv2.imread(r"C:\Users\utkar\Desktop\OpenCV Project\eraser.png", cv2.IMREAD_UNCHANGED)
eraser_icon = cv2.resize(eraser_icon, (40, 40))


# MediaPipe Setup
mp_draw = mp.solutions.drawing_utils
mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Drawing setup
canvas = None
brush_thickness = 4
eraser_thickness = 60
current_color = (0, 255, 0)

# Smoothing using deque
points = deque(maxlen=7)

# Previous position
px, py = 0, 0

def overlay_icon(frame, icon, x, y):
    h, w = icon.shape[:2]

    # Check boundaries
    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
        return frame  # Don't draw if out of frame

    # If PNG has alpha channel
    if icon.shape[2] == 4:
        alpha = icon[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (
                alpha * icon[:, :, c] +
                (1 - alpha) * frame[y:y+h, x:x+w, c]
            )
    else:
        frame[y:y+h, x:x+w] = icon

    return frame


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb)
    face_results = face_mesh.process(rgb)
   
    # ---------------- FACE EMOTION (FAST + ACCURATE) ----------------
    emotion_buffer = deque(maxlen=5)
    current_emotion = "NEUTRAL"

    # ---------------- FACE EMOTION (HAPPY vs SAD) ----------------
    if face_results.multi_face_landmarks:           # face_results comes from MediaPipe face mesh.
                                                    # multi_face_landmarks = list of faces detected.
                                                    # If a face is found → run the code.
        flm = face_results.multi_face_landmarks[0].landmark

        lip_dist = abs(flm[13].y - flm[14].y)  # mouth open         # Point 13 = upper lip
                                                                    # Point 14 = lower lip
        mouth_width = abs(flm[61].x - flm[291].x)  # horizontal stretch
            # Point 61 = left side of mouth
            # Point 291 = right side of mouth
    if lip_dist > 0.009:
        emotion_buffer.append("HAPPY")
    elif lip_dist < 0.02 and mouth_width < 0.25:
        emotion_buffer.append("SAD")
    else:
        emotion_buffer.append("NEUTRAL")

    current_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
 
    if current_emotion == "HAPPY":
        current_color = (0, 255, 0)   # Green
    elif current_emotion == "SAD":
        current_color = (0, 0, 255)   # Sad (sad vibe)

    cv2.putText(frame, f'Emotion: {current_emotion}', (10, 40), 1, 1.5, current_color, 2)

   
    # ---------------- HAND DRAWING + CLEAN SKELETON ----------------
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:

            # Draw skeleton lines ONLY
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands_connections,
                None,  # ❌ removes default points
                mp_draw.DrawingSpec(color=(127, 127, 127), thickness=2)
            )

            hlm = hand_landmarks.landmark  # ✅ FIXED

            # Draw custom circles (clean look)
            for lm in hlm:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 127, 255), -1)

            # Index finger tip
            x = int(hlm[8].x * w)
            y = int(hlm[8].y * h)

            # Highlight drawing finger
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # Smoothing
            points.append((x, y))
            avg_x = int(np.mean([p[0] for p in points]))
            avg_y = int(np.mean([p[1] for p in points]))

            index_up = hlm[8].y < hlm[6].y
            middle_up = hlm[12].y < hlm[10].y
            thumb_up = hlm[4].x < hlm[2].x


            if index_up and middle_up:
                cv2.putText(frame, "MODE: ERASER", (10, 80), 1, 2, (0, 255, 255), 2)

                 # Show eraser icon on finger
                # frame = overlay_icon(frame, eraser_icon, x, y)
                frame = overlay_icon(frame, eraser_icon, x-20, y-60)

                if px != 0:
                    cv2.line(canvas, (px, py), (avg_x, avg_y), (0, 0, 0), eraser_thickness)

            elif index_up:
                cv2.putText(frame, "MODE: DRAW ", (10, 80), 1, 2, current_color, 2)
                if px != 0:
                    cv2.line(canvas, (px, py), (avg_x, avg_y), current_color, brush_thickness, cv2.LINE_AA)

            px, py = avg_x, avg_y

    else:
        px, py = 0, 0
        points.clear()

    # ---------------- FAST OVERLAY ----------------0
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    final = cv2.add(frame_bg, canvas_fg)

    cv2.imshow("Air Drawing", final)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()