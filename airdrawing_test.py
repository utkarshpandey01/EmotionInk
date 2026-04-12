import cv2
import mediapipe as mp
import numpy as np
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# MediaPipe
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_x, prev_y = 0, 0
smoothening = 5
brush_thickness = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            lm_list = []

            for lm in hand_landmarks:
                x = int(lm.x * 640)
                y = int(lm.y * 480)
                lm_list.append((x, y))
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            if lm_list:
                x1, y1 = lm_list[8]

                # Smooth coordinates
                curr_x = prev_x + (x1 - prev_x) / smoothening
                curr_y = prev_y + (y1 - prev_y) / smoothening

                index_up = lm_list[8][1] < lm_list[6][1]
                middle_up = lm_list[12][1] < lm_list[10][1]
                thumb_up = lm_list[4][0] < lm_list[3][0]

                # Draw
                if index_up and not middle_up:
                    cv2.circle(frame, (int(curr_x), int(curr_y)), 8, (0, 255, 0), -1)

                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = curr_x, curr_y

                    # Draw smooth line
                    cv2.line(canvas, (int(prev_x), int(prev_y)),
                             (int(curr_x), int(curr_y)),
                             (0, 255, 0), brush_thickness)

                    prev_x, prev_y = curr_x, curr_y

                # Stop drawing
                elif index_up and thumb_up:
                    prev_x, prev_y = 0, 0

                # Clear canvas
                if middle_up:
                    canvas[:] = 0

    # Merge canvas and frame properly
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Drawing", frame)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()