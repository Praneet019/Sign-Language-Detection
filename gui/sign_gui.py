import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
import datetime
import sys

# ⏰ Enforce 6 PM - 10 PM restriction
now = datetime.datetime.now()
if not (18 <= now.hour < 22):
    print("⛔ Access denied. The application can only run between 6 PM and 10 PM.")
    sys.exit()

# ✅ Load trained model & label encoder
model = tf.keras.models.load_model("C:/Users/HP/sign_language_detection/model/sign_model.h5")
le = joblib.load("C:/Users/HP/sign_language_detection/model/label_encoder.pkl")
IMG_SIZE = 64

# 🔎 Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def predict_on_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Error", "Could not read image file.")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
            y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20

            # Clamp box inside image
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # 🟦 Draw debug bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            hand_img = img[y_min:y_max, x_min:x_max]
            if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                input_data = np.expand_dims(resized / 255.0, axis=0)

                prediction = model.predict(input_data)
                predicted_label = le.classes_[np.argmax(prediction)]

                cv2.putText(img, f"Prediction: {predicted_label}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(img, "No hand detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def start_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
                y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20

                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(w, x_max), min(h, y_max)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.shape[0] > 0 and hand_img.shape[1] > 0:
                    resized = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                    input_data = np.expand_dims(resized / 255.0, axis=0)

                    prediction = model.predict(input_data)
                    predicted_label = le.classes_[np.argmax(prediction)]

                    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Webcam Prediction", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("❌ Exiting webcam...")
            break
    cap.release()
    cv2.destroyAllWindows()

# 🖼️ Build a cleaner GUI window
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

title = tk.Label(root, text="Sign Language Detection", font=("Arial", 22, "bold"), bg="#f0f0f0")
title.pack(pady=30)

btn_img = tk.Button(root, text="Upload Image", font=("Arial", 16), width=25, command=predict_on_image)
btn_img.pack(pady=20)

btn_webcam = tk.Button(root, text="Start Webcam", font=("Arial", 16), width=25, command=start_webcam)
btn_webcam.pack(pady=20)

info = tk.Label(root, text="Press 'q' in webcam window to exit", font=("Arial", 12), bg="#f0f0f0")
info.pack(pady=10)

root.mainloop()
