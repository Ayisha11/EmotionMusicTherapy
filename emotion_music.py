import cv2
from fer.fer import FER
import pygame
import os

# Initialize pygame mixer
pygame.mixer.init()

# Emotion detector
detector = FER()

# Webcam
cap = cv2.VideoCapture(0)

# Music files
music_map = {
    "happy": "music/happy.mp3",
    "sad": "music/sad.mp3",
    "angry": "music/angry.mp3",
    "neutral": "music/calm.mp3"
}

current_emotion = ""

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = detector.detect_emotions(frame)

    if results:

        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        cv2.putText(
            frame,
            dominant_emotion,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Only update when emotion changes
        if dominant_emotion != current_emotion:

            current_emotion = dominant_emotion

            # Update emotion.txt
            with open("shared/emotion.txt", "w") as f:
                f.write(current_emotion)

            print("Emotion file updated:", current_emotion)

            # Play music
            if current_emotion in music_map:

                music_file = music_map[current_emotion]

                if os.path.exists(music_file):

                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(music_file)
                    pygame.mixer.music.play()

                    print("Playing music for:", current_emotion)

    cv2.imshow("Emotion Music Therapy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
