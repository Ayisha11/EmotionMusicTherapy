import cv2
from fer.fer import FER
import pygame
import os

# Initialize emotion detector
detector = FER()

# Initialize music player
pygame.mixer.init()

# Music mapping
music_map = {
    "happy": "music/happy.mp3",
    "sad": "music/sad.mp3",
    "angry": "music/angry.mp3",
    "fear": "music/fear.mp3",
    "surprise": "music/surprise.mp3",
    "disgust": "music/disgust.mp3",
    "neutral": "music/neutral.mp3"
}

# Open webcam
cap = cv2.VideoCapture(0)

print("===================================")
print(" Press SPACE to capture photo")
print(" Press ESC to exit")
print("===================================")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    cv2.imshow("Emotion Capture", frame)

    key = cv2.waitKey(1)

    # SPACE BAR
    if key == 32:

        # Save image
        cv2.imwrite("captured.jpg", frame)

        print("\nPhoto captured successfully!")

        # Detect emotions
        results = detector.detect_emotions(frame)

        if results:

            emotions = results[0]["emotions"]

            dominant = max(emotions, key=emotions.get)

            print("\n===================================")
            print(" EMOTION ANALYSIS")
            print("===================================")

            for emotion, score in emotions.items():
                print(f"{emotion:<10}: {score*100:.2f}%")

            print("\nDominant Emotion:", dominant)

            # Write to emotion.txt
            unity_file = "/Users/ayishasalmira/EmotionTherapyRoom/shared/emotion.txt"
            with open(unity_file, "w") as f:
                f.write(f"Dominant:{dominant}\n")

                for emotion, score in emotions.items():
                    f.write(
                        f"{emotion}:{score*100:.2f}\n"
                    )

            print("\nData written to shared/emotion.txt")

            # Play music
            if dominant in music_map:

                music_file = music_map[dominant]

                if os.path.exists(music_file):

                    pygame.mixer.music.stop()
                    pygame.mixer.music.load(music_file)
                    pygame.mixer.music.play()

                    print(f"\nPlaying music for: {dominant}")

                else:
                    print(
                        f"\nMusic file not found: {music_file}"
                    )

        else:
            print("No face detected!")

        break

    # ESC
    elif key == 27:
        print("Exiting...")
        break

# Keep music playing
print("\nPress ENTER to exit...")
input()

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
