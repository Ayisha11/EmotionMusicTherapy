import cv2
from fer.fer import FER
detector = FER()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read frame")
        break
    results = detector.detect_emotions(frame)
    if results:
        x, y, w, h = results[0]["box"]
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )
        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        cv2.putText(
            frame,
            f"Detected: {dominant_emotion}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )
        start_y = 70
        for emotion, score in emotions.items():
            text = f"{emotion}: {score * 100:.1f}%"
            cv2.putText(
                frame,
                text,
                (20, start_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            start_y += 30
    cv2.imshow("AR Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
