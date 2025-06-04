import cv2
from deepface import DeepFace
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
last_emotion = None  # To avoid repeating the same emotion again and again

# Load video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze emotion
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = results[0]['dominant_emotion']

        # Speak if emotion changed
        if emotion != last_emotion:
            engine.say(f"You are {emotion}")
            engine.runAndWait()
            last_emotion = emotion

        # Display emotion on frame
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except Exception as e:
        print("Error analyzing emotion:", e)

    cv2.imshow("Emotion Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
