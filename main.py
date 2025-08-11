import cv2
import numpy as np
from deepface import DeepFace
import pygame
import time
import os

# Initialize Pygame mixer for alarm
pygame.mixer.init()
ALARM_SOUND = "alarm.mp3"
ALARM_COOLDOWN = 10  # seconds
last_alarm_time = 0

# Validate alarm sound file
if not os.path.exists(ALARM_SOUND):
    print(f"Alarm sound file not found: {ALARM_SOUND}")
    ALARM_SOUND = None

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise RuntimeError("Could not load face cascade classifier")

# Authorized faces and their encodings
authorized_faces = ["authorized/Photo1.jpeg"]
authorized_encodings = []

# Precompute encodings for authorized faces
for auth_face in authorized_faces:
    if not os.path.exists(auth_face):
        print(f"Authorized face image not found: {auth_face}")
        continue
    try:
        faces = DeepFace.extract_faces(auth_face, detector_backend='mtcnn', enforce_detection=False)
        print(f"Number of faces detected in {auth_face}: {len(faces)}")
        if faces and len(faces) > 0:
            face_img = faces[0]['face']  # Extract NumPy array of face
            # Save detected face for inspection
            cv2.imwrite(f"detected_auth_face_{os.path.basename(auth_face)}", face_img)
            print(f"Saved detected face from {auth_face} as detected_auth_face_{os.path.basename(auth_face)}")
            embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
            if embedding and len(embedding) > 0:
                authorized_encodings.append(embedding[0]['embedding'])
                print(f"Embedding generated for {auth_face}")
            else:
                print(f"No embedding generated for {auth_face}")
        else:
            print(f"No faces detected in {auth_face}")
    except Exception as e:
        print(f"Error processing authorized face {auth_face}: {e}")

print(f"Number of authorized encodings: {len(authorized_encodings)}")

def is_authorized(face_img: np.ndarray) -> bool:
    """Check if a face is authorized by comparing embeddings."""
    try:
        # Ensure face image is properly formatted
        if face_img.size == 0:
            print("Empty face image")
            return False
            
        # Preprocess face image
        face_img = cv2.resize(face_img, (160, 160))  # Facenet expected size
        
        # Get embedding
        embedding = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
        if embedding and len(embedding) > 0:
            face_encoding = embedding[0]['embedding']
            
            # Compare with authorized faces
            for i, auth_encoding in enumerate(authorized_encodings):
                distance = np.linalg.norm(np.array(face_encoding) - np.array(auth_encoding))
                print(f"Distance to authorized face {i}: {distance:.4f}")
                
                # Adjust threshold - lower value means stricter matching
                if distance < 0.4:  # Changed from 0.6 to 0.4 for stricter matching
                    print(f"Match found! Distance: {distance:.4f}")
                    return True
                
            print("No match found within threshold")
        else:
            print("No embedding generated for detected face")
    except Exception as e:
        print(f"DeepFace error: {e}")
    return False

def trigger_alarm():
    """Play alarm sound with cooldown."""
    global last_alarm_time
    if ALARM_SOUND is None:
        return
    current_time = time.time()
    if current_time - last_alarm_time > ALARM_COOLDOWN:
        try:
            pygame.mixer.music.load(ALARM_SOUND)
            pygame.mixer.music.play()
            time.sleep(5)
            pygame.mixer.music.stop()
            last_alarm_time = current_time
        except Exception as e:
            print(f"Alarm error: {e}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        print(f"Number of faces detected in frame: {len(faces)}")

        for (x, y, w, h) in faces:
            # Add padding to get more of the face
            pad = 20
            y1 = max(y - pad, 0)
            y2 = min(y + h + pad, frame.shape[0])
            x1 = max(x - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            
            face_img = frame[y1:y2, x1:x2].copy()
            # Draw green rectangle for all detected faces initially
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Check authorization
            if is_authorized(face_img):
                label = "Authorized"
                color = (0, 255, 0)  # Green
            else:
                label = "Unauthorized"
                color = (0, 0, 255)  # Red
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)  # Overwrite with red for unauthorized
                trigger_alarm()

            # Add label above face
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display frame
        cv2.imshow("AI Surveillance", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()