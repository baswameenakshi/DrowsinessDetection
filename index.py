import cv2
import dlib
import os
import pywhatkit as kit
from scipy.spatial import distance
from imutils import face_utils
import tempfile
import streamlit as st
# from playsound import playsound
import pygame

def play_alarm(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
def eye_aspect_ratio(eye):
    try:
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    except Exception as e:
        print(f"Error calculating EAR: {e}")
        return 0

# Constants
THRESH = 0.25  # EAR threshold
FRAME_CHECK = 20  # Consecutive frames for drowsiness
flag = 0  # Initialize flag globally

# Function to send WhatsApp alert message
def send_alert_message(phone_number, message):
    """Send alert message via WhatsApp."""
    try:
        # Sending WhatsApp message instantly
        kit.sendwhatmsg_instantly(phone_number, message)
        st.write("Alert message sent via WhatsApp!")
    except Exception as e:
        st.error(f"Failed to send WhatsApp message: {e}")

# Streamlit app
def main():
    st.title("Drowsiness Detection Website")

    # Input recipient phone number 
    recipient_number = st.text_input("Enter recipient phone number for alerts (include country code):", "")
    if not recipient_number:
        st.warning("Please enter a valid phone number to proceed.")
        return

    st.success("Phone number accepted. You can now proceed.")

    # Initialize dlib models after phone number is entered
    detector = dlib.get_frontal_face_detector()
    predictor_path = "models/shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        st.error("Model file not found. Please upload shape_predictor_68_face_landmarks.dat.")
        return

    predictor = dlib.shape_predictor(predictor_path)

    # Video source selection
    st.write("Upload a video or use your webcam to detect drowsiness in real-time.")
    option = st.radio("Select Video Source:", ("Upload Video", "Use Webcam"))

    if option == "Upload Video":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

        if video_file:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_file.read())
            process_video(temp_file.name, recipient_number, detector, predictor)

    elif option == "Use Webcam":
        st.write("Webcam functionality is now integrated with live feed.")

        run = st.checkbox('Run Webcam')
        FRAME_WINDOW = st.image([])
        camera = cv2.VideoCapture(0)

        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        global flag  # Declare flag as global
        while run:
            ret, frame = camera.read()
            if not ret:
                st.error("Error: Could not access webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detector(gray, 0)

            for subject in subjects:
                shape = predictor(gray, subject)
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw contours around eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # Check EAR threshold
                if ear < THRESH:
                    flag += 1
                else:
                    flag = 0

                if flag >= FRAME_CHECK:
                    # cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    st.warning("Drowsiness detected!.")
                    # os.system('buzz.mp3')
                    play_alarm(r'C:\Users\MEENAKSHI\Downloads\alarm.wav')
                    send_alert_message(recipient_number, "Alert! Drowsiness detected.")
                    camera.release()
                    return
                # else:
                    # cv2.putText(frame, " ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        else:
            st.write('Stopped')
            camera.release()

# Video processing function
def process_video(video_path, phone_number, detector, predictor):
    global flag  # Declare flag as global
    cap = cv2.VideoCapture(video_path)
    st_frame = st.empty()

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detector(gray, 0)

        for subject in subjects:
            shape = predictor(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Draw contours around eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # Check EAR threshold
            if ear < THRESH:
                flag += 1
            else:
                flag = 0

            if flag >= FRAME_CHECK:
                # cv2.putText(frame, "Drowsy", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                st.warning("Drowsiness detected!")
                # os.system('buzz.mp3')  # Ensure buzz.mp3 exists in the same directory
                play_alarm(r'C:\Users\MEENAKSHI\Downloads\alarm.wav')
                send_alert_message(phone_number, "Alert! Drowsiness detected.")
                cap.release()
                return
            # else:
                # cv2.putText(frame, "", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame in Streamlit
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()

if __name__ == "__main__":
    main()
