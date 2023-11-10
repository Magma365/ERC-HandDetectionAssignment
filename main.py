import cv2
import mediapipe as mp

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    results = hands.process(rgb_frame)

    # Check if hand landmarks are available
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for each hand
            for lm_id, lm in enumerate(hand_landmarks.landmark):
                # Access landmark coordinates (x, y, z)
                x, y, z = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0]), lm.z

                # Draw landmarks on the frame
                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            # Determine hand orientation based on specific landmarks
            if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x:
                hand_orientation = "Left Hand"
            else:
                hand_orientation = "Right Hand"

            # Display hand orientation on the frame
            cv2.putText(frame, hand_orientation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Hand Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
