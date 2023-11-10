# Import Libraries
import cv2

# Print OpenCV version
print(cv2.__version__)

# Define the mpHands Class
class mpHands:
    # Import mediapipe as mp
    import mediapipe as mp

    # Constructor to initialize the Hands object
    def __init__(self, maxHands=2, tol1=.5, tol2=.5):
        # Create Hands object with specified parameters
        self.hands = self.mp.solutions.hands.Hands(False, maxHands, tol1, tol2)

    # Method to extract hand landmarks and types from a frame
    def Marks(self, frame):
        # Lists to store hand landmarks and types
        myHands = []
        handsType = []

        # Convert the frame to RGB
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame using the hands object
        results = self.hands.process(frameRGB)

        # Check if hands are detected in the frame
        if results.multi_hand_landmarks != None:
            # Loop over detected hands and their types
            for hand in results.multi_handedness:
                handType = hand.classification[0].label
                handsType.append(handType)

            # Loop over landmarks in each hand
            for handLandMarks in results.multi_hand_landmarks:
                myHand = []

                # Extract (x, y) coordinates of each landmark and append to myHand
                for landMark in handLandMarks.landmark:
                    myHand.append((int(landMark.x * width), int(landMark.y * height)))

                # Append the list of landmarks for the hand to myHands
                myHands.append(myHand)

        # Return the collected hand landmarks and types
        return myHands, handsType

# Set video parameters
width = 1280
height = 720

# Open video capture with specified parameters
cam = cv2.VideoCapture(4, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Instantiate mpHands object with maxHands set to 2
findHands = mpHands(2)

# Main loop for video capture and hand detection
while True:
    # Read a frame from the video capture
    ignore, frame = cam.read()

    # Resize the frame to specified width and height
    frame = cv2.resize(frame, (width, height))

    # Call Marks method to get hand landmarks and types
    handData, handsType = findHands.Marks(frame)

    # Loop over detected hands and draw circles at specific landmarks
    for hand, handType in zip(handData, handsType):
        # Determine hand color based on hand type
        handColor = (255, 0, 0) if handType == 'Right' else (0, 0, 255)

        # Draw circles at specific landmarks
        for ind in [0, 5, 6, 7, 8]:
            cv2.circle(frame, hand[ind], 15, handColor, 5)

    # Display the frame
    cv2.imshow('my WEBcam', frame)
    cv2.moveWindow('my WEBcam', 0, 0)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the video capture
cam.release()
