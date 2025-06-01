import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model("my_model.h5")

# Define the function to process the video stream
def detect_crack():
    # Open the camera
    cap = cv2.VideoCapture(0)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Loop to capture frames from the camera
    while True:
        ret, frame = cap.read()

        # Check if frame is captured successfully
        if not ret:
            print("Error: Could not capture frame.")
            break

        # Preprocess the frame for model input
        resized_frame = cv2.resize(frame, (120, 120))
        resized_frame = np.expand_dims(resized_frame, axis=0) / 255.0

        # Use the model to predict
        prediction = model.predict(resized_frame)

        # Get the predicted class label
        if prediction < 0.5:
            label = "No Crack"
            color = (0, 255, 0)  # Green color for no crack
        else:
            label = "Crack"
            color = (0, 0, 255)  # Red color for crack
        
        # Draw bounding box around the crack
        if prediction >= 0.5:
            cv2.rectangle(frame, (50, 50), (200, 200), color, 2)  # Change coordinates as needed

        # Display the original frame with the predicted label and bounding box
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Crack Detection (Original)', frame)

        # Invert the frame
        inverted_frame = 255 - frame  # Invert pixel intensities

        # Display the inverted frame
        cv2.imshow('Crack Detection (Inverted)', inverted_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start crack detection using camera
detect_crack()
