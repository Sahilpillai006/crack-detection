import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained crack detection model
model = tf.keras.models.load_model("my_model.h5")

# Define the function to process the video stream and detect cracks
def detect_crack():
    # Open the default camera (0 is usually the inbuilt or first external webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera is accessible
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Continuously capture frames from the camera
    while True:
        ret, frame = cap.read()

        # If frame capture fails, break the loop
        if not ret:
            print("Error: Could not capture frame.")
            break

        # Resize frame to the input size expected by the model (120x120)
        resized_frame = cv2.resize(frame, (120, 120))
        # Normalize pixel values and expand dimensions to match model input shape
        resized_frame = np.expand_dims(resized_frame, axis=0) / 255.0

        # Make a prediction using the model
        prediction = model.predict(resized_frame)

        # Interpret the prediction: < 0.5 = No Crack, >= 0.5 = Crack
        if prediction < 0.5:
            label = "No Crack"
            color = (0, 255, 0)  # Green box for "No Crack"
        else:
            label = "Crack"
            color = (0, 0, 255)  # Red box for "Crack"

        # Draw bounding box if a crack is detected
        if prediction >= 0.5:
            # Draw a rectangle around the suspected crack area (adjust coords as needed)
            cv2.rectangle(frame, (50, 50), (200, 200), color, 2)

        # Display the prediction label on the original frame
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # Show the processed original frame
        cv2.imshow('Crack Detection (Original)', frame)

        # Generate inverted version of the frame (may enhance crack visibility)
        inverted_frame = 255 - frame
        # Show the inverted frame
        cv2.imshow('Crack Detection (Inverted)', inverted_frame)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Start the crack detection process
detect_crack()
