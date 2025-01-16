import cv2
import numpy as np

# Load the image
image = cv2.imread('000506.png')

# Load a pre-trained face detector (Haar Cascade for simplicity)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# For simplicity, assume we are working with the first face detected
for (x, y, w, h) in faces:
    face = image[y:y+h, x:x+w]  # Crop the face region
    
    # You can use a pre-trained model (like dlib) to detect lips landmarks
    # Here, for simplicity, let's assume we have a mask of the lips
    # The lips mask should be a binary mask where the lips are white (255) and other areas are black (0)
    lips_mask = np.zeros_like(face[:, :, 0])  # A blank mask of the same size as the face
    
    # Assume the lips are in a certain region and create a simple rectangle mask for demonstration
    lips_mask[100:150, 50:150] = 255  # Example region for lips (this is just for demo)
    
    # Apply makeup (let's say red lipstick color in BGR format)
    lipstick_color = (0, 0, 255)  # Red in BGR format
    
    # Create a lipstick image (same size as the face) with the desired color
    lipstick_image = np.zeros_like(face)
    lipstick_image[:] = lipstick_color
    
    # Apply bitwise AND between the lips mask and lipstick color
    lips_with_lipstick = cv2.bitwise_and(lipstick_image, lipstick_image, mask=lips_mask)
    
    # Apply the lipstick to the original face using bitwise OR to combine
    face_with_lipstick = cv2.bitwise_or(face, lips_with_lipstick)
    
    # Replace the face region in the original image
    image[y:y+h, x:x+w] = face_with_lipstick

# Display the result
cv2.imshow('Face with Lipstick', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
