import cv2
import tensorflow as tf
from keras.models import model_from_json
import numpy as np

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("emotiondetector.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    try:
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            face_roi = cv2.resize(face_roi, (48, 48))
            img = extract_features(face_roi)
            
            pred = model.predict(img)
            
            prediction_label = labels[pred.argmax()]
            
            cv2.putText(im, prediction_label, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 2)
        
        cv2.imshow("Emotion Detector", im)
        
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    
    except cv2.error:
        pass

webcam.release()
cv2.destroyAllWindows()
