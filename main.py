import requests
import cv2
import numpy as np
import face_recognition
import time
from PIL import Image
from keras.models import load_model
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x
#------------------------------------------------------------------------------------------
emotion_model_path = 'emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5'
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
emotion_classifier = load_model(emotion_model_path, compile=False)
emotion_target_size = emotion_classifier.input_shape[1:3]
emotion_window = []


#-----------------------------------------------------------------------------------------------

web = "http://172.16.1.213:5435/shot.jpg"
start_time = time.time()
counter = 0

while True:
    emoticon = []
    page = requests.get(web)
    imgN = np.array(bytearray(page.content), dtype = np.uint8)
    frame = cv2.imdecode(imgN, -1)
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow('Frame', small_frame)
    rgb_small_frame = small_frame[:, :, ::-1]
    gray_image = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    # Processing frames
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        # You can access the actual face itself like this:
        face_image = gray_image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.show()
        #--------------------------------------------------------------------------------------------
        try:
            face_image = cv2.resize(face_image, (emotion_target_size))
        except:
            continue
        face_image = preprocess_input(face_image, True)
        face_image = np.expand_dims(face_image, 0)
        face_image = np.expand_dims(face_image, -1)
        emotion_prediction = emotion_classifier.predict(face_image)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = labels[emotion_label_arg]
        emoticon.append(emotion_text)
    print(emoticon)
    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()