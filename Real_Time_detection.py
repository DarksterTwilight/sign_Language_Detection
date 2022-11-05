# 1 Import lib
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from keras.models import load_model
import mediapipe as mp
# 2 Keypoints using MP Holistic
mp_holistic = mp.solutions.holistic # Holistic model



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])



# New detection variable
sequence = [] # collect 30 frame
sentence = []
threshold = 0.4
my_actions = np.array(['All_clear', 'return', 'call_to_operator'])

my_model = load_model('action2.h5')

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        # print(results)

        # Draw landmarks
        # draw_styled_landmarks(image, results)

        # Prediction Logic
        # print('Start')
        keypoints = extract_keypoints(results)
        sequence.insert(0, keypoints)
        sequence = sequence[:30]
        # print('Sequance')

        if len(sequence) == 30:
            res = my_model.predict(np.expand_dims(sequence, axis=0))[0]
            # reshaping sequence -> (30,1662), expected ->(1,30,1662)
            # print(my_actions[np.argmax(res)])
            # print(res[np.argmax(res)])

        # 3 Viz Logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if (my_actions[np.argmax(res)] != sentence[-1]) and (my_actions[np.argmax(res)] != my_actions[0]):
                        sentence.append(my_actions[np.argmax(res)])
                else:
                    sentence.append(my_actions[np.argmax(res)])
            if len(sentence) > 5:
                sentence = sentence[-5:]
            # Grab last 5 word
        _, width, _ = image.shape
        cv2.rectangle(image, (0,0), (width,40), (245,117,16), -1)
        cv2.putText(image,' '.join(sentence), (3,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA,False)
        print(' '.join(sentence))
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


