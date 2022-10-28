# 1 Import lib
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#from LSTM_Neural_Network_Train import model
from data_preprocessing import X_test, y_test
from make_dir import actions
from keras.models import load_model
print("Loading Modle")
model = load_model('action.h5')

res = model.predict(X_test)

actions[np.argmax(res[4])]

actions[np.argmax(y_test[4])]

## Saving Your Model

# model.save('action.h5')
# print("Your model has been saved !!!!!!!!!! :D,:),:},:]")

# del model

# model.load_weights('action.h5')

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()
multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

print(accuracy_score(ytrue, yhat))