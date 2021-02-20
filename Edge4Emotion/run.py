# -*- coding: UTF-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets
from pose.estimator import TfPoseEstimator
from pose.networks import get_graph_path
from utils.sort import Sort
from utils.actions import actionPredictor
from utils.joint_preprocess import *
from keras.preprocessing.image import img_to_array
import sys
import cv2
import time
import settings
from keras.models import load_model
import imutils

import pyaudio
from audio.Recording import Recording
from audio.emotionProcessor import EmotionProcessor
import audio.scikit_network
from statistics import stdev
import numpy as np
import numpy
from audio.profileManager import *
from os import *


#防止显存不足
import tensorflow as tf

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

config = tf.compat.v1.ConfigProto(allow_soft_placement=True)

config.gpu_options.per_process_gpu_memory_fraction = 0.9
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]


poseEstimator = None

# Hardcoded Variables
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
wave_output_filename = "user_recording.wav"

def load_model():
    global poseEstimator
    poseEstimator = TfPoseEstimator(
        get_graph_path('mobilenet_thin'), target_size=(432, 368))

class Variable:
    def __init__(self, value=None):
        if value is not None:
            self.initialize(value)
    def set(self, value):
        return value
    initialize = set

class StringVar(Variable):
    def __init__(self, value=None):
        Variable.__init__(self,value)

class Ui_MainWindow(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        self.tracker = Sort(settings.sort_max_age, settings.sort_min_hit)
        self.timer_camera = QtCore.QTimer()
        self.cap = cv2.VideoCapture()
        self.CAM_NUM = 0
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.audio_flag = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []
        self.processor =EmotionProcessor(wave_output_filename)
        self.userName = "generic"
    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_open_camera = QtWidgets.QPushButton(u'Camera OFF')

        self.button_mode_1 = QtWidgets.QPushButton(u'Pose Estimation OFF')
        self.button_mode_2 = QtWidgets.QPushButton(u'Multiplayer tracking OFF')
        self.button_mode_3 = QtWidgets.QPushButton(u'Behavior recognize OFF')
        self.button_mode_4 = QtWidgets.QPushButton(u'Facial expression OFF')
        self.button_mode_5 = QtWidgets.QPushButton(u'Start recording')

        self.button_open_camera.setMinimumHeight(50)
        self.button_mode_1.setMinimumHeight(50)
        self.button_mode_2.setMinimumHeight(50)
        self.button_mode_3.setMinimumHeight(50)
        self.button_mode_4.setMinimumHeight(50)
        self.button_mode_5.setMinimumHeight(50)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 380, 200, 100))

        # 信息显示
        self.label_show_camera = QtWidgets.QLabel()
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 100)

        self.label_show_camera.setFixedSize(settings.winWidth + 1, settings.winHeight + 1)
        self.label_show_camera.setAutoFillBackground(True)

        self.__layout_fun_button.addWidget(self.button_open_camera)
        self.__layout_fun_button.addWidget(self.button_mode_1)
        self.__layout_fun_button.addWidget(self.button_mode_2)
        self.__layout_fun_button.addWidget(self.button_mode_3)
        self.__layout_fun_button.addWidget(self.button_mode_4)
        self.__layout_fun_button.addWidget(self.button_mode_5)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)
        self.__layout_main.addWidget(self.label_show_camera)

        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'human behavior recognize system')

    def slot_init(self):
        self.button_open_camera.clicked.connect(self.button_event)
        self.timer_camera.timeout.connect(self.show_camera)

        self.button_mode_1.clicked.connect(self.button_event)
        self.button_mode_2.clicked.connect(self.button_event)
        self.button_mode_3.clicked.connect(self.button_event)
        self.button_mode_4.clicked.connect(self.button_event)
        self.button_mode_5.clicked.connect(self.recordAudio)

    def button_event(self):
        sender = self.sender()
        if sender == self.button_mode_1 and self.timer_camera.isActive():
            if self.__flag_mode != 1:
                self.__flag_mode = 1
                self.button_mode_1.setText(u'Pose Estimation ON')
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.button_mode_3.setText(u'behavior recognize OFF')
                self.infoBox.setText(u'Currently in the human pose estimation mode')
            else:
                self.__flag_mode = 0
                self.button_mode_1.setText(u'Pose Estimation OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_2 and self.timer_camera.isActive():
            if self.__flag_mode != 2:
                self.__flag_mode = 2
                self.button_mode_1.setText(u'Pose Estimation OFF')
                self.button_mode_2.setText(u'Multiplayer Tracking ON')
                self.button_mode_3.setText(u'Behavior Recognize OFF')
                self.button_mode_4.setText(u'facial expression OFF')
                self.infoBox.setText(u'Currently in multiplayer tracking mode')
            else:
                self.__flag_mode = 0
                self.button_mode_2.setText(u'Multiplayer tracking OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_3 and self.timer_camera.isActive():
            if self.__flag_mode != 3:
                self.__flag_mode = 3
                self.button_mode_1.setText(u'Pose Estimation OFF')
                self.button_mode_2.setText(u'Multiplayer Tracking OFF')
                self.button_mode_3.setText(u'Behavior Recognize ON')
                self.button_mode_4.setText(u'facial expression OFF')
                self.infoBox.setText(u'Current in human behavior recognition mode')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'Behavior Recognize OFF')
                self.infoBox.setText(u'Camera is on')
        elif sender == self.button_mode_4 and self.timer_camera.isActive():
            if self.__flag_mode != 4:
                self.__flag_mode = 4
                self.button_mode_1.setText(u'Pose Estimation OFF')
                self.button_mode_2.setText(u'Multiplayer Tracking OFF')
                self.button_mode_3.setText(u'Behavior Recognize OFF')
                self.button_mode_4.setText(u'facial expression ON')
                self.infoBox.setText(u'Currently in facial expression mode')
            else:
                self.__flag_mode = 0
                self.button_mode_3.setText(u'Behavior Recognize OFF')
                self.infoBox.setText(u'Camera is on')
        else:
            self.__flag_mode = 0
            self.button_mode_1.setText(u'Pose Estimation OFF')
            self.button_mode_2.setText(u'Multiplayer Tracking OFF')
            self.button_mode_3.setText(u'Behavior Recognize OFF')
            self.button_mode_4.setText(u'facial expression OFF')
            if self.timer_camera.isActive() == False:
                flag = self.cap.open(self.CAM_NUM)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.winWidth)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.winHeight)
                if flag == False:
                    msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"Please check if the camera and computer are connected correctly",
                                                        buttons=QtWidgets.QMessageBox.Ok,
                                                        defaultButton=QtWidgets.QMessageBox.Ok)
                else:
                    self.timer_camera.start(1)
                    self.button_open_camera.setText(u'Camera ON')
                    self.infoBox.setText(u'Camera is on')
            else:
                self.timer_camera.stop()
                self.cap.release()
                self.label_show_camera.clear()
                self.button_open_camera.setText(u'Camera OFF')
                self.infoBox.setText(u'Camera is off')

    def show_camera(self):
        start = time.time()
        ret, frame = self.cap.read()
        show = cv2.resize(frame, (settings.winWidth, settings.winHeight))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        preds = []
        if ret:
            if self.__flag_mode == 1:
                # self.infoBox.setText(u'Currently in the human pose estimation mode')
                humans = poseEstimator.inference(show)
                show = TfPoseEstimator.draw_humans(show, humans, imgcopy=False)

            elif self.__flag_mode == 2:
                # self.infoBox.setText(u'Currently in multiplayer tracking mode')
                humans = poseEstimator.inference(show)
                show, joints, bboxes, xcenter, sk = TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)

                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            elif self.__flag_mode == 3:
                # self.infoBox.setText(u'Current in human behavior recognition mode')
                humans = poseEstimator.inference(show)
                ori = np.copy(show)
                show, joints, bboxes, xcenter, sk= TfPoseEstimator.get_skeleton(show, humans, imgcopy=False)
                height = show.shape[0]
                width = show.shape[1]
                if bboxes:
                    result = np.array(bboxes)
                    det = result[:, 0:5]
                    det[:, 0] = det[:, 0] * width
                    det[:, 1] = det[:, 1] * height
                    det[:, 2] = det[:, 2] * width
                    det[:, 3] = det[:, 3] * height
                    trackers = self.tracker.update(det)
                    self.current = [i[-1] for i in trackers]

                    if len(self.previous) > 0:
                        for item in self.previous:
                            if item not in self.current and item in self.data:
                                del self.data[item]
                            if item not in self.current and item in self.memory:
                                del self.memory[item]

                    self.previous = self.current
                    for d in trackers:
                        xmin = int(d[0])
                        ymin = int(d[1])
                        xmax = int(d[2])
                        ymax = int(d[3])
                        label = int(d[4])
                        try:
                            j = np.argmin(np.array([abs(i - (xmax + xmin) / 2.) for i in xcenter]))
                        except:
                            j = 0
                        if joint_filter(joints[j]):
                            joints[j] = joint_completion(joint_completion(joints[j]))
                            if label not in self.data:
                                self.data[label] = [joints[j]]
                                self.memory[label] = 0
                            else:
                                self.data[label].append(joints[j])

                            if len(self.data[label]) == settings.L:
                                pred = actionPredictor().move_status(self.data[label])
                                if pred == 0:
                                    pred = self.memory[label]
                                else:
                                    self.memory[label] = pred
                                self.data[label].pop(0)

                                location = self.data[label][-1][1]
                                if location[0] <= 30:
                                    location = (51, location[1])
                                if location[1] <= 10:
                                    location = (location[0], 31)

                                cv2.putText(show, settings.move_status[pred], (location[0] - 30, location[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (0, 255, 0), 2)

                        cv2.rectangle(show, (xmin, ymin), (xmax, ymax),
                                      (int(settings.c[label % 32, 0]),
                                       int(settings.c[label % 32, 1]),
                                       int(settings.c[label % 32, 2])), 4)

            elif self.__flag_mode == 4:
                # self.infoBox.setText(u'Currently in facial expression mode')
                # reading the frame
                # frame = imutils.resize(frame, width=300)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)

                canvas = numpy.zeros((250, 300, 3), dtype="uint8")
                show = frame.copy()
                if len(faces) > 0:
                    faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
                    (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
                    # the ROI for classification via the CNN
                    roi = gray[fY:fY + fH, fX:fX + fW]
                    roi = cv2.resize(roi, (64, 64))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = numpy.expand_dims(roi, axis=0)

                    preds = emotion_classifier.predict(roi)[0]

                    label = EMOTIONS[preds.argmax()]
                else:
                    pass

                for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)

                    # draw the label + probability bar on the canvas
                    # emoji_face = feelings_faces[np.argmax(preds)]

                    w = int(prob * 300)

                    cv2.putText(show, label, (fX, fY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                    cv2.rectangle(show, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            if (time.time() - start) > 0:
                self.fps = 1 / (time.time() - start)
            else:
                self.fps = 1
            cv2.putText(show, 'FPS: %.2f' % self.fps, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def recordAudio(self):
        self.emotionalPrediction = StringVar()
        sender = self.sender()
        if sender == self.button_mode_5:
            if self.audio_flag == 0:
                self.audio_flag = 1
                self.button_mode_5.setText(u'Stop recording')
                self.recorder = Recording(wave_output_filename, CHANNELS, RATE, CHUNK)
                self.recorder.startAudio()
                self.infoBox.setText(u'recording...')
            elif self.audio_flag == 1:
                self.audio_flag = 2
                self.button_mode_5.setText(u'Audio Predict')
                self.infoBox.setText(u'done recording')
                self.recorder.stopAudio()
            elif self.audio_flag == 2:
                self.audio_flag = 3
                self.audio_metrics = self.processor.collectMetrics()
                self.user_profile = profileManager(self.userName)
                self.user_profile.accessProfile()
                self.predicted = audio.scikit_network.compare_new(self.audio_metrics, self.user_profile)
                result = self.emotionalPrediction.set(self.predicted[0])
                self.infoBox.setText(u'Predicted Emotion:' + result)
            elif self.audio_flag == 3:
                self.audio_flag = 0
                self.button_mode_5.setText(u'Start recording')
                self.infoBox.setText(u'')

                # question = ("Was predicted emotion " + self.predicted[0] + " correct?")
                # reply = QtWidgets.QMessageBox.question(self, 'Emotion Prediction Assessment', question, QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                # if reply == True:
                #     self.user_profile.addtoProfile(self.audio_metrics, self.predicted[0])
                # else:
                #     emotions = ("Normal", "Excited", "Angry", "Nervous")
                #     text, ok = QtWidgets.QInputDialog.getItem(self, 'Wrong Emotion Correction', 'Enter your Emotion:',emotions)
                #     if ok:
                #         self.user_profile.addtoProfile(self.audio_metrics, text)


    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"close", u"if close！")

        msg.addButton(ok, QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'yes')
        cancel.setText(u'no')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            event.accept()
            print("System exited.")


if __name__ == '__main__':
    load_model()
    print("Load all models done!")
    print("The system starts ro run.")
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())


