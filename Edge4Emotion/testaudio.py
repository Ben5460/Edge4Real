from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import pyaudio
from audio.Recording import Recording
from audio.emotionProcessor import EmotionProcessor
import audio.scikit_network
from statistics import stdev
import numpy as np
from audio.profileManager import *
from os import *


# Hardcoded Variables
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
wave_output_filename = "user_recording.wav"



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
        self.set_ui()
        self.slot_init()
        self.__flag_mode = 0
        self.fps = 0.00
        self.data = {}
        self.memory = {}
        self.joints = []
        self.current = []
        self.previous = []

    def set_ui(self):

        self.__layout_main = QtWidgets.QHBoxLayout()
        self.__layout_fun_button = QtWidgets.QVBoxLayout()
        self.__layout_data_show = QtWidgets.QVBoxLayout()

        self.button_mode_5 = QtWidgets.QPushButton(u'Start recording')

        self.button_mode_5.setMinimumHeight(50)

        self.infoBox = QtWidgets.QTextBrowser(self)
        self.infoBox.setGeometry(QtCore.QRect(10, 380, 200, 100))

        # 信息显示
        self.label_move = QtWidgets.QLabel()
        self.label_move.setFixedSize(200, 100)

        self.__layout_fun_button.addWidget(self.button_mode_5)
        self.__layout_fun_button.addWidget(self.infoBox)
        self.__layout_fun_button.addWidget(self.label_move)

        self.__layout_main.addLayout(self.__layout_fun_button)


        self.setLayout(self.__layout_main)
        self.label_move.raise_()
        self.setWindowTitle(u'human behavior recognize system')

    def slot_init(self):
        self.button_mode_5.clicked.connect(self.recordAudio)

    def recordAudio(self):
        self.emotionalPrediction = StringVar()
        sender = self.sender()
        if sender == self.button_mode_5:
            if self.__flag_mode != 5:
                self.__flag_mode = 5
                self.button_mode_5.setText(u'Stop recording')
                # self.recorder = Recording(wave_output_filename, CHANNELS, RATE, CHUNK)
                # self.recorder.startAudio()
                self.infoBox.setText(u'recording...')
            elif self.__flag_mode == 5:
                self.__flag_mode = 0
                self.button_mode_5.setText(u'Start recording')
                self.infoBox.setText(u'Predicting...')
                # self.recorder.stopAudio()
                # self.audio_metrics = self.processor.collectMetrics()
                # self.user_profile = profileManager(self.userName)
                # self.predicted = audio.scikit_network.compare_new(self.audio_metrics, self.user_profile)
                # result = self.emotionalPrediction.set(self.predicted[0])
                # self.infoBox.setText(u'Predicted Emotion:' + result)
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

    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())











