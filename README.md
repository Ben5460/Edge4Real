Edge4Emotion

Introduction 

Edge4real new Progress

We added two new modules related to emotion recognition to the original human behavior platform. One is the facial expression recognition module, and the other is the audio emotion recognition module

In the facial expression recognition part. We use a ROI (region of interest) region to quickly identify the face position. For CNN networks to extract facial expression feature more accurately, we grayscale and resize it to a fixed 28x28 pixel image in the ROI region, and then input the image date to a CNN network for classification. Finally, we draw the label of facial expression next the face ROI region in the canvas.

In audio emotion recognition part. Due to the dependence of audio information on time sequence, it is difficult to acquire and analyze audio data in real time. For predicting the emotion about user audio, we adopt an approach of recording first and then predicting. First, we record user audio and save it as a wav format file. Then read the audio file and extract the features of the audio including MFCC, pitch, sound pressure level, gaps between consecutive words. We form the extracted audio features into an audio metric to facilitate audio data analysis. Finally, we access the user profile which save large amount of audio data and corresponding emotion labels. Then acquire data pertaining to the new recording and compares it against the data retrieved from the user profile using the k-nearest neighbor algorithm to find the closest match for the new data. The forecast results are displayed on the system interface.

Dependencies: python3.6, pyaudio,pyAudioAnalysis, scipy, wave, pydub, python_speech_features, numpy, statistics, PyQt5, utils, keras, tensorflow, cv2


Edge4Real

Introduction 

Recognition of human behaviours including body motions and facial expressions plays a significant role in human-centric software engineering. However, powerful machines are required to analysis and to compute human behaviour recognition through video analytics. Edge4Real which can be easily deployed in an edge computing environment with commodity machines. Edge4Real can generate the 3D human pose from webcam/camera to VR device. 

Setup steps
1. Clone this project to your computer
2. Download two Unity projects from the link provided 
3. Base on the dependencies setup and build two project(Set the IP address to the server's IP)
4. Install the apk file from step 3 to Oculus Quest 
5. Turn on the webcam
6. Run the server by using command prompt(or SDE) access the TCP server folder and type command line : dotnet run (Change IP to this        machine)
7. Run the application from Unity
8. Run the Oculus App



Dependencies:

LAN Network
TCP server: Windows 7 or above, .NET core 3.1
Unity Barrucuda:  Unity 2019.2.5f1 or above
Unity for Oculus:  Unity 2019.2.5f1 or above, Android SDK


Download:

Unity Barrucuda for generating 3D Pose from webcam https://deakin365-my.sharepoint.com/:u:/g/personal/chengye_deakin_edu_au/EeDp-Hwo9f5Cr_KTUlGdtrIBN-8XqvaF08hF7u1Q11BHGQ?e=J4MLZV

Oculus App for regenerating 3D Pose from Barrucuda https://deakin365-my.sharepoint.com/:u:/g/personal/chengye_deakin_edu_au/Efk5Fo2rLNxDk5ir0QshmDABxwh728VhrZcj3zZ2HVea5Q?e=fgjZrP

Reference:
1. Barracuda 3D Pose Estimtation by yukihiko https://github.com/yukihiko/ThreeDPoseUnitySample
2. TCP Server and Unity Server and Client by dilmerv https://github.com/dilmerv/TCPServerAndClient

Demo video：https://youtu.be/dH0oWnWk924
