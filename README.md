# 3DPoseEstimationForVR
Introduction 


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
