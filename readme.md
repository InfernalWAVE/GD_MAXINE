# GD_MAXINE_test
test for distributing a release of the NVIDIA RTX face tracking extension for Godot 4.3+

# INSTALL INSTRUCTIONS:
- download and install NVIDIA Broadcast: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/
- download and install NVIDIA AR SDK: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
- download repo
- place both .dll files and the .gdextension file in the bin folder of your project
- reload project

# USE INSTRUCTIONS:
- add ExpressionTrack node to a scene.
- run game

when the game runs the extension will create an OpenCV VideoCapture object, and begin capturing frames and running the facial expression feature on them.

the captured expression information is exposed as properties on the ExpressionTrack node. You can use functions like:
- get_expressions()
- get_landmarks()
- get_pose_transform()
- etc

this is a mess right now, but I plan to open source soon. feel free to mess around and enjoy though!
