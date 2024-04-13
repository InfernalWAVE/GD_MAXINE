# GD_MAXINE_test
implementation of NVIDIA AR SDK facial expression tracking for Godot 4.3+

# NOTE
- idk what i'm doing really but i'm trying my best. if things are sloppy, forgive me.
- extension is still named "gdexample", the intended named for the extension is "GD_MAXINE"
- the face mesh in the demo needs blend shapes for: BrowInnerUpRight and BrowInnerUpLeft. it is setup for BrowInnerUp, but that blendshape is missing from the XRFaceTracker at time of release. NVIDIA's SDK reports each blendshape separately though, so it should work with a properly setup mesh.
- still need to work out position posing for the head mesh in the demo. will need to define an origin and scale for the pose translation.
- the opencv and glm libraries i just took from the samples on the nvidia repo. these versions may be outdated at this point, idk. but i knew they would be compatible with the sdk.

# BUILD INSTRUCTIONS:
- clone repo
- open in VS Code
- build with scons

# INSTALL INSTRUCTIONS:
- download and install NVIDIA Broadcast: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/
- download and install NVIDIA AR SDK: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
- download release
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
