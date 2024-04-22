# GD_MAXINE
implementation of NVIDIA AR SDK facial expression tracking for Godot 4.3.

# NOTE
- idk what i'm doing really but i'm trying my best. if things are sloppy, forgive me.
- this is all done and setup on windows. the dlls in the demo project bin are for windows.
- extension is still named "gdexample", the intended named for the extension is "GD_MAXINE"
- the face mesh in the demo needs blend shapes for: BrowInnerUpRight and BrowInnerUpLeft. it is setup for BrowInnerUp, but that blendshape is missing from the XRFaceTracker at time of release. NVIDIA's SDK reports each blendshape separately though, so it should work with a properly setup mesh.
- still need to work out position posing for the head mesh in the demo. will need to define an origin and scale for the pose translation.
- the opencv and glm libraries i just took from the samples on the nvidia repo. these versions may be outdated at this point, idk. but i knew they would be compatible with the sdk.

# BUILD INSTRUCTIONS:
- download and install NVIDIA Broadcast: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/
- download and install NVIDIA AR SDK: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
  - NOTE: the extension relies on the environment variable set when installing the AR SDK
- clone repo
- open in VS Code
- build with scons

# RELEASE INSTALL INSTRUCTIONS:
- download and install NVIDIA Broadcast: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/
- download and install NVIDIA AR SDK: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
  - NOTE: the extension relies on the environment variable set when installing the AR SDK
- download release
- place both .dll files and the .gdextension file in the bin folder of your project
- reload project

# USE INSTRUCTIONS:
- add ExpressionTrack node to a scene.
- create a script with a reference to the ExpressionTrack node, add logic for using the captured expression info.

The following functions are available:
- get_expression_count()
  - returns int, number of expression coefficients
- get_expressions()
  - returns Array, normalized expression coefficients as float
- get_landmark_count()
  - returns int, number of facial landmarks
- get_landmarks()
  - returns Array, facial landmark positions as Vector2
- get_pose_transform()
  - returns Transform3D, head pose
- get_pose_rotation()
  - returns Quaternion, head pose rotation
- get_pose_translation(),
  - returns Vector3, head pose position
- get_bounding_boxes(),
  - returns Array, bounding boxes for detected face as Dictionary
    - Dictionary members are: x, y, width, height, as float

these are also exposed as properties on the node for convenience. there technically are setters, but they do nothing.

NOTE: _ready() and _process() are overridden on the ExpressionTrack node. attaching a script that uses these functions will break the tracking. I reccomend not attaching any scripts to the ExpressionTrack node unless you understand what the extension is doinig.

NOTE: the extension automatically uses your default system camera for now. adding a camera select option is feasible in the future.

# DEMO INFO:
when the game runs the extension will create an OpenCV VideoCapture object, and begin capturing frames and running the facial expression feature on them.

the captured expression information is exposed as properties on the ExpressionTrack node.

the expression coefficients are then mapped to blendshapes in an XRFaceTracker. this XRFaceTracker is used by an XRFaceModifier to set the blendshapes on the mesh

# CREDITS:
This software is released under the MIT License. See LICENSE file for information on permissions.

This software contains source code provided by NVIDIA Corporation. See LICENSE.NVIDIA.txt file for information on permissions.

This software contains source code from opencv. See LICENSE.OPENCV.txt file for information on permissions.

This software contains source code from glm. See LICENSE.GLM.txt file for information on permissions.

This software contains source code from the Godot engine. See LICENSE.GODOT.txt for information on permissions.

Powered by NVIDIA Broadcast.

Created By: Ryan Powell, 2024.
