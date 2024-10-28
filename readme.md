# GD_MAXINE
implementation of NVIDIA AR SDK features for Godot 4.3+

# NOTE
- idk what i'm doing really but i'm trying my best. if things are sloppy, forgive me.
- this is all made and tested on Windows. there may be a way to use this on a Linux system, but I am unsure.
- the demo app only implements features from the facial expression and gaze trackers. body tracking is enabled, but not being used by the demo (for now)

# BUILD INSTRUCTIONS:
- download and install NVIDIA Broadcast: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-app/
- download and install NVIDIA AR SDK: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/
  - NOTE: the extension relies on the environment variable set when installing the AR SDK
- clone repo
- open in VS Code
- build with scons (this relies on the opencv_world346.lib being in src/opencv to work)
- ensure opencv_world346.dll is in the project bin (it should be by default)

# RELEASE INSTALL INSTRUCTIONS (release is likely behind src):
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
- get_body_bounding_boxes()
  - returns Array of bounding boxes for detected body as Dictionary
    - Dictionary members are: x, y, width, height, as float
- get_body_bounding box_confidence()
  - returns Array of float for confidence in body tracking bounding boxes
- get_gaze_angles_vector()
  - returns Array of float for pitch and yaw rotations of the eyes
- get_gaze_direction()
  - returns Vector3 for vector from eyes to look location in 3D
- get_joint_angles()
  - returns Array of Quaternion for joint rotations
- get_keypoints()
  - returns Array of Vector2 for 2D joint positions
- get_keypoints_3D()
  - returns Array of Vector3 for 3D joint positions
- get_keypoints_confidence()
  - returns array of float for confidence in joint keypoint identification

these are also exposed as properties on the node for convenience. there technically are setters, but they do nothing.

NOTE: _ready() and _process() are overridden on the ExpressionTrack node. attaching a script that uses these functions will break the tracking. I reccomend not attaching any scripts to the ExpressionTrack node unless you understand what the extension is doing.

NOTE: use the "camera device id" property on the ExpressionTrack node to select which camera device you want use

# DEMO INFO:
when the game runs the extension will create an OpenCV VideoCapture object, and begin capturing frames and running the facial expression feature on them.

the captured expression information is exposed as properties on the ExpressionTrack node.

the expression coefficients are then mapped to blendshapes in an XRFaceTracker. this XRFaceTracker is used by an XRFaceModifier to set the blendshapes on the mesh

# CREDITS:
This software is released under the MIT License. See LICENSE file for information on permissions.

This software contains source code provided by NVIDIA Corporation. See LICENSE.NVIDIA.txt file for information on permissions.

This software contains source code from opencv. See LICENSE.OPENCV.txt file for information on permissions.

This software contains source code from glm. See LICENSE.GLM.txt file for information on permissions.

This software contains source code from the Godot engine. See LICENSE.GODOT.md for information on permissions.

Powered by NVIDIA Broadcast.

Created By: Ryan Powell, 2024.
