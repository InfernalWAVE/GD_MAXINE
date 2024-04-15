 # ****************************************************************
 # * Copyright (c) 2024 Ryan Powell
 # *
 # * This software is released under the MIT License.
 # * See the LICENSE file in the project root for more information.
 # *****************************************************************

extends XRFaceModifier3D

@export var expression_tracker: ExpressionTrack
@export var expression_tracker_name: String
@export var debug_mesh: MeshInstance3D
@export var debug_mesh2: MeshInstance3D

var expression_coefficients: Array
var gaze_angles: Array

var xr_face_tracker: XRFaceTracker = XRFaceTracker.new()

# TODO add BrowInnerUp shape when XRFaceTracker node is updated
const EXPRESSION_MAP: Array[int] = [
	108, 107, 23, 22, 25, 24, 35, 34, 33, 32, 9, 8, 7, 3, 5, 1, 4, 0, 6, 2, 11, 10, 13, 12, 42, 
	21, 38, 40, 39, 79, 78, 75, 74, 124, 132, 63, 62, 83, 82, 127, 131, 120, 119, 81, 80, 134,
	133, 77, 76, 61, 60, 27, 26
]

const GAZE_OFFSET = deg_to_rad(10.0)

func _ready() -> void:
	set_face_tracker(expression_tracker_name)
	
	XRServer.add_face_tracker(expression_tracker_name, xr_face_tracker)

func _process(delta: float) -> void:
	expression_coefficients = expression_tracker.get_expressions()
	
	# update XRFaceTracker blendshapes with expression coefficients
	for i in range(EXPRESSION_MAP.size()):
		xr_face_tracker.set_blend_shape(EXPRESSION_MAP[i], expression_coefficients[i])
	
	# set head mesh (target) transform
	get_node(target).set_quaternion(expression_tracker.get_pose_rotation())
	
	gaze_angles = expression_tracker.get_gaze_angles_vector()
	debug_mesh.set_rotation(Vector3(gaze_angles[0] + GAZE_OFFSET, gaze_angles[1] + PI, 0.0))
	debug_mesh2.set_rotation(Vector3(gaze_angles[0] + GAZE_OFFSET, gaze_angles[1] + PI, 0.0))
	
