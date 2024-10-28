 # ****************************************************************
 # * Copyright (c) 2024 Ryan Powell
 # *
 # * This software is released under the MIT License.
 # * See the LICENSE file in the project root for more information.
 # *****************************************************************

extends Sprite2D

@export var expression_tracker: ExpressionTrack

var gaze_angles: Array
var gaze: Vector2
var expressions: Array
var jaw_coefficient: float
var translation_offset: Vector2

const JAW_OPEN_INDEX: int = 26
const COEFF_THRESHOLD: float = 0.15
const MOVE_SPEED: float = 10.0

func _process(delta: float) -> void:
	gaze_angles = expression_tracker.get_gaze_angles_vector()
	gaze = Vector2(-gaze_angles[1], -gaze_angles[0]).normalized()
	expressions = expression_tracker.get_expressions()
	jaw_coefficient = expressions[JAW_OPEN_INDEX]
	
	if jaw_coefficient > COEFF_THRESHOLD:
		translation_offset = gaze * jaw_coefficient * MOVE_SPEED
		translate(translation_offset)
