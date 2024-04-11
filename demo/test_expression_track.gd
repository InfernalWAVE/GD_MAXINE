extends ExpressionTrack

@export var capture_timer: Timer

func _on_timer_timeout():
	print("\nlandmark count: ")
	print(get_landmark_count())
	
	print("\ncaptured landmarks: ")
	print(get_landmarks())
	
	print("\nlandmark confidence: ")
	print(get_landmark_confidence())
	
	print("\nexpression count: ")
	print(get_expression_count())
	
	print("\ncaptured expressions: ")
	print(get_expressions())
	
	print("\npose: ")
	print("rotation: %s" %str(get_pose_rotation()))
	print("translation: %s" %str(get_pose_translation()))
	print("transform: %s" %str(get_pose_transform()))
	
	print("\nbounding boxes: ")
	print(get_bounding_boxes())
