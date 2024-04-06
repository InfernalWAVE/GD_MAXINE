extends Node3D

@export var face_tracker: FaceTrack
@export var texture_capture_path: String
@export var texture_export_path: String
@export var mesh_capture_path: String

var captured_texture: ImageTexture
var captured_landmarks: Array
var captured_mesh: ArrayMesh

func _on_timer_timeout():
	captured_landmarks = face_tracker.get_facial_landmarks()
	print(captured_landmarks)
	
	captured_texture = face_tracker.get_captured_frame_texture()
	if captured_texture != null:
		ResourceSaver.save(captured_texture,texture_capture_path)
		captured_texture.get_image().save_png(texture_export_path)
	
	captured_mesh = face_tracker.get_captured_mesh()
	if captured_mesh != null:
		print(captured_mesh)
		ResourceSaver.save(captured_mesh, mesh_capture_path)
	
