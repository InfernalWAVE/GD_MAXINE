extends Node3D

var window_ID: int = -1
var window_mode: DisplayServer.WindowFlags = DisplayServer.WINDOW_FLAG_BORDERLESS

func _ready() -> void:
	window_ID = DisplayServer.get_window_list()[0]

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey:
		if event.is_action_pressed("toggle_window"):
			DisplayServer.window_set_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, !DisplayServer.window_get_flag(DisplayServer.WINDOW_FLAG_BORDERLESS, window_ID), window_ID)
			get_viewport().set_input_as_handled()
