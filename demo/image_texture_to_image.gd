extends Node2D

@export var image_texture: ImageTexture

func _ready():
	image_texture.get_image().save_jpg("res://exports/capture_1.jpg")
