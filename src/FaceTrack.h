#ifndef FACETRACK_H
#define FACETRACK_H

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/array_mesh.hpp>

#include <opencv2/opencv.hpp>
#include "FaceEngine.h"
#include "BodyEngine.h"

namespace godot {

class FaceTrack : public Node {
	GDCLASS(FaceTrack, Node)

private:
    bool should_run;
    cv::VideoCapture cap;
    cv::Mat frame;
    FaceEngine face_ar_engine;
    BodyEngine body_ar_engine;
    static const char windowTitle[];
    FaceEngine::Err nvErr;
    NvAR_Rect output_bbox;
    std::string modelPath;
    Array captured_facial_landmarks;
    Ref<ImageTexture> captured_frame_texture;
    Array meshPoints;
    Ref<ArrayMesh> captured_mesh;


protected:
	static void _bind_methods();

public:
	FaceTrack();
	~FaceTrack();

    bool get_should_run();
    void set_should_run(bool new_should_run);

	void _process(double delta) override;
    void _notification(int p_what);
    void _ready() override;
    void _exit_tree() override;
    void stop_app();
    void showFaceFitErrorMessage();
    
    Array get_facial_landmarks() const;
    Ref<ImageTexture> convert_frame_to_texture(const cv::Mat& frame);

    Ref<ImageTexture> get_captured_frame_texture() const;
    
    void drawMeshPoints(cv::Mat& frame);

    Array get_mesh_points() const;
    void printMeshVertices();

    Ref<ArrayMesh> get_captured_mesh() const;

    Ref<ArrayMesh> convert_NVARmesh_to_godot();
};

}

#endif