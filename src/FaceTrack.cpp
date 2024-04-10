#include "FaceTrack.h"
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/array_mesh.hpp>

using namespace godot;

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <fstream>

#include "FaceTrack.h"
#include "FaceEngine.h"
#include "RenderingUtils.h"
#include "nvAR.h"
#include "nvAR_defs.h"
#include "opencv2/opencv.hpp"

#if CV_MAJOR_VERSION >= 4
#define CV_CAP_PROP_FRAME_WIDTH cv::CAP_PROP_FRAME_WIDTH
#define CV_CAP_PROP_FRAME_HEIGHT cv::CAP_PROP_FRAME_HEIGHT
#define CV_CAP_PROP_FPS cv::CAP_PROP_FPS
#define CV_CAP_PROP_FRAME_COUNT cv::CAP_PROP_FRAME_COUNT
#endif

#ifndef M_PI
#define M_PI 3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
#define M_2PI 6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
#define M_PI_2 1.5707963267948966192
#endif /* M_PI_2 */
#define F_PI ((float)M_PI)
#define F_PI_2 ((float)M_PI_2)
#define F_2PI ((float)M_2PI)

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif /* _MSC_VER */


#define BAIL(err, code) \
  do {                  \
    err = code;         \
    goto bail;          \
  } while (0)

const char FaceTrack::windowTitle[] = "Facetrack App";


void FaceTrack::_bind_methods() {

    ClassDB::bind_method(D_METHOD("get_should_run"), &FaceTrack::get_should_run);
    ClassDB::bind_method(D_METHOD("set_should_run", "new_should_run"), &FaceTrack::set_should_run);
    ClassDB::bind_method(D_METHOD("_notification"), &FaceTrack::_notification);

    ClassDB::add_property(
        "FaceTrack",
        PropertyInfo(
            Variant::BOOL,
            "should_run"
        ),
        "set_should_run",
        "get_should_run"
    );

    ClassDB::bind_method(D_METHOD("get_facial_landmarks"), &FaceTrack::get_facial_landmarks);
    ClassDB::bind_method(D_METHOD("get_captured_frame_texture"), &FaceTrack::get_captured_frame_texture);
    ClassDB::bind_method(D_METHOD("get_captured_mesh"), &FaceTrack::get_captured_mesh);

}

bool FaceTrack::get_should_run() {
    return should_run;
}

void FaceTrack::set_should_run(bool new_should_run) {
    should_run = new_should_run;
    UtilityFunctions::print(UtilityFunctions::str(new_should_run));
}

FaceTrack::FaceTrack() {
	// Initialize any variables here.
	if(Engine::get_singleton()->is_editor_hint()){
        set_process_mode(Node::ProcessMode::PROCESS_MODE_DISABLED);
    }
    should_run = true;
}

FaceTrack::~FaceTrack() {
	// Add your cleanup here.
    if (cap.isOpened()){
        UtilityFunctions::print("closing cv::VideoCapture");
        face_ar_engine.destroyFeatures();
        cap.release();
        cv::destroyAllWindows();
    }
}

void FaceTrack::_ready() {
    // if game in game
    if(!Engine::get_singleton()->is_editor_hint()){
        
        // open camera
        if (should_run) {
            UtilityFunctions::print("opening cv::VideoCapture");
            cap.open(0);
        }

        // check if camera opened
        if(cap.isOpened()){
            UtilityFunctions::print("cv::VideoCapture open");

            // read frame from camera
            UtilityFunctions::print("reading frame");
            cap.read(frame);

            // get model directory
            const char* _model_path = getenv("NVAR_MODEL_DIR");
            if (_model_path) {
                modelPath = _model_path;
                UtilityFunctions::print("NVAR model dir env var located at: ");
                UtilityFunctions::print(modelPath.c_str());

            } else {
                UtilityFunctions::print("failed to located NVAR model dir env var");
            }

            // initialize body engine
            UtilityFunctions::print("intializing BodyEngine");
            int numKeyPoints = body_ar_engine.getNumKeyPoints();
            NvAR_BBoxes body_bbox;
            std::vector<NvAR_Point2f> keypoints2D(numKeyPoints * 8);
            std::vector<NvAR_Point3f> keypoints3D(numKeyPoints * 8);
            std::vector<NvAR_Quaternion> jointAngles(numKeyPoints * 8);
            BodyEngine::Err body_err = BodyEngine::Err::errNone;

            UtilityFunctions::print("setting up for 3D body pose tracking");
            body_ar_engine.setAppMode(BodyEngine::mode::keyPointDetection);
            body_ar_engine.setMode(0); // high-quality mode
            body_ar_engine.setBodyStabilization(1);
            body_ar_engine.useCudaGraph(1);
            
            body_err = body_ar_engine.createFeatures(modelPath.c_str(), 1);
            if (body_err != BodyEngine::Err::errNone) {
                UtilityFunctions::print("failed to create features for BodyEngine");
            }

            body_err = body_ar_engine.initFeatureIOParams();
            if (body_err != BodyEngine::Err::errNone) {
                UtilityFunctions::print("failed to initialize feature io params for BodyEngine");
            }

            // initialize face engine
            UtilityFunctions::print("intializing FaceEngine");
            int inputWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int inputHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            UtilityFunctions::print("input height: " + UtilityFunctions::str(inputHeight));
            UtilityFunctions::print("input width: " + UtilityFunctions::str(inputWidth));
            face_ar_engine.setInputImageWidth(inputWidth);
            face_ar_engine.setInputImageHeight(inputHeight);
            face_ar_engine.setFaceStabilization(true);
            face_ar_engine.setNumLandmarks(126);
            
            // setup for face mesh generation
            UtilityFunctions::print("setting up for 3D facial mesh tracking");
            face_ar_engine.setAppMode(FaceEngine::mode::faceMeshGeneration);
            
            nvErr = face_ar_engine.createFeatures(modelPath.c_str(), 1, FaceEngine::mode::faceMeshGeneration);
            if (nvErr != FaceEngine::Err::errNone) {
                UtilityFunctions::print("failed to create features for FaceEngine");
            } 
            
            FaceEngine::Err err = face_ar_engine.initFeatureIOParams();
            if (err != FaceEngine::Err::errNone ) {
                UtilityFunctions::print("failed to initialize feature io params for FaceEngine");
            }

            // process frame with FaceEngine
            if (!frame.empty()) {
                // fit body model
                UtilityFunctions::print("acquiring body box and keypoints...");
                unsigned body_fit_err = body_ar_engine.acquireBodyBoxAndKeyPoints(frame, keypoints2D.data(), keypoints3D.data(), jointAngles.data(), &body_bbox, 0);
                if (body_fit_err != 1) {
                    UtilityFunctions::print("failed to acquire body box");
                } else {
                    UtilityFunctions::print("successfully acquired body box");
                    // Validate body boxes
                    if (body_ar_engine.output_bboxes.num_boxes == 0) {
                        UtilityFunctions::print("zero body boxes");
                    } else {
                        UtilityFunctions::print("at least one body box detected");

                        // validate jointAngles
                        printf("jointAngles: [\n");
                        for (const auto &angle : jointAngles) {
                            printf("%7.1f%7.1f%7.1f%7.1f\n", angle.x, angle.y, angle.z, angle.w);
                        }
                        printf("]\n");

                        // validate 2D keypoints
                        printf("keyPoints: [\n");
                        for (const auto &pt : keypoints2D) {
                            printf("%7.1f%7.1f\n", pt.x, pt.y);
                        }
                        printf("]\n");

                        // validate 3D keypoints
                        printf("3D keyPoints: [\n");
                        for (const auto& pt : keypoints3D) {
                            printf("%7.1f%7.1f%7.1f\n", pt.x, pt.y, pt.z);
                        }
                        printf("]\n");

                    }
                }
                
                // fit face model
                UtilityFunctions::print("acquiring face box and landmarks...");
                nvErr = face_ar_engine.fitFaceModel(frame);
                if (nvErr != FaceEngine::Err::errNone) {
                    UtilityFunctions::print("fit face model failed");
                } else {
                    UtilityFunctions::print("face model fit to frame!");
                    
                    // capture mesh
                    UtilityFunctions::print("capturing face mesh for engine");
                    captured_mesh = convert_NVARmesh_to_godot();
                    
                    // capture landmarks
                    UtilityFunctions::print("capturing landmarks for engine");
                    captured_facial_landmarks.clear();
                    
                    unsigned int numLandmarks = face_ar_engine.getNumLandmarks();
                    NvAR_Point2f* landmarks = face_ar_engine.getLandmarks();

                    for (unsigned int i = 0; i < numLandmarks; ++i) {
                        captured_facial_landmarks.append(Vector2(landmarks[i].x, landmarks[i].y));
                    }

                    // capture bbox
                    output_bbox = *face_ar_engine.getLargestBox();

                    // capture facial expressions
                    UtilityFunctions::print("capturing facial expressions for engine");
                    unsigned int numCoefficients = face_ar_engine.getNumExpressionCoefficients();
                    float* expressionCoefficients = face_ar_engine.getExpressionCoefficients();

                    for (unsigned int i = 0; i < numCoefficients; ++i) {
                        godot::String coefficientStr = UtilityFunctions::str(expressionCoefficients[i]);
                        UtilityFunctions::print("Coefficient " + UtilityFunctions::str(i) + ": " + coefficientStr);
                    }

                    // capture pose
                    UtilityFunctions::print("capturing pose for engine");
                    NvAR_Quaternion* pose = face_ar_engine.getPose();
                    if (pose != nullptr) {
                        godot::String poseStr = "Quaternion Pose: [" 
                                            + UtilityFunctions::str(pose->x) + ", " 
                                            + UtilityFunctions::str(pose->y) + ", " 
                                            + UtilityFunctions::str(pose->z) + ", " 
                                            + UtilityFunctions::str(pose->w) + "]";
                        UtilityFunctions::print(poseStr);
                    } else {
                        UtilityFunctions::print("Pose is null.");
                    }


                }

                // annotate frame
                if (nvErr == FaceEngine::Err::errNone) {
                    // Draw a rectangle around the detected face
                    UtilityFunctions::print("drawing face box");
                    if (output_bbox.width > 0 && output_bbox.height > 0) {
                        cv::rectangle(frame, 
                            cv::Point(lround(output_bbox.x), lround(output_bbox.y)), 
                            cv::Point(lround(output_bbox.x + output_bbox.width), lround(output_bbox.y + output_bbox.height)), 
                            cv::Scalar(0, 255, 0), 2);
                    }

                    // draw landmark locations
                    UtilityFunctions::print("drawing landmarks");
                    unsigned int numLandmarks = face_ar_engine.getNumLandmarks(); // Assuming you have this method
                    NvAR_Point2f* landmarks = face_ar_engine.getLandmarks();

                    for (unsigned int i = 0; i < numLandmarks; ++i) {
                        const NvAR_Point2f& landmark = landmarks[i];
                        cv::circle(frame, cv::Point(static_cast<int>(landmark.x), static_cast<int>(landmark.y)), 2, cv::Scalar(0, 0, 255), -1);
                    }

                } else if (nvErr == face_ar_engine.FaceEngine::Err::errRun) {
                    UtilityFunctions::print("face detection failed, errRun");
                } else {
                    UtilityFunctions::print("face detection failed, unknown case");
                }
                
                // display annotated frame
                cv::imshow("captured frame", frame);

                // capture annotated frame
                UtilityFunctions::print("capturing frame for engine");
                captured_frame_texture = convert_frame_to_texture(frame);
            }
        }
    }
}

void FaceTrack::_notification(int p_what) {
    if (p_what == NOTIFICATION_WM_CLOSE_REQUEST) {
        // Your _ready() code goes here
        UtilityFunctions::print("handling close request");
        stop_app();
    }
}

void FaceTrack::_exit_tree() {
    UtilityFunctions::print("handling exit tree");
    stop_app();
}

void FaceTrack::_process(double delta) {
	
}

void FaceTrack::stop_app() {
    if (cap.isOpened()){
        UtilityFunctions::print("closing cv::VideoCapture");
        cap.release();
    }
    cv::destroyAllWindows();
    face_ar_engine.destroyFeatures();
}

void FaceTrack::showFaceFitErrorMessage() {
    cv::Mat errBox = cv::Mat::zeros(120, 640, CV_8UC3);
    cv::putText(errBox, cv::String("Warning: Face Fitting needs face_model2.nvf in the path --model_path"), cv::Point(20, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(errBox, cv::String("or in NVAR_MODEL_DIR environment variable."), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(errBox, cv::String("See samples\\FaceTrack\\Readme.txt"), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX,
                0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(errBox, cv::String("Press any key to continue with other features"), cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);
    cv::imshow(windowTitle, errBox);
    cv::waitKey(0);
}

Array FaceTrack::get_facial_landmarks() const {
    return captured_facial_landmarks;
}

Ref<ImageTexture> FaceTrack::convert_frame_to_texture(const cv::Mat &frame) {
    // Convert the frame from BGR to RGB as Godot uses RGB format
    cv::Mat rgb_frame;
    cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);

    // Create a PackedByteArray and copy the rgb_frame data into it
    PackedByteArray pbArray;
    size_t total_bytes = rgb_frame.total() * rgb_frame.elemSize();
    pbArray.resize(total_bytes);
    std::memcpy(pbArray.ptrw(), rgb_frame.data, total_bytes);

    // Create an Image using create_from_data
    Ref<Image> img = Image::create_from_data(
        rgb_frame.cols, // width
        rgb_frame.rows, // height
        false, // no mipmaps
        Image::FORMAT_RGB8,
        pbArray
    );

    // Create an ImageTexture from the Image
    Ref<ImageTexture> texture = ImageTexture::create_from_image(img);

    return texture;
}

Ref<ImageTexture> FaceTrack::get_captured_frame_texture() const {
    return captured_frame_texture;
}

void FaceTrack::drawMeshPoints(cv::Mat& frame) {
    NvAR_FaceMesh* faceMesh = face_ar_engine.getFaceMesh();
    if (!faceMesh || faceMesh->num_vertices == 0) {
        std::cout << "No face mesh data available." << std::endl;
        return;
    }

    int inputWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int inputHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    for (unsigned int i = 0; i < faceMesh->num_vertices; i++) {
        const NvAR_Vector3f& vertex = faceMesh->vertices[i];

        // Assuming the mesh coordinates are roughly around a center (0,0,10)
        // Normalize the coordinates
        float normalizedX = vertex.vec[0] / 10.0f;
        float normalizedY = vertex.vec[1] / 10.0f;

        // Scale and translate points to the image frame
        float imageX = (normalizedX + 1.0f) * (inputWidth / 2.0f);
        float imageY = (1.0f - normalizedY) * (inputHeight / 2.0f); // Flipping Y axis

        // Draw the point
        cv::circle(frame, cv::Point(static_cast<int>(imageX), static_cast<int>(imageY)), 1, cv::Scalar(0, 255, 0), -1);
    }
}

void FaceTrack::printMeshVertices() {
    NvAR_FaceMesh* faceMesh = face_ar_engine.getFaceMesh();
    if (!faceMesh || faceMesh->num_vertices == 0) {
        std::cout << "No face mesh data available." << std::endl;
        return;
    } else {
        UtilityFunctions::print(UtilityFunctions::str(faceMesh->num_vertices));
    }

    for (unsigned int i = 0; i < faceMesh->num_vertices; i++) {
        const NvAR_Vector3f& vertex = faceMesh->vertices[i];
        std::cout << "Vertex " << i << ": x=" << vertex.vec[0] << ", y=" << vertex.vec[1] << ", z=" << vertex.vec[2] << std::endl;
    }
}


Ref<ArrayMesh> FaceTrack::get_captured_mesh() const {
    return captured_mesh;
}

Ref<ArrayMesh> FaceTrack::convert_NVARmesh_to_godot() {
    // Access the face mesh directly from face_ar_engine
    NvAR_FaceMesh* faceMesh = face_ar_engine.getFaceMesh();

    // Create a new ArrayMesh
    Ref<ArrayMesh> godot_mesh;
    godot_mesh.instantiate();

    // Prepare vertex array
    PackedVector3Array vertices;
    vertices.resize(faceMesh->num_vertices);

    for (int i = 0; i < faceMesh->num_vertices; ++i) {
        // Convert each vertex from NvAR_FaceMesh to Godot Vector3 and add to vertices
        vertices.set(i, Vector3(faceMesh->vertices[i].vec[0], faceMesh->vertices[i].vec[1], faceMesh->vertices[i].vec[2]));
    }

    // Prepare indices array
    PackedInt32Array indices;
    indices.resize(faceMesh->num_triangles * 3);

    for (int i = 0; i < faceMesh->num_triangles; ++i) {
        // respect winding order for normals
        indices.set(i * 3, faceMesh->tvi[i].vec[1]);
        indices.set(i * 3 + 1, faceMesh->tvi[i].vec[0]);
        indices.set(i * 3 + 2, faceMesh->tvi[i].vec[2]);
    }

    // Create a new surface
    Array arrays;
    arrays.resize(Mesh::ARRAY_MAX);
    arrays[Mesh::ARRAY_VERTEX] = vertices;
    arrays[Mesh::ARRAY_INDEX] = indices;

    // Add surface to mesh
    godot_mesh->add_surface_from_arrays(Mesh::PRIMITIVE_TRIANGLES, arrays);

    return godot_mesh;
}





