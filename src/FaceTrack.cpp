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

#define DEFAULT_FACE_MODEL    "face_model2.nvf"

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
    if(!Engine::get_singleton()->is_editor_hint()){
        UtilityFunctions::print("hello from ready");
        if (should_run) {
            UtilityFunctions::print("opening cv::VideoCapture");
            cap.open(0);
        }

        if(cap.isOpened()){
            UtilityFunctions::print("cv::VideoCapture open");

            UtilityFunctions::print("reading frame");
            cap.read(frame);

            int inputWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            int inputHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            UtilityFunctions::print("input height: " + UtilityFunctions::str(inputHeight));
            UtilityFunctions::print("input width: " + UtilityFunctions::str(inputWidth));
            face_ar_engine.setInputImageWidth(inputWidth);
            face_ar_engine.setInputImageHeight(inputHeight);
            face_ar_engine.setFaceStabilization(true);
            face_ar_engine.setNumLandmarks(126);
            
            const char* _model_path = getenv("NVAR_MODEL_DIR");
            if (_model_path) {
                modelPath = _model_path;
                UtilityFunctions::print("NVAR model dir env var located at: ");
                UtilityFunctions::print(modelPath.c_str());

            } else {
                UtilityFunctions::print("failed to located NVAR model dir env var");
            }

            // solve expressions
            NvCV_Status expression_err;
            NvAR_FeatureHandle expression_handle;
            CUstream expression_stream = 0;
            std::vector<NvAR_Point2f> _expression_landmarks;
            std::vector<NvAR_Rect> _outputBboxData;
            NvAR_BBoxes        _outputBboxes;
            std::vector<float> _expressions, _expressionZeroPoint, _expressionScale, _expressionExponent, _eigenvalues, _landmark_confidence;
            NvCVImage frame_image;
             struct Pose {
                NvAR_Quaternion rotation;
                NvAR_Vector3f translation;
                float* data() { return &rotation.x; }
                const float* data() const { return &rotation.x; }
            };            
            Pose _pose;

            unsigned _filtering = 0;
            unsigned _pose_mode = 0;
            unsigned _expr_mode = 2;
            unsigned _enable_cheek_puff = 0;
            unsigned _exprCount;

            expression_err = NvAR_Create(NvAR_Feature_FaceExpressions, &expression_handle);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to create facial expression feature");
            }

            expression_err = NvAR_SetString(expression_handle, NvAR_Parameter_Config(ModelDir), modelPath.c_str());
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set model dir");
            }


            expression_err = NvAR_SetCudaStream(expression_handle, NvAR_Parameter_Config(CUDAStream), expression_stream);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set CUDA stream");
            }

            expression_err = NvAR_SetU32(expression_handle, NvAR_Parameter_Config(Temporal), _filtering);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set temporal filtering");
            }

            expression_err = NvAR_SetU32(expression_handle, NvAR_Parameter_Config(PoseMode), _pose_mode);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set pose_mode");
            }

            expression_err = NvAR_SetU32(expression_handle, NvAR_Parameter_Config(EnableCheekPuff), _enable_cheek_puff);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set cheek puff");
            }

            expression_err = NvAR_Load(expression_handle);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to load facial expression feature");
            } else {
                UtilityFunctions::print("facial expression feature loaded");
            }

            _outputBboxData.assign(25, { 0.f, 0.f, 0.f, 0.f });
            _outputBboxes.boxes = _outputBboxData.data();
            _outputBboxes.max_boxes = (uint8_t)_outputBboxData.size();
            _outputBboxes.num_boxes = 0;
            expression_err = NvAR_SetObject(expression_handle, NvAR_Parameter_Output(BoundingBoxes), &_outputBboxes, sizeof(NvAR_BBoxes));
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set bbox object");
            }

            _expression_landmarks.resize(126);
            expression_err = NvAR_SetObject(expression_handle, NvAR_Parameter_Output(Landmarks), _expression_landmarks.data(), sizeof(NvAR_Point2f));
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set landmarks object");
            }

            _landmark_confidence.resize(126);
            expression_err = NvAR_SetF32Array(expression_handle, NvAR_Parameter_Output(LandmarksConfidence), _landmark_confidence.data(), 126);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set landmark confidence array");
            }

            expression_err = NvAR_GetU32(expression_handle, NvAR_Parameter_Config(ExpressionCount), &_exprCount);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to get expression count");
            }

            _expressions.resize(_exprCount);
            _expressionZeroPoint.resize(_exprCount, 0.0f);
            _expressionScale.resize(_exprCount, 1.0f);
            _expressionExponent.resize(_exprCount, 1.0f);

            expression_err = NvAR_SetF32Array(expression_handle, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount);
            if (expression_err!=NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set expression coefficient array");
            } else {
                UtilityFunctions::print("successfully set expression coefficient array");
            }

            ConvertMatToNvCVImage(frame, &frame_image); // Ensure the implementation of this function

            expression_err = NvAR_SetObject(expression_handle, NvAR_Parameter_Input(Image), &frame_image, sizeof(frame_image));
            if (expression_err != NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set the source image for expression analysis");
            }

            float cameraIntrinsicParams[3] = {static_cast<float>(inputHeight), inputWidth / 2.0f, inputHeight / 2.0f}; // focal length, cx, cy
            expression_err = NvAR_SetF32Array(expression_handle, NvAR_Parameter_Input(CameraIntrinsicParams), cameraIntrinsicParams, 3);
            if (expression_err != NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set camera intrinsic parameters");
            }

            expression_err = NvAR_SetObject(expression_handle, NvAR_Parameter_Output(Pose), &_pose.rotation, sizeof(NvAR_Quaternion));
            if (expression_err != NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set pose rotation object");
            }

            expression_err = NvAR_SetObject(expression_handle, NvAR_Parameter_Output(PoseTranslation), &_pose.translation, sizeof(NvAR_Vector3f));
            if (expression_err != NVCV_SUCCESS) {
                UtilityFunctions::print("failed to set pose rotation object");
            }

            // Run the facial expression feature
            expression_err = NvAR_Run(expression_handle);
            if (expression_err != NVCV_SUCCESS) {
                UtilityFunctions::print("failed to run facial expression feature");
            } else {
                UtilityFunctions::print("facial expression feature run successfully");
            }

            // Process and print the expression coefficients
            // After running the facial expression feature
            for (size_t i = 0; i < _expressions.size(); ++i) {
                std::cout << "Expression Coefficient " << i << ": " << _expressions[i] << std::endl;
            }

            // Cleanup for expression handling
            NvAR_Destroy(expression_handle); // Destroy expression feature handle

            // Deallocate NvCVImage used for capturing frame
            NvCVImage_Dealloc(&frame_image);

            // Deallocate std::vectors
            _expression_landmarks.clear();
            _outputBboxData.clear();
            _expressions.clear();
            _expressionZeroPoint.clear();
            _expressionScale.clear();
            _expressionExponent.clear();
            _eigenvalues.clear();
            _landmark_confidence.clear();

// Note: CUstream (expression_stream) does not require manual deallocation
// if created using CUDA's default stream creation methods.

// Other variables like Pose, unsigned ints don't require explicit deallocation.


            // setup for face mesh generation
            UtilityFunctions::print("intializing FaceEngine...");
            face_ar_engine.setAppMode(FaceEngine::mode::faceMeshGeneration);
            
            // set model path
        
            nvErr = face_ar_engine.createFeatures(modelPath.c_str(), 1, FaceEngine::mode::faceMeshGeneration);
            if (nvErr != FaceEngine::Err::errNone) {
                UtilityFunctions::print("FaceEngine initialization failed");
            }
            else {
                UtilityFunctions::print("FaceEngine successfully initialized");
            } 
            
            UtilityFunctions::print("initializing FaceEngine featureIOParams...");
            FaceEngine::Err err = face_ar_engine.initFeatureIOParams();
            if (err != FaceEngine::Err::errNone ) {
                UtilityFunctions::print("FaceEngine featureIOParams failed to initialize!");
            } else {
                UtilityFunctions::print("FaceEngine featureIOParams successfully intialized");
            }
        
            if (!frame.empty()) {
                UtilityFunctions::print("acquiring face box and landmarks...");
                int num_landmarks = face_ar_engine.getNumLandmarks();
                std::vector<NvAR_Point2f> facial_landmarks(num_landmarks);

                nvErr = face_ar_engine.fitFaceModel(frame);
                if (nvErr != FaceEngine::Err::errNone) {
                    UtilityFunctions::print("fit face model failed");
                } else {
                    UtilityFunctions::print("face model fit to frame!");
                    captured_mesh = convert_NVARmesh_to_godot();
                }

                face_ar_engine.destroyFeatures();
                face_ar_engine.setAppMode(FaceEngine::mode::landmarkDetection);
                nvErr = face_ar_engine.createFeatures(modelPath.c_str(), 1, FaceEngine::mode::landmarkDetection);
                if (nvErr != FaceEngine::Err::errNone) {
                    UtilityFunctions::print("FaceEngine initialization failed (2nd)");
                }
                else {
                    UtilityFunctions::print("FaceEngine successfully initialized (2nd)");
                } 

                FaceEngine::Err err = face_ar_engine.initFeatureIOParams();
                if (err != FaceEngine::Err::errNone ) {
                    UtilityFunctions::print("FaceEngine featureIOParams failed to initialize! (2nd)");
                } else {
                    UtilityFunctions::print("FaceEngine featureIOParams successfully intialized (2nd)");
                }

                nvErr = face_ar_engine.acquireFaceBoxAndLandmarks(frame, facial_landmarks.data(), output_bbox, 0);
                if (nvErr == FaceEngine::Err::errNone) {
                    // Draw a rectangle around the detected face
                   
                    UtilityFunctions::print("Face Box acquired, processing...");
                    if (output_bbox.width > 0 && output_bbox.height > 0) {
                        cv::rectangle(frame, 
                            cv::Point(lround(output_bbox.x), lround(output_bbox.y)), 
                            cv::Point(lround(output_bbox.x + output_bbox.width), lround(output_bbox.y + output_bbox.height)), 
                            cv::Scalar(0, 255, 0), 2);

                        UtilityFunctions::print("face detected!");
                    }

                    UtilityFunctions::print("drawing landmarks...");
                    for (const auto& landmark : facial_landmarks) {
                       cv::circle(frame, cv::Point(static_cast<int>(landmark.x), static_cast<int>(landmark.y)), 2, cv::Scalar(0, 0, 255), -1);
                    }

                    UtilityFunctions::print("capturing landmarks for engine");
                    captured_facial_landmarks.clear();

                    for (const auto& landmark : facial_landmarks) {
                        captured_facial_landmarks.append(Vector2(landmark.x, landmark.y));
                    }

                } else if (nvErr == face_ar_engine.FaceEngine::Err::errRun) {
                    UtilityFunctions::print("face detection failed, errRun");
                } else {
                    UtilityFunctions::print("face detection failed, unknown case");
                }
                
                cv::imshow("captured frame", frame);

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

NvCV_Status FaceTrack::ConvertMatToNvCVImage(const cv::Mat& mat, NvCVImage* nvImage) {
  if (!mat.empty() && mat.depth() == CV_8U) {
    NvCV_Status status = NvCVImage_Alloc(nvImage, mat.cols, mat.rows, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU, 1);
    if (status != NVCV_SUCCESS) return status;

    // Assuming BGR format and contiguous memory in cv::Mat
    memcpy(nvImage->pixels, mat.data, mat.total() * mat.elemSize());
    return NVCV_SUCCESS;
  } else {
    return NVCV_ERR_FEATURENOTFOUND;
  }
}






