#include "ExpressionTrack.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/array_mesh.hpp>

using namespace godot;

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ExpressionTrack.h"

#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include "opencv2/opencv.hpp"

#if CV_MAJOR_VERSION >= 4
  #define CV_CAP_PROP_FPS           cv::CAP_PROP_FPS
  #define CV_CAP_PROP_FRAME_COUNT   cv::CAP_PROP_FRAME_COUNT
  #define CV_CAP_PROP_FRAME_HEIGHT  cv::CAP_PROP_FRAME_HEIGHT
  #define CV_CAP_PROP_FRAME_WIDTH   cv::CAP_PROP_FRAME_WIDTH
  #define CV_CAP_PROP_POS_FRAMES    cv::CAP_PROP_POS_FRAMES
  #define CV_INTER_AREA             cv::INTER_AREA
  #define CV_INTER_LINEAR           cv::INTER_LINEAR
#endif // CV_MAJOR_VERSION

#ifndef M_PI
  #define M_PI                      3.1415926535897932385
#endif /* M_PI */
#ifndef M_2PI
  #define M_2PI                     6.2831853071795864769
#endif /* M_2PI */
#ifndef M_PI_2
  #define M_PI_2                    1.5707963267948966192
#endif /* M_PI_2 */
#define D_RADIANS_PER_DEGREE        (M_PI / 180.)
#define F_PI                        ((float)M_PI)
#define F_PI_2                      ((float)M_PI_2)
#define F_2PI                       ((float)M_2PI)
#define F_RADIANS_PER_DEGREE        (float)(M_PI / 180.)
#define CTL(x)                      ((x) & 0x1F)
#define HELP_REQUESTED              411

#define BAIL(err, code)             do {                            err = code; goto bail;   } while(0)

#define DEFAULT_CODEC         "avc1"
#define DEFAULT_FACE_MODEL    "face_model2.nvf"


void ExpressionTrack::_bind_methods() {
  ClassDB::bind_method(D_METHOD("get_landmarks"), &ExpressionTrack::get_landmarks);
  ClassDB::bind_method(D_METHOD("get_landmark_count"), &ExpressionTrack::get_landmark_count);
  ClassDB::bind_method(D_METHOD("get_expression_count"), &ExpressionTrack::get_expression_count);
  ClassDB::bind_method(D_METHOD("get_expressions"), &ExpressionTrack::get_expressions);
  ClassDB::bind_method(D_METHOD("get_landmark_confidence"), &ExpressionTrack::get_landmark_confidence);
  ClassDB::bind_method(D_METHOD("get_pose_rotation"), &ExpressionTrack::get_pose_rotation);
  ClassDB::bind_method(D_METHOD("get_pose_translation"), &ExpressionTrack::get_pose_translation);
  ClassDB::bind_method(D_METHOD("get_pose_transform"), &ExpressionTrack::get_pose_transform);
  ClassDB::bind_method(D_METHOD("get_bounding_boxes"), &ExpressionTrack::get_bounding_boxes);
}

ExpressionTrack::ExpressionTrack() {
  if(Engine::get_singleton()->is_editor_hint()){
    set_process_mode(Node::ProcessMode::PROCESS_MODE_DISABLED);
  }

  nvErr = NvAR_CudaStreamCreate(&_stream);
  if (nvErr!=NVCV_SUCCESS) {
    UtilityFunctions::print("failed to create CUDA stream");
  }
}

ExpressionTrack::~ExpressionTrack() {
  if (_stream) {
    nvErr = NvAR_CudaStreamDestroy(_stream);
    if (nvErr!=NVCV_SUCCESS) {
      UtilityFunctions::print("failed to destroy CUDA stream");
    }
  }
  
  if (_vidIn.isOpened()) {
    _vidIn.release();
  }

  // Deallocate NvCVImage objects
  NvCVImage_Dealloc(&_srcGpu);
  NvCVImage_Dealloc(&_srcImg);

  // Destroy feature handle
  if (_featureHan) {
    NvAR_Destroy(_featureHan);
  }
}

void ExpressionTrack::_ready() {
  if(!Engine::get_singleton()->is_editor_hint()) {
    UtilityFunctions::print("hello from ready :D");

    // get model directory
    const char* _model_path = getenv("NVAR_MODEL_DIR");
    if (_model_path) {
        modelPath = _model_path;
        UtilityFunctions::print("NVAR model dir env var located at: ");
        UtilityFunctions::print(modelPath.c_str());

    } else {
        UtilityFunctions::print("failed to located NVAR model dir env var");
    }

    // open video capture
    if (_vidIn.open(0)) {
      UtilityFunctions::print("successfully opened video capture");

      // intialize face expression feature
      unsigned width, height, frame_rate;
      width = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_WIDTH);
      height = (unsigned)_vidIn.get(CV_CAP_PROP_FRAME_HEIGHT);
      frame_rate = (unsigned)_vidIn.get(CV_CAP_PROP_FPS);

      static const int fps_precision = FPS_PRECISION; // round frame rate for opencv compatibility
      frame_rate = static_cast<int>((frame_rate + 0.5) * fps_precision) / static_cast<double>(fps_precision);

      UtilityFunctions::print("Video Width: ", UtilityFunctions::str(width));
      UtilityFunctions::print("Video Height: ", UtilityFunctions::str(height));
      UtilityFunctions::print("Video FPS: ", UtilityFunctions::str(frame_rate));

      _filtering = 0x037; // bitfield, default, all on except 0x100 enhaced closures
      _poseMode = 1; // 0 - 3DOF implicit for only rotation, 1 - 6DOF explicit for head position
      _enableCheekPuff = 0; // experimental, 0 - off, 1 - on

      nvErr = NvCVImage_Alloc(&_srcGpu, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to allocate srcGpu NvImage memory");
      }

      nvErr = NvCVImage_Alloc(&_srcImg, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU_PINNED, 0);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to allocate srcImg NvImage memory");
      }

      CVWrapperForNvCVImage(&_srcImg, &_ocvSrcImg);

      _cameraIntrinsicParams[0] = static_cast<float>(_srcGpu.height);
      _cameraIntrinsicParams[1] = static_cast<float>(_srcGpu.width) / 2.0f;
      _cameraIntrinsicParams[2] = static_cast<float>(_srcGpu.height) / 2.0f;
      nvErr = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Input(CameraIntristicParams), _cameraIntrinsicParams, NUM_CAMERA_INTRINSIC_PARAMS);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set camera intrinsic params for facial expression feature handle");
      }

      _landmarkCount = NUM_LANDMARKS;
      nvErr = NvAR_Create(NvAR_Feature_FaceExpressions, &_featureHan);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to create facial expression feature handle");
      }

      nvErr = NvAR_SetCudaStream(_featureHan, NvAR_Parameter_Config(CUDAStream), _stream);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set CUDA stream for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(Temporal), _filtering);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set temporal filtering for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(PoseMode), _poseMode);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose mode for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_featureHan, NvAR_Parameter_Config(EnableCheekPuff), _enableCheekPuff);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set enable cheek puff for facial expression feature handle");
      }

      nvErr = NvAR_Load(_featureHan);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to load facial expression feature handle");
      }

      _outputBboxData.assign(25, {0.f, 0.f, 0.f, 0.f});
      _outputBboxes.boxes = _outputBboxData.data();
      _outputBboxes.max_boxes = (uint8_t)_outputBboxData.size();
      _outputBboxes.num_boxes = 0;
      nvErr = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(BoundingBoxes), &_outputBboxes, sizeof(NvAR_BBoxes));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set bounding boxes output for facial expression feature handle");
      }

      _landmarks.resize(_landmarkCount);
      nvErr = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Landmarks), _landmarks.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set landmarks output for facial expression feature handle");
      }

      _landmarkConfidence.resize(_landmarkCount);
      nvErr = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(LandmarksConfidence), _landmarkConfidence.data(), _landmarkCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set landmark confidence output for facial expression feature handle");
      }

      nvErr = NvAR_GetU32(_featureHan, NvAR_Parameter_Config(ExpressionCount), &_exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to get expression count for facial expression feature handle");
      }
      _expressions.resize(_exprCount);
      _expressionZeroPoint.resize(_exprCount, 0.0f);
      _expressionScale.resize(_exprCount, 1.0f);
      _expressionExponent.resize(_exprCount, 1.0f);

      nvErr = NvAR_SetF32Array(_featureHan, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set expression coefficient output for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_featureHan, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set image output for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(Pose), &_pose.rotation, sizeof(NvAR_Quaternion));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose rotation output for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_featureHan, NvAR_Parameter_Output(PoseTranslation), &_pose.translation, sizeof(NvAR_Vector3f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose translation output for facial expression feature handle");
      }

      // capture image
      if (_vidIn.read(_ocvSrcImg)) {
        UtilityFunctions::print("succesfully captured video frame");
        
        // process image
        nvErr = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _stream, nullptr);
        if (nvErr!=NVCV_SUCCESS) {
          UtilityFunctions::print("failed to transfer image to gpu");
        }

        nvErr = NvAR_Run(_featureHan);
        if (nvErr!=NVCV_SUCCESS) {
          UtilityFunctions::print("failed to run facial expression feature");
        } else {
          UtilityFunctions::print("successfully ran facial expression feature");

          // validate expression capture
          /* printPoseRotation();

          if (_poseMode == 1) {
            printPoseTranslation();
          }
          
          printExpressionCoefficients();
          printLandmarkLocations();
          printLandmarkConfidence();
          printBoundingBoxes(); */
        }


      } else {
        UtilityFunctions::print("failed to capture video frame");
      }


    } else {
      UtilityFunctions::print("failed to open video capture");
    }


  }
}

void ExpressionTrack::_process(double delta) {
	
}

void ExpressionTrack::printPoseRotation() {
  UtilityFunctions::print("\nFacial Pose Rotation:");
  
  const auto& rotation = _pose.rotation;
  godot::String rotationStr = "Pose Rotation Quaternion: (" 
  + UtilityFunctions::str(rotation.x) + ", " 
  + UtilityFunctions::str(rotation.y) + ", " 
  + UtilityFunctions::str(rotation.z) + ", " 
  + UtilityFunctions::str(rotation.w) + ")";
  UtilityFunctions::print(rotationStr);
}

void ExpressionTrack::printExpressionCoefficients() {
  UtilityFunctions::print("\nFacial Expression Coefficients:");

  godot::String coeffsStr = "";
  for (size_t i = 0; i < _expressions.size(); ++i) {
      coeffsStr += UtilityFunctions::str(_expressions[i]);
      if (i < _expressions.size() - 1) {
          coeffsStr += ", ";
      }
  }
  UtilityFunctions::print(coeffsStr);
}

void ExpressionTrack::printLandmarkLocations() {
  UtilityFunctions::print("\nFacial Landmark Locations:");
  
  godot::String landmarksStr = "";
  for (size_t i = 0; i < _landmarks.size(); ++i) {
      landmarksStr += "(" + UtilityFunctions::str(_landmarks[i].x) + ", " 
                          + UtilityFunctions::str(_landmarks[i].y) + ")";
      if (i < _landmarks.size() - 1) {
          landmarksStr += ", ";
      }
  }
  UtilityFunctions::print(landmarksStr);
}

void ExpressionTrack::printBoundingBoxes() {
  UtilityFunctions::print("\nBounding Boxes: (x, y, width, height)");

  godot::String bboxesStr = "";
  for (size_t i = 0; i < _outputBboxes.num_boxes; ++i) {
      const auto& box = _outputBboxes.boxes[i];
      bboxesStr += "("+ UtilityFunctions::str(box.x) + ", " 
                      + UtilityFunctions::str(box.y) + ", " 
                      + UtilityFunctions::str(box.width) + ", " 
                      + UtilityFunctions::str(box.height) + ")";
      if (i < _outputBboxes.num_boxes - 1) {
          bboxesStr += ", ";
      }
  }
  UtilityFunctions::print(bboxesStr);
}

void ExpressionTrack::printLandmarkConfidence() {
  UtilityFunctions::print("\nFacial Landmark Confidence:");

  godot::String confidenceStr = "";
  for (size_t i = 0; i < _landmarkConfidence.size(); ++i) {
      confidenceStr += UtilityFunctions::str(_landmarkConfidence[i]);
      if (i < _landmarkConfidence.size() - 1) {
          confidenceStr += ", ";
      }
  }
  UtilityFunctions::print(confidenceStr);
}

void ExpressionTrack::printPoseTranslation() {
  UtilityFunctions::print("\nFacial Pose Translation:");

  const auto& translation = _pose.translation;
  godot::String translationStr = "(" 
      + UtilityFunctions::str(translation.vec[0]) + ", "   // X component
      + UtilityFunctions::str(translation.vec[1]) + ", "   // Y component
      + UtilityFunctions::str(translation.vec[2]) + ")";   // Z component
  UtilityFunctions::print(translationStr);
}

Array ExpressionTrack::get_landmarks() const {
  Array landmarks;
  for (const auto& landmark : _landmarks) {
      landmarks.push_back(Vector2(landmark.x, landmark.y));
  }
  return landmarks;
}

int ExpressionTrack::get_landmark_count() const {
  return _landmarkCount;
}

int ExpressionTrack::get_expression_count() const {
  return _exprCount;
}

Array ExpressionTrack::get_expressions() const {
  Array expressions;
  for (const auto& expression : _expressions) {
      expressions.push_back(expression);
  }
  return expressions;
}

Array ExpressionTrack::get_landmark_confidence() const {
  Array confidences;
  for (const auto& confidence : _landmarkConfidence) {
      confidences.push_back(confidence);
  }
  return confidences;
}

Quaternion ExpressionTrack::get_pose_rotation() const {
  const auto& rotation = _pose.rotation;
  return Quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
}


Vector3 ExpressionTrack::get_pose_translation() const {
  if (_poseMode == 1) {
      const auto& translation = _pose.translation;
      return Vector3(translation.vec[0], translation.vec[1], translation.vec[2]);
  } else {
      return Vector3(0, 0, 0);
  }
}

Transform3D ExpressionTrack::get_pose_transform() const {
    Quaternion rotation_quat = get_pose_rotation();
    Basis basis = Basis(rotation_quat).orthonormalized();
    Vector3 translation = get_pose_translation();

    return Transform3D(basis, translation);
}

Dictionary ExpressionTrack::bounding_box_to_dict(const NvAR_Rect& box) const {
    Dictionary bbox;
    bbox["x"] = box.x;
    bbox["y"] = box.y;
    bbox["width"] = box.width;
    bbox["height"] = box.height;
    return bbox;
}

Array ExpressionTrack::get_bounding_boxes() const {
    Array boxes;
    for (size_t i = 0; i < _outputBboxes.num_boxes; ++i) {
        boxes.push_back(bounding_box_to_dict(_outputBboxes.boxes[i]));
    }
    return boxes;
}
