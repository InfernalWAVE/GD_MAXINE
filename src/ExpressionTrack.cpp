/****************************************************************
 * Copyright (c) 2024 Ryan Powell
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 ****************************************************************/

#include "ExpressionTrack.h"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

using namespace godot;

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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

#define BIND_METHOD(f_name) \
    ClassDB::bind_method(D_METHOD(#f_name), &ExpressionTrack::f_name)


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

  ClassDB::bind_method(D_METHOD("set_landmarks", "p_value"), &ExpressionTrack::set_landmarks);
  ClassDB::bind_method(D_METHOD("set_landmark_count", "p_value"), &ExpressionTrack::set_landmark_count);
  ClassDB::bind_method(D_METHOD("set_expression_count", "p_value"), &ExpressionTrack::set_expression_count);
  ClassDB::bind_method(D_METHOD("set_expressions", "p_value"), &ExpressionTrack::set_expressions);
  ClassDB::bind_method(D_METHOD("set_landmark_confidence", "p_value"), &ExpressionTrack::set_landmark_confidence);
  ClassDB::bind_method(D_METHOD("set_pose_rotation", "p_value"), &ExpressionTrack::set_pose_rotation);
  ClassDB::bind_method(D_METHOD("set_pose_translation", "p_value"), &ExpressionTrack::set_pose_translation);
  ClassDB::bind_method(D_METHOD("set_pose_transform", "p_value"), &ExpressionTrack::set_pose_transform);
  ClassDB::bind_method(D_METHOD("set_bounding_boxes", "p_value"), &ExpressionTrack::set_bounding_boxes);


  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::ARRAY, "landmarks"), "set_landmarks", "get_landmarks");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::INT, "landmark_count"), "set_landmark_count", "get_landmark_count");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::INT, "expression_count"), "set_expression_count", "get_expression_count");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::ARRAY, "expressions"), "set_expressions", "get_expressions");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::ARRAY, "landmark_confidence"), "set_landmark_confidence", "get_landmark_confidence");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::QUATERNION, "pose_rotation"), "set_pose_rotation", "get_pose_rotation");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::VECTOR3, "pose_translation"), "set_pose_translation", "get_pose_translation");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::TRANSFORM3D, "pose_transform"), "set_pose_transform", "get_pose_transform");
  ClassDB::add_property("ExpressionTrack", PropertyInfo(Variant::ARRAY, "bounding_boxes"), "set_bounding_boxes", "get_bounding_boxes");
}

ExpressionTrack::ExpressionTrack() { 
  // initialize members
  _landmarks.clear();
  _expressions.clear();
  _landmarkConfidence.clear();
  _expressionOutputBboxData.clear();
  _referencePose.clear();

  _landmarkCount = 0;
  _exprCount = 0;
  _numKeyPoints = 0;
  _globalExpressionParam = 1.0f;
  
  _pose.rotation = NvAR_Quaternion{0.0f, 0.0f, 0.0f, 1.0f};
  _pose.translation = NvAR_Vector3f{0.0f, 0.0f, 0.0f};

  _expressionOutputBboxData.resize(25, {0.0, 0.0, 0.0, 0.0});
  _expressionOutputBboxes.boxes = _expressionOutputBboxData.data();
  _expressionOutputBboxes.max_boxes = static_cast<uint8_t>(_expressionOutputBboxData.size());
  _expressionOutputBboxes.num_boxes = 0;
  
  nvErr = NvAR_CudaStreamCreate(&_expressionStream);
  if (nvErr!=NVCV_SUCCESS) {
    UtilityFunctions::print("failed to create expression CUDA stream");
  }

  nvErr = NvAR_CudaStreamCreate(&_bodyStream);
  if (nvErr!=NVCV_SUCCESS) {
    UtilityFunctions::print("failed to create body CUDA stream");
  }

  // only process in game
  if(Engine::get_singleton()->is_editor_hint()){
    set_process_mode(Node::ProcessMode::PROCESS_MODE_DISABLED);
  }

}

ExpressionTrack::~ExpressionTrack() {
  if (_expressionStream) {
    nvErr = NvAR_CudaStreamDestroy(_expressionStream);
    if (nvErr!=NVCV_SUCCESS) {
      UtilityFunctions::print("failed to destroy expression CUDA stream");
    }
  }

  if (_bodyStream) {
    nvErr = NvAR_CudaStreamDestroy(_bodyStream);
    if (nvErr!=NVCV_SUCCESS) {
      UtilityFunctions::print("failed to destroy body CUDA stream");
    }
  }
  
  if (_vidIn.isOpened()) {
    _vidIn.release();
  }

  // Deallocate NvCVImage objects
  NvCVImage_Dealloc(&_srcGpu);
  NvCVImage_Dealloc(&_srcImg);

  // Destroy feature handle
  if (_expressionFeature) {
    NvAR_Destroy(_expressionFeature);
  }

  if (_bodyFeature) {
    NvAR_Destroy(_bodyFeature);
  }

  continue_processing = false;
  if (processing_thread.joinable()) {
    processing_thread.join();
  }
}

void ExpressionTrack::_ready() {
  if(!Engine::get_singleton()->is_editor_hint()) {
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

      UtilityFunctions::print("video width: ", UtilityFunctions::str(width));
      UtilityFunctions::print("video height: ", UtilityFunctions::str(height));
      UtilityFunctions::print("video FPS: ", UtilityFunctions::str(frame_rate));

      _expressionFiltering = 0x037; // bitfield, default, all on except 0x100 enhaced closures
      _poseMode = 1; // 0 - 3DOF implicit for only rotation, 1 - 6DOF explicit for head position
      _enableCheekPuff = 0; // experimental, 0 - off, 1 - on
      
      _bodyTrackMode = 1; // 0 - High Quality, 1 - High Performance
      _bodyFiltering = 1; // 0 - disabled, 1 - enabled
      _useCudaGraph = true;


      // allocate src images
      nvErr = NvCVImage_Alloc(&_srcGpu, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_GPU, 1);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to allocate srcGpu NvImage memory");
      }

      nvErr = NvCVImage_Alloc(&_srcImg, width, height, NVCV_BGR, NVCV_U8, NVCV_CHUNKY, NVCV_CPU_PINNED, 0);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to allocate srcImg NvImage memory");
      }

      CVWrapperForNvCVImage(&_srcImg, &_ocvSrcImg);
      _landmarkCount = NUM_LANDMARKS;


      // create features
      nvErr = NvAR_Create(NvAR_Feature_FaceExpressions, &_expressionFeature);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to create facial expression feature handle");
      }

      nvErr = NvAR_Create(NvAR_Feature_BodyPoseEstimation, &_bodyFeature);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to create body track feature handle");
      }


      // set feature cuda streams
      nvErr = NvAR_SetCudaStream(_expressionFeature, NvAR_Parameter_Config(CUDAStream), _expressionStream);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set CUDA stream for facial expression feature handle");
      }

      nvErr = NvAR_SetCudaStream(_bodyFeature, NvAR_Parameter_Config(CUDAStream), _bodyStream);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set CUDA stream for body track feature handle");
      }


      // set temporal filtering
      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(Temporal), _expressionFiltering);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set temporal filtering for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_bodyFeature, NvAR_Parameter_Config(Temporal), _bodyFiltering);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set temporal filtering for body track feature handle");
      }

      // set facial expression config
      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(PoseMode), _poseMode);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose mode for facial expression feature handle");
      }

      nvErr = NvAR_SetU32(_expressionFeature, NvAR_Parameter_Config(EnableCheekPuff), _enableCheekPuff);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set enable cheek puff for facial expression feature handle");
      }

      // set body track config
      nvErr = NvAR_SetU32(_bodyFeature, NvAR_Parameter_Config(Mode), _bodyTrackMode);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set tracking mode for body tracking feature handle");
      }

      nvErr = NvAR_SetF32(_bodyFeature, NvAR_Parameter_Config(UseCudaGraph), _useCudaGraph);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set use CUDA grpah for body tracking feature handle");
      }

      // load features
      nvErr = NvAR_Load(_expressionFeature);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to load facial expression feature handle");
      }

      nvErr = NvAR_Load(_bodyFeature);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to load body tracking feature handle");
      }

      _expressionOutputBboxData.assign(25, {0.f, 0.f, 0.f, 0.f});
      _expressionOutputBboxes.boxes = _expressionOutputBboxData.data();
      _expressionOutputBboxes.max_boxes = (uint8_t)_expressionOutputBboxData.size();
      _expressionOutputBboxes.num_boxes = 0;
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(BoundingBoxes), &_expressionOutputBboxes, sizeof(NvAR_BBoxes));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set bounding boxes output for facial expression feature handle");
      }

      _landmarks.resize(_landmarkCount);
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(Landmarks), _landmarks.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set landmarks output for facial expression feature handle");
      }

      _landmarkConfidence.resize(_landmarkCount);
      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Output(LandmarksConfidence), _landmarkConfidence.data(), _landmarkCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set landmark confidence output for facial expression feature handle");
      }

      // get feature counts
      nvErr = NvAR_GetU32(_expressionFeature, NvAR_Parameter_Config(ExpressionCount), &_exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to get expression count for facial expression feature handle");
      }
      _expressions.resize(_exprCount);
      _expressionZeroPoint.resize(_exprCount, 0.0f);
      _expressionScale.resize(_exprCount, 1.0f);
      _expressionExponent.resize(_exprCount, 1.0f);

      nvErr = NvAR_GetU32(_bodyFeature, NvAR_Parameter_Config(NumKeyPoints), &_numKeyPoints);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to get keypoint count for boody tracking feature handle");
      } else {
        UtilityFunctions::print("keypoints retrieved: ", UtilityFunctions::str(_numKeyPoints));
      }

      _keypoints.assign(_numKeyPoints, { 0.f, 0.f });
      _keypoints3D.assign(_numKeyPoints, { 0.f, 0.f, 0.f });
      _jointAngles.assign(_numKeyPoints, { 0.f, 0.f, 0.f, 1.f });
      _keypoints_confidence.assign(_numKeyPoints, 0.f);
      _referencePose.assign(_numKeyPoints, { 0.f, 0.f, 0.f });

      const void* pReferencePose;
      nvErr = NvAR_GetObject(_bodyFeature, NvAR_Parameter_Config(ReferencePose), &pReferencePose,
                            sizeof(NvAR_Point3f));
      memcpy(_referencePose.data(), pReferencePose, sizeof(NvAR_Point3f) * _numKeyPoints);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to get reference pose for boody tracking feature handle");
      }


      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Output(ExpressionCoefficients), _expressions.data(), _exprCount);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set expression coefficient output for facial expression feature handle");
      }

      // set feature inputs
      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set image input for facial expression feature handle");
      }
      
      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Input(Image), &_srcGpu, sizeof(NvCVImage));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set image input for body tracking feature handle");
      }

      _cameraIntrinsicParams[0] = static_cast<float>(_srcGpu.height);
      _cameraIntrinsicParams[1] = static_cast<float>(_srcGpu.width) / 2.0f;
      _cameraIntrinsicParams[2] = static_cast<float>(_srcGpu.height) / 2.0f;

      nvErr = NvAR_SetF32Array(_expressionFeature, NvAR_Parameter_Input(CameraIntrinsicParams), _cameraIntrinsicParams, NUM_CAMERA_INTRINSIC_PARAMS);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set camera intrinsic params for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(Pose), &_pose.rotation, sizeof(NvAR_Quaternion));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose rotation output for facial expression feature handle");
      }

      nvErr = NvAR_SetObject(_expressionFeature, NvAR_Parameter_Output(PoseTranslation), &_pose.translation, sizeof(NvAR_Vector3f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set pose translation output for facial expression feature handle");
      }

      // finish body track output, TODO reorder this once it is all working
      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(KeyPoints), _keypoints.data(), sizeof(NvAR_Point2f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set keypoints output for body track feature handle");
      }

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(KeyPoints3D), _keypoints3D.data(), sizeof(NvAR_Point3f));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set keypoints3D output for body track feature handle");
      }

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(JointAngles), _jointAngles.data(), sizeof(NvAR_Quaternion));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set joint angles output for body track feature handle");
      }

      nvErr = NvAR_SetF32Array(_bodyFeature, NvAR_Parameter_Output(KeyPointsConfidence), _keypoints_confidence.data(), sizeof(float));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set keypoints confidence output for body track feature handle");
        UtilityFunctions::print("ERROR CODE: ", UtilityFunctions::str(nvErr));
      }

      _bodyOutputBboxData.assign(25, { 0.f, 0.f, 0.f, 0.f });
      _bodyOutputBboxConfData.assign(25, 0.f);
      _bodyOutputBboxes.boxes = _bodyOutputBboxData.data();
      _bodyOutputBboxes.max_boxes = (uint8_t)_bodyOutputBboxData.size();
      _bodyOutputBboxes.num_boxes = 0;

      nvErr = NvAR_SetObject(_bodyFeature, NvAR_Parameter_Output(BoundingBoxes), &_bodyOutputBboxes, sizeof(NvAR_BBoxes));
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set bounding box output for body track feature handle");
      }

      nvErr = NvAR_SetF32Array(_bodyFeature, NvAR_Parameter_Output(BoundingBoxesConfidence), _bodyOutputBboxConfData.data(), _bodyOutputBboxes.max_boxes);
      if (nvErr!=NVCV_SUCCESS) {
        UtilityFunctions::print("failed to set bounding box confidence output for body track feature handle");
      }

      // capture image
      if (_vidIn.read(_ocvSrcImg)) {        
        // process image
        nvErr = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _expressionStream, nullptr);
        if (nvErr!=NVCV_SUCCESS) {
          UtilityFunctions::print("failed to transfer image to gpu");
        }

        nvErr = NvAR_Run(_expressionFeature);
        if (nvErr!=NVCV_SUCCESS) {
          UtilityFunctions::print("failed to run facial expression feature");
        } else {
          normalizeExpressionsWeights();

          // start capture on separate thread
          start_processing_thread();

          UtilityFunctions::print("successfully initialized facial expression feature");
        }

        nvErr = NvAR_Run(_bodyFeature);
        if (nvErr!=NVCV_SUCCESS) {
          UtilityFunctions::print("failed to run body tracking feature");
        } else {
          UtilityFunctions::print("successfully initialized body tracking feature");
        }

      } else {
        UtilityFunctions::print("failed to capture video frame");
      }
    } else {
      UtilityFunctions::print("failed to open video capture");
    }
  }
}

void ExpressionTrack::printCapture() {
  // validate expression capture
  printPoseRotation();

  if (_poseMode == 1) {
    printPoseTranslation();
  }
  
  printExpressionCoefficients();
  printLandmarkLocations();
  printLandmarkConfidence();
  printBoundingBoxes();
}


void ExpressionTrack::_process(double delta) {
    {
        // lock to ensure thread-safe access to _ocvSrcImg
        std::lock_guard<std::mutex> lock(processing_mutex);

        cv::imshow("Src Image", _ocvSrcImg);

        // assumes _ocvSrcImg has already been updated by processing_loop
        // transfer image to GPU
        nvErr = NvCVImage_Transfer(&_srcImg, &_srcGpu, 1.f, _expressionStream, nullptr);
        if (nvErr!=NVCV_SUCCESS) {
            UtilityFunctions::print("failed to transfer image to gpu");
            return;
        }

        // run facial expression feature
        nvErr = NvAR_Run(_expressionFeature);
        if (nvErr!=NVCV_SUCCESS) {
            UtilityFunctions::print("failed to run facial expression feature");
            return;
        }

        normalizeExpressionsWeights();
    }

    // additional process logic
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
  for (size_t i = 0; i < _expressionOutputBboxes.num_boxes; ++i) {
    const auto& box = _expressionOutputBboxes.boxes[i];
    bboxesStr += "("+ UtilityFunctions::str(box.x) + ", " 
                    + UtilityFunctions::str(box.y) + ", " 
                    + UtilityFunctions::str(box.width) + ", " 
                    + UtilityFunctions::str(box.height) + ")";
    if (i < _expressionOutputBboxes.num_boxes - 1) {
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
  for (size_t i = 0; i < _expressionOutputBboxes.num_boxes; ++i) {
    boxes.push_back(bounding_box_to_dict(_expressionOutputBboxes.boxes[i]));
  }
  return boxes;
}

void ExpressionTrack::start_processing_thread() {
    continue_processing = true;
    processing_thread = std::thread(&ExpressionTrack::processing_loop, this);
}

void ExpressionTrack::processing_loop() {
  while (continue_processing) {
    if (_vidIn.read(_processingFrame)) {
      {
        std::lock_guard<std::mutex> lock(processing_mutex);
        // copy async frame data to _ocvSrcImg
        _processingFrame.copyTo(_ocvSrcImg); 
      }
    } else {
      UtilityFunctions::print("failed to read frame");
    }
  }
}

void ExpressionTrack::normalizeExpressionsWeights() {
  assert(_expressions.size() == _exprCount);
  assert(_expressionScale.size() == _exprCount);
  assert(_expressionZeroPoint.size() == _exprCount);

  for (size_t i = 0; i < _exprCount; i++) {
    float tempExpr = _expressions[i];
    // Normalize expression based on zero point and scale
    _expressions[i] = 1.0f - std::pow(1.0f - (std::max(_expressions[i] - _expressionZeroPoint[i], 0.0f) * _expressionScale[i]),
                                      _expressionExponent[i]);
    // Blend with the previous value using a global parameter
    _expressions[i] = _globalExpressionParam * _expressions[i] + (1.0f - _globalExpressionParam) * tempExpr;
  }
}