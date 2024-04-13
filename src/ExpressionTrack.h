/****************************************************************
 * Copyright (c) 2024 Ryan Powell
 *
 * This software is released under the MIT License.
 * See the LICENSE file in the project root for more information.
 ****************************************************************/

#ifndef EXPRESSIONTRACK_H
#define EXPRESSIONTRACK_H

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

#include <godot_cpp/classes/node.hpp>

#include "nvAR.h"
#include "nvAR_defs.h"
#include "nvCVOpenCV.h"
#include <opencv2/opencv.hpp>

#define NUM_CAMERA_INTRINSIC_PARAMS 3
#define NUM_LANDMARKS 126
#define FPS_PRECISION 1000

namespace godot {

class ExpressionTrack: public Node {
    GDCLASS(ExpressionTrack, Node)

private:
    NvCV_Status nvErr;
    std::string modelPath;
    cv::VideoCapture _vidIn;

    NvCVImage _srcImg, _srcGpu;
    NvAR_FeatureHandle _expressionFeature{}, _bodyFeature{};

    CUstream _expressionStream, _bodyStream;
    unsigned _poseMode, _enableCheekPuff, _expressionFiltering, _exprCount, _landmarkCount, _bodyTrackMode, _bodyFiltering, _numKeyPoints;
    bool _useCudaGraph;

    std::vector<NvAR_Point3f> _referencePose;
    std::vector<NvAR_Rect> _expressionOutputBboxData, _bodyOutputBboxData;
    NvAR_BBoxes _expressionOutputBboxes{}, _bodyOutputBboxes{};

    std::vector<NvAR_Point2f> _landmarks;
    std::vector<float> _expressions, _expressionZeroPoint, _expressionScale, _expressionExponent, _eigenvalues, _landmarkConfidence;

    std::vector<NvAR_Point2f> _keypoints;
    std::vector<float> _keypoints_confidence;
    std::vector<NvAR_Point3f> _keypoints3D;
    std::vector<NvAR_Quaternion> _jointAngles;
    std::vector<float> _bodyOutputBboxConfData;

    struct Pose {
        NvAR_Quaternion rotation;
        NvAR_Vector3f translation;
        float* data() { return &rotation.x; }
        const float* data() const { return &rotation.x; }
    };

    Pose _pose;

    cv::Mat _ocvSrcImg, _processingFrame;
    float _cameraIntrinsicParams[NUM_CAMERA_INTRINSIC_PARAMS];
    float _globalExpressionParam;

    std::thread processing_thread;
    std::atomic<bool> continue_processing{false};
    std::mutex processing_mutex;

protected:
    static void _bind_methods();


public:
    ExpressionTrack();
    ~ExpressionTrack();
    void _ready() override;
    void _process(double delta) override;

    void printPoseRotation();
    void printExpressionCoefficients();
    void printLandmarkLocations();
    void printBoundingBoxes();
    void printLandmarkConfidence();
    void printPoseTranslation();
    void printCapture();

    Array get_landmarks() const;
    int get_landmark_count() const;
    int get_expression_count() const;
    Array get_expressions() const;
    Array get_landmark_confidence() const;
    Vector3 get_pose_translation() const;
    Quaternion get_pose_rotation() const;
    Transform3D get_pose_transform() const;
    Dictionary bounding_box_to_dict(const NvAR_Rect& box) const;
    Array get_bounding_boxes() const;

    void set_landmarks(const Array& p_value) {};
    void set_landmark_count(int p_value) {};
    void set_expression_count(int p_value) {};
    void set_expressions(const Array& p_value) {};
    void set_landmark_confidence(const Array& p_value) {};
    void set_pose_rotation(const Quaternion& p_value) {};
    void set_pose_translation(const Vector3& p_value) {};
    void set_pose_transform(const Transform3D& p_value) {};
    void set_bounding_boxes(const Array& p_value) {};

    void processing_loop();
    void start_processing_thread();

    void normalizeExpressionsWeights();
};

}

#endif