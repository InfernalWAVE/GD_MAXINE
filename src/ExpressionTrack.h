#ifndef EXPRESSIONTRACK_H
#define EXPRESSIONTRACK_H

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <godot_cpp/classes/node.hpp>
#include <godot_cpp/classes/image.hpp>
#include <godot_cpp/classes/image_texture.hpp>
#include <godot_cpp/classes/array_mesh.hpp>

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
    class MyTimer {
        public:
            MyTimer()     { dt = dt.zero();                                      }  /**< Clear the duration to 0. */
            void start()  { t0 = std::chrono::high_resolution_clock::now();      }  /**< Start  the timer. */
            void pause()  { dt = std::chrono::high_resolution_clock::now() - t0; }  /**< Pause  the timer. */
            void resume() { t0 = std::chrono::high_resolution_clock::now() - dt; }  /**< Resume the timer. */
            void stop()   { pause();                                             }  /**< Stop   the timer. */
            double elapsedTimeFloat() const {
                return std::chrono::duration<double>(dt).count();
            } /**< Report the elapsed time as a float. */
        private:
            std::chrono::high_resolution_clock::time_point t0;
            std::chrono::high_resolution_clock::duration dt;
    };

    NvCV_Status nvErr;
    std::string modelPath;
    cv::VideoCapture _vidIn;

    NvCVImage _srcImg, _srcGpu;
    NvAR_FeatureHandle _featureHan{};

    CUstream _stream;
    unsigned _poseMode, _enableCheekPuff, _filtering, _exprCount, _landmarkCount;

    std::vector<NvAR_Rect> _outputBboxData;
    NvAR_BBoxes _outputBboxes;

    std::vector<NvAR_Point2f> _landmarks;
    std::vector<float> _expressions, _expressionZeroPoint, _expressionScale, _expressionExponent, _eigenvalues, _landmarkConfidence;

    struct Pose {
        NvAR_Quaternion rotation;
        NvAR_Vector3f translation;
        float* data() { return &rotation.x; }
        const float* data() const { return &rotation.x; }
    };

    Pose _pose;

    cv::Mat _ocvSrcImg;
    float _cameraIntrinsicParams[NUM_CAMERA_INTRINSIC_PARAMS];

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
};

}

#endif