/*
 * Copyright (C) 2017 Martin Lambers <marlam@marlam.de>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "webcam-head-tracker.hpp"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>

/* Helpers for timing */

class timer
{
public:
    std::chrono::steady_clock::time_point t;
    void setNow() { t = std::chrono::steady_clock::now(); }
};

float duration(const timer& start, const timer& end)
{
    std::chrono::duration<long long, std::micro> time_span
        = std::chrono::duration_cast<std::chrono::duration<long long, std::micro>>(end.t - start.t);
    return time_span.count() / 1e3f;
}

/* Double Exponential Smoothing
 * This implements double exponential smoothing-based prediction
 * as described in Sec. 2 of "Double Exponential Smoothing: An alternative to
 * Kalman Filter-Based Predictive Tracking" by Joseph J. LaViola Jr.
 */

class DoubleExponentialSmoothing
{
private:
    bool _isInitialized;
    double _lastVecS[3], _lastVecS2[3];
    double _lastQuatS[4], _lastQuatS2[4];

    inline void copy3(double* result, const double* value)
    {
        result[0] = value[0];
        result[1] = value[1];
        result[2] = value[2];
    }

    inline void copy4(double* result, const double* value)
    {
        result[0] = value[0];
        result[1] = value[1];
        result[2] = value[2];
        result[3] = value[3];
    }

    inline double mix(double alpha, double x, double y)
    {
        return alpha * y + (1.0 - alpha) * x;
    }

    inline void mix3(double* result, double alpha, const double* v, const double* w)
    {
        result[0] = mix(alpha, v[0], w[0]);
        result[1] = mix(alpha, v[1], w[1]);
        result[2] = mix(alpha, v[2], w[2]);
    }

    inline void mix4(double* result, double alpha, const double* v, const double* w)
    {
        result[0] = mix(alpha, v[0], w[0]);
        result[1] = mix(alpha, v[1], w[1]);
        result[2] = mix(alpha, v[2], w[2]);
        result[3] = mix(alpha, v[3], w[3]);
    }

    inline double dot4(const double* v, const double* w)
    {
        return (v[0] * w[0] + v[1] * w[1] + v[2] * w[2] + v[3] * w[3]);
    }

    inline void normalize4(double* v)
    {
        double s = std::sqrt(dot4(v, v));
        v[0] /= s;
        v[1] /= s;
        v[2] /= s;
        v[3] /= s;
    }

    inline void slerp(double* result, double alpha, const double* q, const double* r)
    {
        double w[4] = { r[0], r[1], r[2], r[3] };
        double cosHalfAngle = dot4(q, r);
        if (cosHalfAngle < 0.0) {
            // quat(x, y, z, w) and quat(-x, -y, -z, -w) represent the same rotation
            w[0] = -w[0]; w[1] = -w[1]; w[2] = -w[2]; w[3] = -w[3];
            cosHalfAngle = -cosHalfAngle;
        }
        double tmpQ, tmpW;
        if (std::fabs(cosHalfAngle) >= 1.0) {
            // angle is zero => rotations are identical
            tmpQ = 1.0;
            tmpW = 0.0;
        } else {
            double halfAngle = acos(cosHalfAngle);
            double sinHalfAngle = sqrt(1.0 - cosHalfAngle * cosHalfAngle);
            if (std::fabs(sinHalfAngle) < 0.001) {
                // angle is 180 degrees => result is not clear
                tmpQ = 0.5;
                tmpW = 0.5;
            } else {
                tmpQ = std::sin((1.0 - alpha) * halfAngle) / sinHalfAngle;
                tmpW = sin(alpha * halfAngle) / sinHalfAngle;
            }
        }
        result[0] = q[0] * tmpQ + w[0] * tmpW;
        result[1] = q[1] * tmpQ + w[1] * tmpW;
        result[2] = q[2] * tmpQ + w[2] * tmpW;
        result[3] = q[3] * tmpQ + w[3] * tmpW;
    }

public:
    DoubleExponentialSmoothing() : _isInitialized(false) {}

    void step(const double* vec, const double* quat,
            double alpha, double tau, double* estimatedVec, double* estimatedQuat)
    {
        if (!_isInitialized) {
            copy3(_lastVecS, vec);
            copy3(_lastVecS2, vec);
            copy4(_lastQuatS, quat);
            copy4(_lastQuatS2, quat);
            _isInitialized = true;
        }
        double vecS[3], vecS2[3], quatS[4], quatS2[4];
        // Eq. (1), (2)
        mix3(vecS, alpha, _lastVecS, vec);
        mix3(vecS2, alpha, _lastVecS2, vecS);
        mix4(quatS, alpha, _lastQuatS, quat);
        mix4(quatS2, alpha, _lastQuatS2, quatS);
        copy3(_lastVecS, vecS);
        copy3(_lastVecS2, vecS2);
        copy4(_lastQuatS, quatS);
        copy4(_lastQuatS2, quatS2);
        // Eq. (6) for floor(tau) and ceil(tau)
        double floorTau = std::floor(tau);
        double betaFloorTau = 2.0 + alpha * floorTau / (1.0 - alpha);
        double estimatedVecFloorTau[3], estimatedQuatFloorTau[4];
        mix3(estimatedVecFloorTau, betaFloorTau, vecS2, vecS);
        mix4(estimatedQuatFloorTau, betaFloorTau, quatS2, quatS);
        normalize4(estimatedQuatFloorTau);
        double ceilTau = std::ceil(tau);
        double betaCeilTau = 2.0 + alpha * ceilTau / (1.0 - alpha);
        double estimatedVecCeilTau[3], estimatedQuatCeilTau[4];
        mix3(estimatedVecCeilTau, betaCeilTau, vecS2, vecS);
        mix4(estimatedQuatCeilTau, betaCeilTau, quatS2, quatS);
        normalize4(estimatedQuatCeilTau);
        // mix results for floor(tau) and ceil(tau)
        mix3(estimatedVec, tau - floorTau, estimatedVecCeilTau, estimatedVecFloorTau);
        slerp(estimatedQuat, tau - floorTau, estimatedQuatCeilTau, estimatedQuatFloorTau);
    }
};

/* WebcamHeadTracker */

WebcamHeadTracker::WebcamHeadTracker(unsigned int debugOptions) :
    _debugOptions(debugOptions),
    _isReady(false),
    _capture(NULL),
    _frame(NULL),
    _w(0), _h(0),
    _fps(0.0f),
    _fx(0.0f), _fy(0.0f),
    _cx(0.0f), _cy(0.0f),
    _k1(0.0f), _k2(0.0f), _p1(0.0f), _p2(0.0f), _k3(0.0f),
    _faceCascade(NULL),
    _faceModel(NULL),
    _filter(Filter_Double_Exponential),
    _kalmanFilter(NULL),
    _despFilter(NULL),
    _headPosition { 0.0f, 0.0f, 0.5f },
    _headOrientation { 0.0f, 0.0f, 0.0f, 0.0f }
{
}

WebcamHeadTracker::~WebcamHeadTracker()
{
    delete _capture;
    delete _frame;
    delete _faceCascade;
    delete _faceModel;
    delete _kalmanFilter;
    delete _despFilter;
}

bool WebcamHeadTracker::initWebcam()
{
    if (_capture)
        return isReady();
    _capture = new cv::VideoCapture(0);
    if (_capture && _capture->isOpened()) {
        _capture->set(CV_CAP_PROP_FRAME_WIDTH, 640);
        _capture->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
        _frame = new cv::Mat;
        _w = _capture->get(CV_CAP_PROP_FRAME_WIDTH);
        _h = _capture->get(CV_CAP_PROP_FRAME_HEIGHT);
        _fps = _capture->get(CV_CAP_PROP_FPS);
        if (_fps <= 0.0f)
            _fps = 30.0f;
        const char* intrinsics;
        float fx, fy, cx, cy;
        if ((intrinsics = std::getenv("WEBCAM_INTRINSIC_PARAMETERS"))
                && std::sscanf(intrinsics, "%g,%g,%g,%g", &fx, &fy, &cx, &cy) == 4) {
            _fx = fx;
            _fy = fy;
            _cx = cx;
            _cy = cy;
        } else {
            _fx = 0.9f * _w;
            _fy = _fx;
            _cx = _w / 2.0f;
            _cy = _h / 2.0f;
        }
        const char* distCoeffs;
        float k1, k2, p1, p2, k3;
        if ((distCoeffs = std::getenv("WEBCAM_DISTORTION_COEFFICIENTS"))
                && std::sscanf(distCoeffs, "%g,%g,%g,%g,%g", &k1, &k2, &p1, &p2, &k3) == 5) {
            _k1 = k1;
            _k2 = k2;
            _p1 = p1;
            _p2 = p2;
            _k3 = k3;
        }
        return true;
    } else {
        return false;
    }
}

#define STRINGIFY(s) STRINGIFY_HELPER(s)
#define STRINGIFY_HELPER(s) #s

const char* WebcamHeadTracker::filePathFrontalFaceXml()
{
#ifdef HAARCASCADE_FRONTALFACE_ALT_XML
    static const char s[] = STRINGIFY(HAARCASCADE_FRONTALFACE_ALT_XML);
#else
    static const char s[] = "";
#endif
    return s;
}

const char* WebcamHeadTracker::filePathFaceLandmarksDat()
{
#ifdef SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT
    static const char s[] = STRINGIFY(SHAPE_PREDICTOR_68_FACE_LANDMARKS_DAT);
#else
    static const char s[] = "";
#endif
    return s;
}

bool WebcamHeadTracker::initPoseEstimator(const char* frontalFaceXml, const char* faceLandmarksDat)
{
    if (isReady())
        return true;

    _faceCascade = new cv::CascadeClassifier;
    if (!_faceCascade->load(frontalFaceXml)) {
        delete _faceCascade;
        _faceCascade = NULL;
        return false;
    }

    _faceModel = new dlib::shape_predictor;
    try {
        dlib::deserialize(faceLandmarksDat) >> *_faceModel;
    }
    catch (std::exception& e) {
        delete _faceCascade;
        _faceCascade = NULL;
        delete _faceModel;
        _faceModel = NULL;
        return false;
    }

    // See http://docs.opencv.org/trunk/dc/d2c/tutorial_real_time_pose.html
    // for information on this!
    _kalmanFilter = new cv::KalmanFilter;
    _kalmanFilter->init(18, 6, 0, CV_64F);
    cv::setIdentity(_kalmanFilter->processNoiseCov, cv::Scalar::all(1e-3));
    cv::setIdentity(_kalmanFilter->measurementNoiseCov, cv::Scalar::all(1e-1));
    cv::setIdentity(_kalmanFilter->errorCovPost, cv::Scalar::all(1));
    float dt = 1.0f / _fps;
    _kalmanFilter->transitionMatrix.at<double>(0, 3) = dt;
    _kalmanFilter->transitionMatrix.at<double>(1, 4) = dt;
    _kalmanFilter->transitionMatrix.at<double>(2, 5) = dt;
    _kalmanFilter->transitionMatrix.at<double>(3, 6) = dt;
    _kalmanFilter->transitionMatrix.at<double>(4, 7) = dt;
    _kalmanFilter->transitionMatrix.at<double>(5, 8) = dt;
    _kalmanFilter->transitionMatrix.at<double>(0, 6) = 0.5f * dt * dt;
    _kalmanFilter->transitionMatrix.at<double>(1, 7) = 0.5f * dt * dt;
    _kalmanFilter->transitionMatrix.at<double>(2, 8) = 0.5f * dt * dt;
    _kalmanFilter->transitionMatrix.at<double>(9, 12) = dt;
    _kalmanFilter->transitionMatrix.at<double>(10, 13) = dt;
    _kalmanFilter->transitionMatrix.at<double>(11, 14) = dt;
    _kalmanFilter->transitionMatrix.at<double>(12, 15) = dt;
    _kalmanFilter->transitionMatrix.at<double>(13, 16) = dt;
    _kalmanFilter->transitionMatrix.at<double>(14, 17) = dt;
    _kalmanFilter->transitionMatrix.at<double>(9, 15) = 0.5f * dt * dt;
    _kalmanFilter->transitionMatrix.at<double>(10, 16) = 0.5f * dt * dt;
    _kalmanFilter->transitionMatrix.at<double>(11, 17) = 0.5f * dt * dt;
    _kalmanFilter->measurementMatrix.at<double>(0, 0) = 1;
    _kalmanFilter->measurementMatrix.at<double>(1, 1) = 1;
    _kalmanFilter->measurementMatrix.at<double>(2, 2) = 1;
    _kalmanFilter->measurementMatrix.at<double>(3, 9) = 1;
    _kalmanFilter->measurementMatrix.at<double>(4, 10) = 1;
    _kalmanFilter->measurementMatrix.at<double>(5, 11) = 1;

    _despFilter = new DoubleExponentialSmoothing;

    _isReady = true;
    return true;
}

void WebcamHeadTracker::setFocalLengthsInPixels(float fx, float fy)
{
    _fx = fx;
    _fy = fy;
}

void WebcamHeadTracker::setPrincipalPointInPixels(float cx, float cy)
{
    _cx = cx;
    _cy = cy;
}

void WebcamHeadTracker::setDistortionCoefficients(float k1, float k2, float p1, float p2, float k3)
{
    _k1 = k1;
    _k2 = k2;
    _p1 = p1;
    _p2 = p2;
    _k3 = k3;
}

void WebcamHeadTracker::setFilter(enum Filter filter)
{
    _filter = filter;
}

void WebcamHeadTracker::getNewFrame()
{
    timer t0, t1;
    t0.setNow();
    *_capture >> *_frame;
    t1.setNow();
    if (_debugOptions & Debug_Timing) {
        fprintf(stderr, "WHT: acquiring webcam frame:  %4.1f ms\n", duration(t0, t1));
    }
}

static void rodriguesToQuaternion(const double* r, double* q)
{
    // Note: the OpenCV use of Rodrigues rotation vectors seems to differ from
    // other uses. They multiply the unit rotation axis with the rotation angle
    // instead of with the tangens of half the angle. This is all crap. Everyone
    // should always use quaternions ;)
    double angle = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    double axis[3] = { r[0] / angle, r[1] / angle, r[2] / angle};
    double sinHalfAngle = std::sin(angle / 2.0f);
    q[0] = axis[0] * sinHalfAngle;
    q[1] = axis[1] * sinHalfAngle;
    q[2] = axis[2] * sinHalfAngle;
    q[3] = std::cos(angle / 2.0f);
}

static void quaternionToRodrigues(const double* q, double* r)
{
    double halfAngle = std::acos(q[3]);
    double sinHalfAngle = std::sin(halfAngle);
    double angle = halfAngle * 2.0;
    double factor = angle / sinHalfAngle;
    r[0] = factor * q[0];
    r[1] = factor * q[1];
    r[2] = factor * q[2];
}

static void quaternionToEuler(const double* q, double* euler)
{
    double singularityTest = q[0] * q[1] + q[2] * q[3];
    if (singularityTest > 0.4999) {
        // north pole
        euler[0] = 2.0 * std::atan2(q[0], q[3]);
        euler[1] = M_PI_2;
        euler[2] = 0.0;
    } else if (singularityTest < -0.4999) {
        // south pole
        euler[0] = -2.0 * std::atan2(q[0], q[3]);
        euler[1] = -M_PI_2;
        euler[2] = 0.0f;
    } else {
        euler[0] = std::atan2(2.0 * (q[3] * q[0] + q[1] * q[2]), 1.0 - 2.0 * (q[0] * q[0] + q[1] * q[1]));
        euler[1] = std::asin(2.0 * (q[3] * q[1] - q[0] * q[2]));
        euler[2] = std::atan2(2.0 * (q[3] * q[2] + q[0] * q[1]), 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
    }
}

static void eulerToQuaternion(const double* euler, double* q)
{
    double x2 = euler[0] / 2.0;
    double y2 = euler[1] / 2.0;
    double z2 = euler[2] / 2.0;
    double sx2 = std::sin(x2);
    double cx2 = std::cos(x2);
    double sy2 = std::sin(y2);
    double cy2 = std::cos(y2);
    double sz2 = std::sin(z2);
    double cz2 = std::cos(z2);
    q[0] = sx2 * cy2 * cz2 - cx2 * sy2 * sz2;
    q[1] = cx2 * sy2 * cz2 + sx2 * cy2 * sz2;
    q[2] = cx2 * cy2 * sz2 - sx2 * sy2 * cz2;
    q[3] = cx2 * cy2 * cz2 + sx2 * sy2 * sz2;
}

bool WebcamHeadTracker::computeHeadPose()
{
    if (!_faceCascade)
        return false;

    timer t0, t1, t2, t3, t4;

    /* Face detection */
    t0.setNow();
    const int minFaceSize = 80;
    std::vector<cv::Rect> faces;
    _faceCascade->detectMultiScale(*_frame, faces, 1.1, 2,
            CV_HAAR_SCALE_IMAGE | CV_HAAR_FIND_BIGGEST_OBJECT,
            cv::Size(minFaceSize, minFaceSize));
    if (faces.size() < 1)
        return false;
    cv::Rect faceRect = faces[0];
    t1.setNow();

    /* Face landmark detection */
    dlib::cv_image<dlib::bgr_pixel> dlibFrame(*_frame); // does not copy data
    dlib::rectangle dlibRect(faceRect.x, faceRect.y,
            faceRect.x + faceRect.width - 1, faceRect.y + faceRect.height - 1);
    dlib::full_object_detection shape = (*_faceModel)(dlibFrame, dlibRect);
    if (shape.num_parts() != 68)
        return false;
    std::vector<cv::Point2f> landmarks;
    landmarks.resize(68);
    for (int i = 0; i < 68; i++) {
        dlib::point p = shape.part(i);
        landmarks[i] = cv::Point(p.x(), p.y());
    }
    t2.setNow();

    /* Extract a subset of landmarks for which we have good guesses for
     * average positions from <https://en.wikipedia.org/wiki/Human_head>.
     * Everything must be in mm to be consistent with OpenCV assumptions.
     * We use landmarks that tend not to change too much with varying
     * facial expressions. */
    const cv::Point3f landmarkLeftEctocanthi  (-60.0f,   0.0f,   0.0f);
    const cv::Point3f landmarkRightEctocanthi (+60.0f,   0.0f,   0.0f);
    const cv::Point3f landmarkSellion         (  0.0f,   5.0f, -20.0f);
    const cv::Point3f landmarkSubnasale       (  0.0f, -42.0f, -30.0f);
    const cv::Point3f landmarkStomion         (  0.0f, -67.0f, -32.0f);
    const cv::Point3f landmarkLeftTragion     (-70.0f,   0.0f,  99.9f);
    const cv::Point3f landmarkRightTragion    (+70.0f,   0.0f,  99.9f);
    const std::vector<cv::Point3f> modelLandmarks( {
            landmarkLeftEctocanthi,
            landmarkLeftEctocanthi,
            landmarkRightEctocanthi,
            landmarkRightEctocanthi,
            landmarkSellion,
            landmarkSubnasale,
            landmarkStomion,
            landmarkLeftTragion,
            landmarkRightTragion
            });
    const int landmarkLeftEctocanthiIndex = 36;
    const int landmarkRightEctocanthiIndex = 45;
    const int landmarkSellionIndex = 27;
    const int landmarkSubnasaleIndex = 33;
    const int landmarkStomionIndex = 51;
    const int landmarkLeftTragionIndex = 0;
    const int landmarkRightTragionIndex = 16;
    const std::vector<cv::Point2f> imageLandmarks( {
            landmarks[landmarkLeftEctocanthiIndex],
            landmarks[landmarkLeftEctocanthiIndex],
            landmarks[landmarkRightEctocanthiIndex],
            landmarks[landmarkRightEctocanthiIndex],
            landmarks[landmarkSellionIndex],
            landmarks[landmarkSubnasaleIndex],
            landmarks[landmarkStomionIndex],
            landmarks[landmarkLeftTragionIndex],
            landmarks[landmarkRightTragionIndex]
            });
    cv::Matx33f cameraMatrix;
    cameraMatrix(0, 0) = _fx;
    cameraMatrix(0, 1) = 0.0f;
    cameraMatrix(0, 2) = _cx;
    cameraMatrix(1, 0) = 0.0f;
    cameraMatrix(1, 1) = _fy;
    cameraMatrix(1, 2) = _cy;
    cameraMatrix(2, 0) = 0.0f;
    cameraMatrix(2, 1) = 0.0f;
    cameraMatrix(2, 2) = 1.0f;
    cv::Mat distCoeffs(1, 5, CV_32F);
    distCoeffs.at<float>(0) = _k1;
    distCoeffs.at<float>(1) = _k2;
    distCoeffs.at<float>(2) = _p1;
    distCoeffs.at<float>(3) = _p2;
    distCoeffs.at<float>(4) = _k3;
    cv::Mat rvec(1, 3, CV_64F);
    rvec.at<double>(0) = M_PI; // 180 deg around x axis: null rotation in OpenCV orientation
    rvec.at<double>(1) = 0.0f;
    rvec.at<double>(2) = 0.0f;
    cv::Mat tvec(1, 3, CV_64F);
    tvec.at<double>(0) = 0.0f;
    tvec.at<double>(1) = 0.0f;
    tvec.at<double>(2) = 500.0f;
    // in my tests, using the CV_P3P solver with 4 points was less stable than using the iterative solver with 7
    //cv::solvePnP(modelLandmarks, imageLandmarks, cameraMatrix, distCoeffs, rvec, tvec, false, CV_P3P);
    cv::solvePnP(modelLandmarks, imageLandmarks, cameraMatrix, distCoeffs, rvec, tvec, true, CV_ITERATIVE);
    double observedVec[3] = { tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2) };
    double observedQuat[4];
    rodriguesToQuaternion(&(rvec.at<double>(0)), observedQuat);
    t3.setNow();

    /* Feed the new measurement to the filter and save result */
    double estimatedVec[3] = { 0, 0, 0 };
    double estimatedQuat[4] = { 0, 0, 0, 0 };
    switch (_filter) {
    case Filter_None:
        {
            estimatedVec[0] = observedVec[0];
            estimatedVec[1] = observedVec[1];
            estimatedVec[2] = observedVec[2];
            estimatedQuat[0] = observedQuat[0];
            estimatedQuat[1] = observedQuat[1];
            estimatedQuat[2] = observedQuat[2];
            estimatedQuat[3] = observedQuat[3];
        }
        break;
    case Filter_Kalman:
        {
            double observedEulerAngles[3];
            quaternionToEuler(observedQuat, observedEulerAngles);
            // See http://docs.opencv.org/trunk/dc/d2c/tutorial_real_time_pose.html
            // for information on this!
            cv::Mat measurement(6, 1, CV_64F);
            measurement.at<double>(0) = observedVec[0];
            measurement.at<double>(1) = observedVec[1];
            measurement.at<double>(2) = observedVec[2];
            measurement.at<double>(3) = observedEulerAngles[0];
            measurement.at<double>(4) = observedEulerAngles[1];
            measurement.at<double>(5) = observedEulerAngles[2];
            cv::Mat prediction = _kalmanFilter->predict();
            cv::Mat estimation = _kalmanFilter->correct(measurement);
            estimatedVec[0] = estimation.at<double>(0);
            estimatedVec[1] = estimation.at<double>(1);
            estimatedVec[2] = estimation.at<double>(2);
            eulerToQuaternion(&(estimation.at<double>(9)), estimatedQuat);
        }
        break;
    case Filter_Double_Exponential:
        {
            _despFilter->step(observedVec, observedQuat,
                    0.2, 0.7,
                    estimatedVec, estimatedQuat);
        }
        break;
    }
    t4.setNow();

    /* Convert the internal representation to the external representation */
    // convert position
    _headPosition[0] = -estimatedVec[0] / 1000.0;
    _headPosition[1] = -estimatedVec[1] / 1000.0;
    _headPosition[2] =  estimatedVec[2] / 1000.0;
    // convert orientation (rotate 180 deg around x)
    _headOrientation[0] =  estimatedQuat[3];
    _headOrientation[1] = -estimatedQuat[2];
    _headOrientation[2] =  estimatedQuat[1];
    _headOrientation[3] = -estimatedQuat[0];

    /* Debug output */
    if (_debugOptions & Debug_Timing) {
        fprintf(stderr, "WHT: face detection:          %4.1f ms\n", duration(t0, t1));
        fprintf(stderr, "WHT: face landmark detection: %4.1f ms\n", duration(t1, t2));
        fprintf(stderr, "WHT: face model matching:     %4.1f ms\n", duration(t2, t3));
        fprintf(stderr, "WHT: filtering:               %4.1f ms\n", duration(t4, t3));
    }
    if (_debugOptions & Debug_Window) {
        // render face rectangle
        cv::rectangle(*_frame, faceRect, cv::Scalar(0, 0, 255));
        // render face model
        for (int i = 1; i <= 16; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 18; i <= 21; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 23; i <= 26; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 28; i <= 30; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 31; i <= 35; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        cv::line(*_frame, landmarks[30], landmarks[35], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 37; i <= 41; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        cv::line(*_frame, landmarks[36], landmarks[41], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 43; i <= 47; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        cv::line(*_frame, landmarks[42], landmarks[47], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 49; i <= 59; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        cv::line(*_frame, landmarks[48], landmarks[49], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 61; i <= 67; i++)
            cv::line(*_frame, landmarks[i - 1], landmarks[i], cv::Scalar(0, 255, 0), 1, 1, 0);
        cv::line(*_frame, landmarks[60], landmarks[67], cv::Scalar(0, 255, 0), 1, 1, 0);
        for (int i = 0; i < 68; i++)
            cv::circle(*_frame, landmarks[i], 2.5f, cv::Scalar(0, 0, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkLeftEctocanthiIndex],  3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkRightEctocanthiIndex], 3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkSellionIndex],         3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkSubnasaleIndex],       3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkStomionIndex],         3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkLeftTragionIndex],     3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        cv::circle(*_frame, landmarks[landmarkRightTragionIndex],    3.0f, cv::Scalar(255, 255, 255), 1, 1, 0);
        // render projected face model landmarks
        std::vector<cv::Point2f> projectedModelLandmarks;
        cv::projectPoints(modelLandmarks, rvec, tvec, cameraMatrix, distCoeffs, projectedModelLandmarks);
        cv::line(*_frame, projectedModelLandmarks[7], projectedModelLandmarks[0], cv::Scalar(255, 0, 0));
        cv::line(*_frame, projectedModelLandmarks[0], projectedModelLandmarks[4], cv::Scalar(255, 0, 0));
        cv::line(*_frame, projectedModelLandmarks[4], projectedModelLandmarks[2], cv::Scalar(255, 0, 0));
        cv::line(*_frame, projectedModelLandmarks[2], projectedModelLandmarks[8], cv::Scalar(255, 0, 0));
        cv::line(*_frame, projectedModelLandmarks[4], projectedModelLandmarks[5], cv::Scalar(255, 0, 0));
        cv::line(*_frame, projectedModelLandmarks[5], projectedModelLandmarks[6], cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[0], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[2], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[4], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[5], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[6], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[7], 3.0f, cv::Scalar(255, 0, 0));
        cv::circle(*_frame, projectedModelLandmarks[8], 3.0f, cv::Scalar(255, 0, 0));
        // render projected filtered model
        std::vector<cv::Point2f> projectedFilteredModelLandmarks;
        tvec.at<double>(0) = estimatedVec[0];
        tvec.at<double>(1) = estimatedVec[1];
        tvec.at<double>(2) = estimatedVec[2];
        quaternionToRodrigues(estimatedQuat, &(rvec.at<double>(0)));
        cv::projectPoints(modelLandmarks, rvec, tvec, cameraMatrix, distCoeffs, projectedFilteredModelLandmarks);
        cv::line(*_frame, projectedFilteredModelLandmarks[7], projectedFilteredModelLandmarks[0], cv::Scalar(255, 255, 0));
        cv::line(*_frame, projectedFilteredModelLandmarks[0], projectedFilteredModelLandmarks[4], cv::Scalar(255, 255, 0));
        cv::line(*_frame, projectedFilteredModelLandmarks[4], projectedFilteredModelLandmarks[2], cv::Scalar(255, 255, 0));
        cv::line(*_frame, projectedFilteredModelLandmarks[2], projectedFilteredModelLandmarks[8], cv::Scalar(255, 255, 0));
        cv::line(*_frame, projectedFilteredModelLandmarks[4], projectedFilteredModelLandmarks[5], cv::Scalar(255, 255, 0));
        cv::line(*_frame, projectedFilteredModelLandmarks[5], projectedFilteredModelLandmarks[6], cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[0], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[2], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[4], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[5], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[6], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[7], 3.0f, cv::Scalar(255, 255, 0));
        cv::circle(*_frame, projectedFilteredModelLandmarks[8], 3.0f, cv::Scalar(255, 255, 0));
        // show
        cv::imshow("webcam head tracker", *_frame);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q')
            _isReady = false;
        if (key == 'f')
            _filter = (_filter == Filter_None ? Filter_Kalman
                    : _filter == Filter_Kalman ? Filter_Double_Exponential
                    : Filter_None);
    }
    return true;
}

void WebcamHeadTracker::getHeadPosition(float* headPosition) const
{
    headPosition[0] = _headPosition[0];
    headPosition[1] = _headPosition[1];
    headPosition[2] = _headPosition[2];
}

void WebcamHeadTracker::getHeadOrientation(float* headOrientation) const
{
    headOrientation[0] = _headOrientation[0];
    headOrientation[1] = _headOrientation[1];
    headOrientation[2] = _headOrientation[2];
    headOrientation[3] = _headOrientation[3];
}
