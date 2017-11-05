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

#ifndef WEBCAM_HEAD_TRACKER_HPP
#define WEBCAM_HEAD_TRACKER_HPP

/*! \cond */
namespace cv {
class VideoCapture;
class Mat;
class CascadeClassifier;
class KalmanFilter;
}
namespace dlib {
class shape_predictor;
}
class DoubleExponentialSmoothing;
/*! \endcond */

/*!
 * \mainpage libwebcamheadtracker: Webcam-based head tracking
 *
 * Usage:
 * - Create a single instance of the \a WebcamHeadTracker class.
 * - Call \a WebcamHeadTracker::initWebcam() to initialize the webcam. If this fails, no usable
 *   webcam could be found.
 * - Call \a WebcamHeadTracker::initPoseEstimator() to initialize the head pose estimator. This
 *   requires two data files: `haarcascade_frontalface_alt.xml` from OpenCV
 *   and `shape_predictor_68_face_landmarks.dat` from dlib. If you call this
 *   function without arguments, then the library will try to find these files
 *   at the places where they were when the library was built. This works fine
 *   on development systems and on Linux(ish) systems, but if you deploy your
 *   application, you might want to bundle these files.
 * - While \a WebcamHeadTracker::isReady() returns true:
 *   - Acquire a new webcam frame with \a WebcamHeadTracker::getNewFrame().
 *   - Compute a new head pose with \a WebcamHeadTracker::computeHeadPose(). This may fail if no
 *     face could be detected in the webcam frame for some reason.
 *   - Get the latest known pose with \a WebcamHeadTracker::getHeadPosition() and
 *     \a WebcamHeadTracker::getHeadOrientation().
 */

/*!
 * \brief Webcam-based head tracker
 */
class WebcamHeadTracker
{
public:
    /*! \brief Debug options */
    enum DebugOption {
        /*! \brief Show a GUI window with the webcam frame and face detection results */
        Debug_Window = 1,
        /*! \brief Print timings for each step to `stderr`. */
        Debug_Timing = 2
    };

    /*! \brief Filter types */
    enum Filter {
        /*! \brief Do not apply any filtering (very strong jittering, but no additional delays) */
        Filter_None,
        /*! \brief Apply Kalman filter (smooth results, but long update delays on pose changes) */
        Filter_Kalman,
        /*! \brief Apply double exponential smoothing (smooth results, acceptable delays) */
        Filter_Double_Exponential
    };

    /*! \brief Constructor
     * \param debugOptions      Bitwise combination of \a DebugOption flags. */
    WebcamHeadTracker(unsigned int debugOptions = 0);
    /*! \brief Destructor */
    ~WebcamHeadTracker();

    /*! \brief Initialize the webcam
     *
     * Tries to use the first camera that OpenCV detects. If this fails, no usable webcam
     * is available. */
    bool initWebcam();

    /*! \brief Default path to `haarcascade_frontalface_alt.xml` (location at build time, if it was found) */
    static const char* filePathFrontalFaceXml();
    /*! \brief Default path to `shape_predictor_68_face_landmarks.dat` (location at build time, if it was found) */
    static const char* filePathFaceLandmarksDat();

    /*! \brief Initialize the pose estimator
     * \param frontalFaceXml    Full path to the file haarcascade_frontalface_alt.xml from OpenCV
     * \param faceLandmarksDat  Full path to the file shape_predictor_68_face_landmarks.dat from dlib
     *
     * This function loads the two data files from OpenCV and dlib. It returns false if this fails.
     * The default parameters should work fine on development systems and Linux(ish) systems, but
     * you might want to bundle these files with your application and use custom file paths as
     * parameters to this function. */
    bool initPoseEstimator(
            const char* frontalFaceXml = filePathFrontalFaceXml(),
            const char* faceLandmarksDat = filePathFaceLandmarksDat());

    /*! \brief Set intrinsic camera parameters: focal lengths
     * \param fx        Horizontal focal length
     * \param fy        Vertical focal length
     *
     * Calling this function is optional, the defaults should be ok.
     * To find correct values for your webcam, you need to calibrate it. See OpenCV documentation. */
    void setFocalLengthsInPixels(float fx, float fy);

    /*! \brief Set intrinsic camera parameters: principal points
     * \param cx        Horizontal principal point
     * \param cy        Vertical principal point
     *
     * Calling this function is optional, the defaults should be ok.
     * To find correct values for your webcam, you need to calibrate it. See OpenCV documentation. */
    void setPrincipalPointInPixels(float cx, float cy);

    /*! \brief Set camera distortion coefficients
     * \param k1        Parameter k1
     * \param k2        Parameter k2
     * \param p1        Parameter p1
     * \param p2        Parameter p2
     * \param k3        Parameter k3
     *
     * Calling this function is optional, the defaults should be ok.
     * To find correct values for your webcam, you need to calibrate it. See OpenCV documentation. */
    void setDistortionCoefficients(float k1, float k2, float p1, float p2, float k3 = 0.0f);

    /*! \brief Set the filter that is applied to the pose
     * \param filter    The filter
     *
     * The default is \a Filter_Double_Exponential.
     */
    void setFilter(enum Filter filter);

    /*! \brief Returns true if this tracker is ready to get a new frame and compute a new head pose
     *
     * This returns true once the tracker is successfully initialized.
     * When \a Debug_Window is set, this can change to false when the user presses 'ESC'. */
    bool isReady() const { return _isReady; }

    /*! \brief Get a new frame from the webcam. */
    void getNewFrame();

    /*! \brief Compute a new head pose.
     *
     * This works on the latest available webcam frame.
     * If head pose estimation fails for some reason (e.g. no face could be found),
     * then this function returns false. */
    bool computeHeadPose();

    /*! \brief Get the last known head position.
     *
     * Returns the last known head orientation as a vector (x, y, z).
     *
     * The head reference point is the center between the left and right eyes.
     * The coordinate system is the usual system in computer graphics:
     * y points upwards, x points to the right, and the user looks along -z (towards the webcam).
     */
    void getHeadPosition(float* headPosition) const;

    /*! \brief Get the last known head orientation.
     *
     * Returns the last known head orientation as a quaternion (x, y, z, w).
     *
     * The head reference point is the center between the left and right eyes.
     * The coordinate system is the usual system in computer graphics:
     * y points upwards, x points to the right, and the user looks along -z (towards the webcam).
     */
    void getHeadOrientation(float* headOrientation) const;

private:
    unsigned int _debugOptions;
    bool _isReady;
    // the webcam
    cv::VideoCapture* _capture;
    cv::Mat* _frame;
    cv::Mat* _frameGray;
    // frame dimensions
    int _w, _h;
    // frame rate
    float _fps;
    // camera intrinsics
    float _fx, _fy;
    float _cx, _cy;
    float _k1, _k2, _p1, _p2, _k3;
    // classifiers and detectors
    cv::CascadeClassifier* _faceCascade;
    dlib::shape_predictor* _faceModel;
    // filters
    enum Filter _filter;
    cv::KalmanFilter* _kalmanFilter;
    DoubleExponentialSmoothing* _despFilter;
    // last known head pose
    float _headPosition[3];
    float _headOrientation[4];

};

#endif
