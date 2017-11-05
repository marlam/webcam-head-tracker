# A library for webcam-based head tracking

## Introduction

This library provides head tracking based on the webcam typically found in
laptop computers or on top of desktop monitors.

Head tracking means that you get a position and orientation of the user's head
(6DOF tracking).

Under good conditions, you can get new values at the typical webcam frame
rate of 30 frames per second.

## How do I use this?

- Create an instance of the `WebcamHeadTracker` class.
- Call `WebcamHeadTracker::initWebcam()` to initialize the webcam.
- Call `WebcamHeadTracker::initPoseEstimator()` to initialize the head pose estimator.
- While `WebcamHeadTracker::isReady()` returns true:
  - Acquire a new webcam frame with `WebcamHeadTracker::getNewFrame()`.
  - Compute a new head pose with `WebcamHeadTracker::computeHeadPose()`.
  - Get the latest known pose with `WebcamHeadTracker::getHeadPosition()`
    and `WebcamHeadTracker::getHeadOrientation()`.

## How does it work?

We use mainly [OpenCV](https://opencv.org/) and [dlib](http//dlib.net/) functionality:
- Acquire a webcam video frame with OpenCV
- Detect the main face in it using the
  [OpenCV haar feature-based cascade classifier](http://docs.opencv.org/3.2.0/d5/d54/group__objdetect.html)
  (this is much faster than the dlib face detector)
- Find face landmarks using the
  [dlib implementation of Millisecond Face Alignment](http://blog.dlib.net/2014/08/real-time-face-pose-estimation.html)
- Extract a subset of face landmarks that fulfill two requirements:
  - They are reasonably robust with regard to changes in facial expressions
  - We can estimate 3D positions of them on a model of the
    [average human head](https://en.wikipedia.org/wiki/Human_head)
- Use the correspondences of the 3D average human head points and the extracted
  2D image landmarks to estimate the head pose, using OpenCV
  [solvePnP](http://docs.opencv.org/3.2.0/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- Filter the estimated head pose to remove jitter and predict the pose at a
  short time step into the future, to make the pose data usable in interactive
  systems. We offer both a [Kalman filter for object tracking](http://docs.opencv.org/trunk/dc/d2c/tutorial_real_time_pose.html)
  and a [double exponential smoothing-based prediction filter](http://dl.acm.org/citation.cfm?id=769976)
  for this purpose. The latter is the default.

These ideas were borrowed from various sources, including
[screenReality](https://github.com/agirault/screenReality),
[eyeLike](https://github.com/trishume/eyeLike), 
[gazr](https://github.com/severin-lemaignan/gazr),
[this OpenCV tutorial](http://docs.opencv.org/trunk/dc/d2c/tutorial_real_time_pose.html),
and [this paper](http://dl.acm.org/citation.cfm?id=769976).
We ended up using an approach similar to gazr, but faster, independent of ROS,
and with better filtering.

## Limitations

- Both the face detector and the face landmark detector work best for frontal
  faces. They fail early if you tilt your head too far.
- The pose estimation before filtering is very noisy, so extensive filtering is
  required, which leads to swimming artefacts. With a less noisy pose
  estimation, we could tweak filter parameters to reduce this effect...
- This is all just approximation, do not expect the resulting values to
  guarantee reasonable error bounds.
- The library uses crude guesses for the camera intrinsic parameters and
  distortion coefficients. This seems to work surprisingly well most of the
  time. However, you can also properly calibrate your webcam and use the
  correct values (see comments in `webcam-snapshot.cpp`)
- Under bad lighting, the webcam will have trouble delivering reasonably good
  images, and the detectors/estimators will have trouble with noisy data that
  is very unlike the data they were trained with.
