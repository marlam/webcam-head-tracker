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

#include <cstdio>
#include <cmath>

int main(void)
{
    WebcamHeadTracker tracker(WebcamHeadTracker::Debug_Timing | WebcamHeadTracker::Debug_Window);
    if (!tracker.initWebcam()) {
        fprintf(stderr, "No usable webcam found\n");
        return 1;
    }
    if (!tracker.initPoseEstimator()) {
        fprintf(stderr, "Cannot initialize pose esimator:\n"
                "haarcascade_frontalface_alt.xml and shape_predictor_68_face_landmarks.dat\n"
                "are not where they were when libwebcamheadtracker was built\n");
        return 1;
    }
    float lastPos[3] = { 0.0f, 0.0f, -1.0f };
    while (tracker.isReady()) {
        tracker.getNewFrame();
        bool gotPose = tracker.computeHeadPose();
        if (gotPose) {
            float pos[3];
            tracker.getHeadPosition(pos);
            pos[0] *= 1000.0f;
            pos[1] *= 1000.0f;
            pos[2] *= 1000.0f;
            fprintf(stderr, "position in mm: %+6.1f %+6.1f %+6.1f\n", pos[0], pos[1], pos[2]);
            if (lastPos[2] >= 0.0f) {
                float diff[3] = { pos[0] - lastPos[0], pos[1] - lastPos[1], pos[2] - lastPos[2] };
                float dist = std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);
                fprintf(stderr, "distance to last position in mm: %+6.1f\n", dist);
            }
            float quaternion[4];
            tracker.getHeadOrientation(quaternion);
            float halfAngle = std::acos(quaternion[3]);
            float axis[3];
            axis[0] = quaternion[0] / std::sin(halfAngle);
            axis[1] = quaternion[1] / std::sin(halfAngle);
            axis[2] = quaternion[2] / std::sin(halfAngle);
            fprintf(stderr, "orientation: rotated %+4.1f degrees around axis (%+4.2f %+4.2f %+4.2f)\n",
                    2.0f * halfAngle / (float)M_PI * 180.0f, axis[0], axis[1], axis[2]);
            lastPos[0] = pos[0];
            lastPos[1] = pos[1];
            lastPos[2] = pos[2];
        }
    }
    return 0;
}
