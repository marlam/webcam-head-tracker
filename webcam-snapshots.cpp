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

/* The purpose of this tool is to take snapshots from your webcam for calibration.
 *
 * Do the following:
 * - Print the OpenCV calibration chessboard
 *   http://docs.opencv.org/2.4/_downloads/pattern.png
 * - Start this tool
 * - Hold the chessboard into the view, make a snapshot by pressing space bar
 * - Make at least 10 such snapshots with the chessboard pattern at different
 *   positions and angles (but still facing the camera approximately frontal)
 * - Run the OpenCV python sample calibrate.py:
 *   ./calibrate.py 'webcam-*.ppm'
 * - Check the output images for correctness
 * - Put the reported camera intrinsics into environment variables, preferably
 *   in .profile or .bashrc:
 *   export WEBCAM_INTRINSIC_PARAMETERS="636.68,637.73,323.22,231.55"
 *   export WEBCAM_DISTORTION_COEFFICIENTS="-0.0027493,0.11596,-0.0000753,0.0054066,-0.691798"
 * - The webcam head tracker will use these environment variables instead of
 *   guessing the values.
 */

#include <cstdio>

#include <opencv2/highgui/highgui.hpp>

int main(void)
{
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        fprintf(stderr, "cannot find usable camera\n");
        return 1;
    }

    cv::Mat frame;
    char filename[] = "webcam-0000.ppm";
    int i = 0;
    for (;;) {
        capture >> frame;
        cv::imshow("webcam", frame);
        int key = cv::waitKey(10);
        if (key == 27 || key == 'q') {
            break;
        } else if (key == ' ') {
            snprintf(filename , sizeof(filename), "webcam-%04d.ppm", i);
            fprintf(stderr, "saving %s\n", filename);
            cv::imwrite(filename, frame);
            i++;
            if (i == 9999)
                break;
        }
    }

    return 0;
}
