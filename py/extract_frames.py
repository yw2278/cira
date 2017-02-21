#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

if __name__ == "__main__":

    # -- set the output path and base filename
    obase = os.path.join("..", "output", "IRTL_frames", 
                         "IRTL_frame_{0:04}.npy")

    # -- open the video
    fname = os.path.join(os.path.expanduser("~"), "data", "ir", "IRTL.mp4")
    cap   = cv2.VideoCapture(fname)

    # -- read frames
    print("reading frames from {0}...".format(fname))
    nfr = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    frs = np.array([cap.read()[1][..., ::-1] for i in range(nfr)])

    # -- close the video
    cap.release()

    # -- write frames to file
    print("writing frames to files...")
    for ii in range(nfr):
        np.save(obase.format(ii), frs[ii])
