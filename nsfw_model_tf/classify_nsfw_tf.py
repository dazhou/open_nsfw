#!/usr/bin/env python
"""
Copyright 2016 Yahoo Inc.
Licensed under the terms of the 2 clause BSD license. 
Please see LICENSE file in the project root for terms.
"""

import numpy as np
import os
import sys
import argparse
import glob
import time
from PIL import Image
from StringIO import StringIO

import skimage.io
import skimage.transform
from caffe2.python import core, workspace
from matplotlib import pyplot


def crop_center(img,cropx,cropy):
    y,x,c = img.shape
    startx = max((x-cropx)/2,0)
    starty = max((y-cropy)/2,0)    
    return img[starty:starty+cropy,startx:startx+cropx]

def rescale(img, input_height, input_width):
    print("Original image shape:" + str(img.shape) + " and remember it should be in H, W, C!")
    print("Model's input shape is %dx%d") % (input_height, input_width)
    aspect = img.shape[1]/float(img.shape[0])
    print("Orginal aspect ratio: " + str(aspect))
    
    """
    if(aspect>1):
        # landscape orientation - wide image
        res = int(aspect * input_height)
        imgScaled = skimage.transform.resize(img, (input_width, res))
    if(aspect<1):
        # portrait orientation - tall image
        res = int(input_width/aspect)
        imgScaled = skimage.transform.resize(img, (res, input_height))
    if(aspect == 1):
        imgScaled = skimage.transform.resize(img, (input_width, input_height))
    
    """
    imgScaled = skimage.transform.resize(img, (input_width, input_height))

    pyplot.figure()
    pyplot.imshow(imgScaled)
    pyplot.axis('on')
    pyplot.title('Rescaled image')
    print("New image shape:" + str(imgScaled.shape) + " in HWC")
    return imgScaled
print "Functions set."


def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input file.
    parser.add_argument(
        "input_file",
        help="Path to the input image file"
    )


    args = parser.parse_args()

    with open("init_net.pb") as f:
        init_net = f.read()
    with open("predict_net.pb") as f:
    	predict_net = f.read()

    nsfw_net = workspace.Predictor(init_net, predict_net)

    img = skimage.img_as_float(skimage.io.imread(args.input_file)).astype(np.float32)
    img = rescale(img, 256, 256)
    img = crop_center(img, 224, 224)

    img = img.swapaxes(1, 2).swapaxes(0, 1)
    
    #switch to BGR
    img = img[(2, 1, 0), :, :]

    mean = np.empty([3,224,224])

    mean[0] = 104
    mean[1] = 117
    mean[2] = 123
 
    img = img*255 - mean

    img = img[np.newaxis, :, :, :].astype(np.float32)

    #img.shape = (1,) + img.shape

    print "NCHW: ", img.shape


    # Classify.
    outputs = nsfw_net.run({'data':img})
    scores = outputs[0][0].astype(float)

    # Scores is the array containing SFW / NSFW image probabilities
    # scores[1] indicates the NSFW probability
    print "NSFW score:  " , scores[1]



if __name__ == '__main__':
    main(sys.argv)
