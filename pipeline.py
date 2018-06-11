#!/usr/bin/python

import sys, traceback
import cv2
import os
import re
import numpy as np
import argparse
import string
from plantcv import plantcv as pcv

### Parse command-line arguments
def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-n", "--names", help="path to txt file with names of genotypes to split images into", required =False)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", action=None)
    args = parser.parse_args()
    return args



### Main pipeline
def main():
    # Get options
    args = options()

    # Read image
    img, path, filename = pcv.readimage(args.image)

    debug=args.debug 

    # Pipeline step
    device = 0


    # Step 1: Check if this is a night image, for some of these datasets images were captured
    # at night, even if nothing is visible. To make sure that images are not taken at
    # night we check that the image isn't mostly dark (0=black, 255=white).
    # if it is a night image it throws a fatal error and stops the pipeline.

    if np.average(img) < 50:
        pcv.fatal_error("Night Image")
    else:
        pass


    # Step 2: Normalize the white color so you can later
    # compare color between images.
    # Inputs:
    # device = device number. Used to count steps in the workflow
    # img = image object, RGB colorspace
    # debug = None, print, or plot. Print = save to file, Plot = print to screen.
    # roi = region for white reference, if none uses the whole image,
    # otherwise (x position, y position, box width, box height)

    #white balance image based on white toughspot
    device,img1=pcv.white_balance(device,img,debug,roi=(56,42,10,10))
    # img1 = img


    # Step 3: Rotate the image 

    # device, rotate_img = pcv.rotate(img1, -1, device, debug)


    #Step 4: Shift image. This step is important for clustering later on.
    # For this image it also allows you to push the green raspberry pi camera
    # out of the image. This step might not be necessary for all images.
    # The resulting image is the same size as the original.
    # Input:
    # img = image object
    # device = device number. Used to count steps in the workflow
    # number = integer, number of pixels to move image
    # side = direction to move from "top", "bottom", "right","left"
    # debug = None, print, or plot. Print = save to file, Plot = print to screen.

    # device, shift1 = pcv.shift_img(img1, device, 300, 'top', debug)
    # img1 = shift1


    # STEP 5: Convert image from RGB colorspace to LAB colorspace
    # Keep only the green-magenta channel (grayscale)
    # Inputs:
    #    img     = image object, RGB colorspace
    #    channel = color subchannel (l = lightness, a = green-magenta , b = blue-yellow)
    #    device  = device number. Used to count steps in the workflow
    #    debug   = None, print, or plot. Print = save to file, Plot = print to screen.
    device, a = pcv.rgb2gray_lab(img1, 'a', device, debug)


    # STEP 6: Set a binary threshold on the Saturation channel image
    # Inputs:
    #    img         = img object, grayscale
    #    threshold   = threshold value (0-255)
    #    maxValue    = value to apply above threshold (usually 255 = white)
    #    object_type = light or dark
    #                  - If object is light then standard thresholding is done
    #                  - If object is dark then inverse thresholding is done
    #    device      = device number. Used to count steps in the pipeline
    #    debug       = None, print, or plot. Print = save to file, Plot = print to screen.
    device, img_binary = pcv.binary_threshold(a, 120, 255, 'dark', device, debug)
    #                                            ^
    #                                            |
    #                                           adjust this value


    # STEP 7: Fill in small objects (speckles)
    # Inputs:
    #    img    = image object, grayscale. img will be returned after filling
    #    mask   = image object, grayscale. This image will be used to identify contours
    #    size   = minimum object area size in pixels (integer)
    #    device = device number. Used to count steps in the pipeline
    #    debug  = None, print, or plot. Print = save to file, Plot = print to screen.
    mask = np.copy(img_binary)
    device, fill_image = pcv.fill(img_binary, mask, 100, device, debug)
    #                                               ^
    #                                               |
    #                                               adjust this value


    # STEP 8: Dilate so that you don't lose leaves (just in case)
    # Inputs:
    #    img     = input image
    #    kernel  = integer
    #    i       = interations, i.e. number of consecutive filtering passes
    #    device  = device number. Used to count steps in the pipeline
    #    debug   = None, print, or plot. Print = save to file, Plot = print to screen.

    device, dilated = pcv.dilate(fill_image, 1, 1, device, debug)


    # STEP 9: Find objects (contours: black-white boundaries)
    # Inputs:
    #    img       = image that the objects will be overlayed
    #    mask      = what is used for object detection
    #    device    = device number.  Used to count steps in the pipeline
    #    debug     = None, print, or plot. Print = save to file, Plot = print to screen.
    device, id_objects, obj_hierarchy = pcv.find_objects(img1, dilated, device, debug)



    # STEP 10: Define region of interest (ROI)
    # Inputs:
    #    img       = img to overlay roi
    #    roi       = default (None) or user input ROI image, object area should be white and background should be black,
    #                has not been optimized for more than one ROI
    #    roi_input = type of file roi_base is, either 'binary', 'rgb', or 'default' (no ROI inputted)
    #    shape     = desired shape of final roi, either 'rectangle' or 'circle', if  user inputs rectangular roi but chooses
    #                'circle' for shape then a circle is fitted around rectangular roi (and vice versa)
    #    device    = device number.  Used to count steps in the pipeline
    #    debug     = None, print, or plot. Print = save to file, Plot = print to screen.
    #    adjust    = either 'True' or 'False', if 'True' allows user to adjust ROI
    #    x_adj     = adjust center along x axis
    #    y_adj     = adjust center along y axis
    #    w_adj     = adjust width
    #    h_adj     = adjust height
    roi_contour, roi_hierarchy = pcv.roi.rectangle(x=0, y=84, h=204, w=399, img=img1)
    # device, roi, roi_hierarchy = pcv.define_roi(img1, 'rectangle', device, None, 'default', debug, False, 
    #                                             0, 0, 0, 0)
    #                                            ^                ^
    #                                            |________________|
    #                                            adjust these four values


    # STEP 11: Keep objects that overlap with the ROI
    # Inputs:
    #    img            = img to display kept objects
    #    roi_type       = 'cutto' or 'partial' (for partially inside)
    #    roi_contour    = contour of roi, output from "View and Ajust ROI" function
    #    roi_hierarchy  = contour of roi, output from "View and Ajust ROI" function
    #    object_contour = contours of objects, output from "Identifying Objects" fuction
    #    obj_hierarchy  = hierarchy of objects, output from "Identifying Objects" fuction
    #    device         = device number.  Used to count steps in the pipeline
    #    debug          = None, print, or plot. Print = save to file, Plot = print to screen.
    device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img1, 'partial', roi_contour, roi_hierarchy,
                                                                            id_objects, obj_hierarchy, device,
                                                                            debug)


    #Step 12: This function take a image with multiple contours and 
    # clusters them based on user input of rows and columns

    #Inputs:
    #    img - An RGB image array
    #    roi_objects - object contours in an image that are needed to be clustered.
    #    nrow - number of rows to cluster (this should be the approximate  number of desired rows in the entire image (even if there isn't a literal row of plants)
    #    ncol - number of columns to cluster (this should be the approximate number of desired columns in the entire image (even if there isn't a literal row of plants)
    #    file -  output of filename from read_image function
    #    filenames - input txt file with list of filenames in order from top to bottom left to right
    #    debug - print debugging images

    device, clusters_i, contours = pcv.cluster_contours(device, img1, roi_objects, 4, 6, debug)


    #Step 13:This function takes clustered contours and splits them into multiple images, 
    #also does a check to make sure that the number of inputted filenames matches the number
    #of clustered contours. If no filenames are given then the objects are just numbered

    #Inputs:
    #    img - ideally a masked RGB image.
    #    grouped_contour_indexes - output of cluster_contours, indexes of clusters of contours
    #    contours - contours to cluster, output of cluster_contours
    #    filenames - input txt file with list of filenames in order from top to bottom left to right (likely list of genotypes)
    #    debug - print debugging images

    out = args.outdir
    names = args.names
    device, output_path = pcv.cluster_contour_splitimg(device, img1, clusters_i, contours, out, file=filename, filenames=names, debug=debug)


main()
