
from __future__ import annotations
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import json

import pycocotools.coco as coco
from matplotlib.patches import Polygon
from customVisualizer import Visualizer, GenericMask
import os
from matplotlib.path import Path

img_folder = "/home/khiemphi"
file_open = open("via_export_json.json")
json_dict = json.load(file_open) 


for key in json_dict.keys(): 
    anno = json_dict[key]


    orig_file_name = anno["filename"]
    mask_file_name =  "segm_mask_" + orig_file_name  
    vis_file_name = "vis_" +   orig_file_name 

    all_bbox = []
    all_classes = []
    all_masks = []
    all_classes = []

    all_segm_x = []
    all_segm_y = []


    if len(anno["regions"]) > 0:
        for item in anno["regions"]:
            segm_x = item["shape_attributes"]["all_points_x"]
            segm_y = item["shape_attributes"]["all_points_y"]
            class_name = item["region_attributes"]["Object Class"]
            
            poly = [(x+0.5, y+0.5) for x, y in zip(segm_x, segm_y)]
            poly = [p for x in poly for p in x]            
            bbox =  [np.min(segm_x), np.min(segm_y), np.max(segm_x), np.max(segm_y)]
            
            all_segm_x.append(segm_x)
            all_segm_y.append(segm_y)
            
            all_masks.append(poly)
            all_bbox.append(bbox)
            all_classes.append(class_name)
            
            
            
            # mask = polygons_to_bitmask([anno["segmentation"]], height=720, width=1280) # binary mask generated
            # cv2.imwrite(mask_file_name, mask*255)

        # Now visualize overlaid-mask
        img_rgb = cv2.imread(os.path.join(img_folder, orig_file_name))
        img_rgb = img_rgb[:, :, ::-1]
        visualizer = Visualizer(img_rgb, 1)

        # Initialize visualization dictionary        
        vis_dict = {"pred_boxes": all_bbox, "pred_classes": all_classes, "pred_masks": all_masks}
        overlay_vis, binary_mask_obj = visualizer.draw_instance_predictions(vis_dict, score_show=False) 
        binary_mask = [ x.mask*255 for x in binary_mask_obj]
        
        for x, y, name in  zip(all_segm_x, all_segm_y, all_classes):
            bin_mask_name = "bin_mask_" + name + "_" + orig_file_name 
            
           
            #cv2.imwrite(bin_mask_name, mask)
            ny, nx = img_rgb.shape[0:2]
            poly_verts = [ (idx_y, idx_x) for idx_y, idx_x in zip(x, y)]
            x, y = np.meshgrid(np.arange(nx), np.arange(ny))
            x, y = x.flatten(), y.flatten()

            points = np.vstack((x,y)).T
            path = Path(poly_verts)
            mask = path.contains_points(points)
            mask = mask.reshape((ny,nx))
            mask = mask.astype(np.uint8) *255
            cv2.imwrite(bin_mask_name, mask)
          


        overlay_vis = overlay_vis.get_image()[:, :, ::-1]
        cv2.imwrite(vis_file_name, overlay_vis)

