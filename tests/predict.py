import copy
import json
import os.path
from types import SimpleNamespace
import cv2
import pyrealsense2 as rs
from FoundationPose.estimater11 import *
from FoundationPose.datareader import *
from FoundationPose.foundationpose_main import Detect_foundationpose


def load_resources(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir):
    est, reader, bbox, debug, to_origin = Detect_foundationpose.load_model(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir)
    return est, reader, bbox, debug, to_origin


def main():
    image_path = "demo_data/test_img"
    color_path = os.path.join(image_path, "rgb.png")
    depth_path = os.path.join(image_path, "depth.png")
    mask_path = os.path.join(image_path, "mask.png")

    mesh_path = "demo_data/haoliyou/mesh/textured_mesh.obj"

    predict_ckpt_dir = "weights/predict_ckpt/predict_ckpt.pth"
    refine_ckpt_dir = "weights/refine_ckpt/refine_ckpt.pth"

    json_file = os.path.join(image_path, "intrinsics.json")
    with open(json_file, 'r+') as fp:
        intrinsics = json.load(fp, object_hook=lambda d: SimpleNamespace(**d))

    color_img = cv2.imread(color_path)
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 调用加载mesh和相关的资源
    est, reader, bbox, debug, to_origin = load_resources(mesh_path, intrinsics, predict_ckpt_dir, refine_ckpt_dir)
    # 根据mesh进行位姿估计
    pose, color, to_origin = Detect_foundationpose.pose_est(color_img, depth_img, mask, reader, est,
                                                            to_origin, bbox,
                                                            show=True)
    # 位姿估计的可视化
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    cv2.imshow("pose", color)
    cv2.waitKey(0)


if __name__ == '__main__':

    main()
