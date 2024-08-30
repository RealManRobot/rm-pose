from abc import ABC, abstractmethod

import cv2
import numpy as np

from FoundationPose.estimater11 import *
from FoundationPose.datareader import *


class DetectBase(ABC):

    @staticmethod
    @abstractmethod
    def forward_handle_input(color_frame, depth_frame):
        pass

    @staticmethod
    @abstractmethod
    def gen_model(weights):
        pass

    @staticmethod
    @abstractmethod
    def backward_handle_output(output):
        pass


class Detect_foundationpose(DetectBase):

    @staticmethod
    def load_model(mesh_file, intrinsics, predict_ckpt_dir, refine_ckpt_dir):
        """
        xxxx
        :param mesh_file:
        :param intrinsics:
        :return:
        """

        debug = 1
        debug_dir = f'FoundationPose/debug'
        mesh = trimesh.load(mesh_file)
        os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

        # 可视化部分代码
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        # to_origin = mesh.apply_obb()
        # print('---------------------------------------------------------------------------------------', to_origin)

        # to_origin = np.linalg.inv(to_origin)
        bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # bbox = np.array([[-0.024689, -0.046259, -0.096266],
        #                  [0.024689, 0.046259, 0.096266]])

        # 评分标准和后处理代码
        scorer = ScorePredictor(predict_ckpt_dir)
        refiner = PoseRefinePredictor(refine_ckpt_dir)
        glctx = dr.RasterizeCudaContext()
        est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer,
                             refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)

        K = np.eye(3)
        K[0][0] = intrinsics.fx
        K[1][1] = intrinsics.fy
        K[0][2] = intrinsics.ppx
        K[1][2] = intrinsics.ppy
        reader = YcbineoatReader(K, shorter_side=None, zfar=np.inf)

        return est, reader, bbox, debug, to_origin

    @staticmethod
    def pose_est(color_img, depth_img, mask, reader, est, to_origin, bbox, show=False):
        """
        xxxx
        :param color_frame:
        :param depth_frame:
        :param mask:
        :return:
        """

        est_refine_iter = 5
        color = reader.get_color(color_img)
        depth = reader.get_depth(depth_img)
        pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=est_refine_iter)

        if show:
            center_pose = pose @ np.linalg.inv(to_origin)
            draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                is_input_rgb=True)

            mtx = np.array([[606, 0.0, 314],
                            [0.0, 606, 250],
                            [0.0, 0.0, 1.0]])
            dist = np.array([0, 0, 0, 0, 0], dtype=np.double)

            cv2.drawFrameAxes(color, mtx, dist, center_pose[:3, :3], center_pose[:3, 3], 0.07)

            return pose, color, to_origin
        else:
            return pose, color, to_origin

    @staticmethod
    def pose_track(color_img, depth_img, reader, est, to_origin, bbox, show=False):
        """
        xxxx
        :param color_frame:
        :param depth_frame:
        :param mask:
        :return:
        """

        track_refine_iter = 2
        color = reader.get_color(color_img)
        depth = reader.get_depth(depth_img)

        pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=track_refine_iter)

        center_pose = pose @ np.linalg.inv(to_origin)
        # center_pose = pose @ to_origin

        if show:

            print('center_pose', center_pose)
            draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0,
                                is_input_rgb=True)

            return center_pose, vis[..., ::-1]
            # return center_pose, color
        else:
            return center_pose, color
