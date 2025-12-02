"""
RGBD融合模块 - 将点云和图像融合生成RGBD数据

该模块提供了可扩展的融合算法框架，支持：
1. 基础的点云到深度图投影算法
2. 外部算法接口（可插拔）
3. 时间同步和坐标变换
4. 多种融合策略
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2

from rosclient import RosClient
from rosclient.processors.image_processor import ImageProcessor
from rosclient.processors.pointcloud_processor import PointCloudProcessor


class FusionStrategy(Enum):
    """融合策略枚举"""
    PROJECTION = "projection"  # 点云投影到图像平面
    INTERPOLATION = "interpolation"  # 深度插值
    BILATERAL = "bilateral"  # 双边滤波融合
    MULTI_SCALE = "multi_scale"  # 多尺度融合


@dataclass
class CameraIntrinsics:
    """相机内参"""
    fx: float  # 焦距x
    fy: float  # 焦距y
    cx: float  # 主点x
    cy: float  # 主点y
    width: int  # 图像宽度
    height: int  # 图像高度
    
    def to_matrix(self) -> np.ndarray:
        """转换为相机内参矩阵"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    @classmethod
    def from_matrix(cls, K: np.ndarray, width: int, height: int) -> CameraIntrinsics:
        """从矩阵创建"""
        return cls(
            fx=K[0, 0],
            fy=K[1, 1],
            cx=K[0, 2],
            cy=K[1, 2],
            width=width,
            height=height
        )


@dataclass
class ExtrinsicParams:
    """相机外参（点云到相机的变换）"""
    rotation: np.ndarray  # 3x3旋转矩阵
    translation: np.ndarray  # 3x1平移向量
    
    def to_matrix(self) -> np.ndarray:
        """转换为4x4齐次变换矩阵"""
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation.flatten()
        return T
    
    @classmethod
    def identity(cls) -> ExtrinsicParams:
        """创建单位变换"""
        return cls(
            rotation=np.eye(3, dtype=np.float32),
            translation=np.zeros((3, 1), dtype=np.float32)
        )


@dataclass
class RGBDData:
    """RGBD数据容器"""
    rgb_image: np.ndarray  # RGB图像 (H, W, 3)
    depth_map: np.ndarray  # 深度图 (H, W)
    timestamp: float  # 时间戳
    point_cloud: Optional[np.ndarray] = None  # 原始点云 (N, 3)
    confidence_map: Optional[np.ndarray] = None  # 置信度图 (H, W)
    
    def to_colored_pointcloud(self, camera_intrinsics: CameraIntrinsics) -> np.ndarray:
        """
        将RGBD转换为彩色点云 (N, 6) [x, y, z, r, g, b]
        
        Args:
            camera_intrinsics: 相机内参
            
        Returns:
            彩色点云数组 (N, 6)
        """
        h, w = self.depth_map.shape
        fx = camera_intrinsics.fx
        fy = camera_intrinsics.fy
        cx = camera_intrinsics.cx
        cy = camera_intrinsics.cy
        
        # 创建像素坐标网格
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        u = u.flatten()
        v = v.flatten()
        d = self.depth_map.flatten()
        
        # 过滤无效深度
        valid = d > 0
        u, v, d = u[valid], v[valid], d[valid]
        
        if len(u) == 0:
            return np.empty((0, 6), dtype=np.float32)
        
        # 反投影到3D
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        
        # 获取颜色
        rgb = self.rgb_image[v, u]
        
        # 组合
        points = np.column_stack([x, y, z, rgb])
        return points


class FusionAlgorithm(ABC):
    """融合算法基类 - 可扩展接口"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.log = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fuse(
        self,
        rgb_image: np.ndarray,
        point_cloud: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        extrinsics: ExtrinsicParams,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        融合RGB图像和点云生成深度图
        
        Args:
            rgb_image: RGB图像 (H, W, 3)
            point_cloud: 点云 (N, 3)
            camera_intrinsics: 相机内参
            extrinsics: 外参（点云到相机的变换）
            **kwargs: 额外参数
            
        Returns:
            Tuple[深度图, 置信度图]
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回算法名称"""
        pass


class ProjectionFusionAlgorithm(FusionAlgorithm):
    """基础投影融合算法 - 将点云投影到图像平面"""
    
    def __init__(
        self,
        depth_range: Tuple[float, float] = (0.1, 100.0),
        interpolation_method: str = "nearest",
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.depth_range = depth_range
        self.interpolation_method = interpolation_method
    
    def get_name(self) -> str:
        return "ProjectionFusion"
    
    def fuse(
        self,
        rgb_image: np.ndarray,
        point_cloud: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        extrinsics: ExtrinsicParams,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        使用投影方法融合点云和图像
        
        算法步骤：
        1. 将点云从点云坐标系变换到相机坐标系
        2. 使用相机内参投影到图像平面
        3. 生成深度图和置信度图
        """
        h, w = rgb_image.shape[:2]
        depth_map = np.zeros((h, w), dtype=np.float32)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        if point_cloud is None or len(point_cloud) == 0:
            self.log.warning("Empty point cloud")
            return depth_map, confidence_map
        
        # 1. 变换点云到相机坐标系
        R = extrinsics.rotation
        t = extrinsics.translation.reshape(3, 1)
        
        # 点云转置为 (3, N)
        pc_camera = (R @ point_cloud.T) + t
        pc_camera = pc_camera.T  # 转回 (N, 3)
        
        # 过滤在相机前方的点
        z = pc_camera[:, 2]
        valid = (z > self.depth_range[0]) & (z < self.depth_range[1])
        pc_camera = pc_camera[valid]
        
        if len(pc_camera) == 0:
            self.log.warning("No valid points after filtering")
            return depth_map, confidence_map
        
        # 2. 投影到图像平面
        K = camera_intrinsics.to_matrix()
        fx, fy = camera_intrinsics.fx, camera_intrinsics.fy
        cx, cy = camera_intrinsics.cx, camera_intrinsics.cy
        
        # 投影
        x = pc_camera[:, 0] / pc_camera[:, 2]
        y = pc_camera[:, 1] / pc_camera[:, 2]
        u = (fx * x + cx).astype(np.int32)
        v = (fy * y + cy).astype(np.int32)
        z = pc_camera[:, 2]
        
        # 过滤在图像范围内的点
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[valid], v[valid], z[valid]
        
        if len(u) == 0:
            self.log.warning("No points projected into image")
            return depth_map, confidence_map
        
        # 3. 生成深度图（使用最近邻或最小深度）
        if self.interpolation_method == "nearest":
            # 直接赋值（如果有多个点投影到同一像素，保留最近的）
            for i in range(len(u)):
                if depth_map[v[i], u[i]] == 0 or z[i] < depth_map[v[i], u[i]]:
                    depth_map[v[i], u[i]] = z[i]
                    confidence_map[v[i], u[i]] = 1.0
        elif self.interpolation_method == "min_depth":
            # 使用最小深度
            for i in range(len(u)):
                if depth_map[v[i], u[i]] == 0:
                    depth_map[v[i], u[i]] = z[i]
                    confidence_map[v[i], u[i]] = 1.0
                else:
                    depth_map[v[i], u[i]] = min(depth_map[v[i], u[i]], z[i])
                    confidence_map[v[i], u[i]] += 1.0
        
        # 4. 可选：深度图插值填充空洞
        if kwargs.get("fill_holes", False):
            depth_map = self._fill_holes(depth_map, confidence_map)
        
        return depth_map, confidence_map
    
    def _fill_holes(
        self,
        depth_map: np.ndarray,
        confidence_map: np.ndarray,
        max_hole_size: int = 5
    ) -> np.ndarray:
        """使用形态学操作填充小空洞"""
        filled = depth_map.copy()
        mask = (confidence_map == 0).astype(np.uint8)
        
        # 使用形态学闭运算填充小洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_hole_size, max_hole_size))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 使用inpainting填充
        if np.any(mask_closed):
            filled = cv2.inpaint(
                (depth_map * 1000).astype(np.uint16),
                mask_closed,
                3,
                cv2.INPAINT_TELEA
            ).astype(np.float32) / 1000.0
        
        return filled


class InterpolationFusionAlgorithm(FusionAlgorithm):
    """插值融合算法 - 使用插值方法生成密集深度图"""
    
    def __init__(
        self,
        interpolation_method: str = "linear",
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(logger)
        self.interpolation_method = interpolation_method
    
    def get_name(self) -> str:
        return "InterpolationFusion"
    
    def fuse(
        self,
        rgb_image: np.ndarray,
        point_cloud: np.ndarray,
        camera_intrinsics: CameraIntrinsics,
        extrinsics: ExtrinsicParams,
        **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """使用插值方法生成密集深度图"""
        # 先使用投影算法获得稀疏深度图
        projection = ProjectionFusionAlgorithm(logger=self.log)
        depth_map, confidence_map = projection.fuse(
            rgb_image, point_cloud, camera_intrinsics, extrinsics
        )
        
        # 对稀疏深度图进行插值
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # 找到有效深度点
        valid_mask = confidence_map > 0
        if not np.any(valid_mask):
            return depth_map, confidence_map
        
        valid_u = u[valid_mask]
        valid_v = v[valid_mask]
        valid_depth = depth_map[valid_mask]
        
        # 使用griddata插值
        try:
            from scipy.interpolate import griddata
        except ImportError:
            self.log.error("scipy is required for InterpolationFusionAlgorithm. Install with: pip install scipy")
            return depth_map, confidence_map
        
        points = np.column_stack([valid_u, valid_v])
        grid_points = np.column_stack([u.flatten(), v.flatten()])
        
        interpolated = griddata(
            points,
            valid_depth,
            grid_points,
            method=self.interpolation_method,
            fill_value=0.0
        )
        
        depth_map = interpolated.reshape(h, w)
        confidence_map = (depth_map > 0).astype(np.float32)
        
        return depth_map, confidence_map


class RGBDFusion:
    """RGBD融合主类 - 管理融合流程和算法"""
    
    def __init__(
        self,
        client: RosClient,
        camera_intrinsics: CameraIntrinsics,
        extrinsics: Optional[ExtrinsicParams] = None,
        fusion_algorithm: Optional[FusionAlgorithm] = None,
        time_sync_threshold: float = 0.1,
        logger: Optional[logging.Logger] = None
    ):
        """
        初始化RGBD融合器
        
        Args:
            client: ROS客户端
            camera_intrinsics: 相机内参
            extrinsics: 外参（默认单位变换）
            fusion_algorithm: 融合算法（默认使用投影算法）
            time_sync_threshold: 时间同步阈值（秒）
            logger: 日志器
        """
        self.client = client
        self.camera_intrinsics = camera_intrinsics
        self.extrinsics = extrinsics or ExtrinsicParams.identity()
        self.time_sync_threshold = time_sync_threshold
        self.log = logger or logging.getLogger(self.__class__.__name__)
        
        # 设置融合算法
        self.fusion_algorithm = fusion_algorithm or ProjectionFusionAlgorithm(logger=self.log)
        
        # 数据缓存
        self._image_cache: List[Tuple[np.ndarray, float]] = []
        self._pointcloud_cache: List[Tuple[np.ndarray, float]] = []
        self._max_cache_size = 10
    
    def register_external_algorithm(self, algorithm: FusionAlgorithm) -> None:
        """
        注册外部融合算法
        
        Args:
            algorithm: 实现FusionAlgorithm接口的算法
        """
        if not isinstance(algorithm, FusionAlgorithm):
            raise ValueError("Algorithm must implement FusionAlgorithm interface")
        self.fusion_algorithm = algorithm
        self.log.info(f"Registered external fusion algorithm: {algorithm.get_name()}")
    
    def set_fusion_algorithm(self, algorithm: FusionAlgorithm) -> None:
        """设置融合算法"""
        self.fusion_algorithm = algorithm
        self.log.info(f"Using fusion algorithm: {algorithm.get_name()}")
    
    def _sync_data(
        self,
        image: np.ndarray,
        image_timestamp: float,
        point_cloud: np.ndarray,
        pc_timestamp: float
    ) -> bool:
        """检查数据时间同步"""
        time_diff = abs(image_timestamp - pc_timestamp)
        if time_diff > self.time_sync_threshold:
            self.log.warning(
                f"Time sync issue: image_ts={image_timestamp:.3f}, "
                f"pc_ts={pc_timestamp:.3f}, diff={time_diff:.3f}s"
            )
            return False
        return True
    
    def fuse_latest(self, **fusion_kwargs) -> Optional[RGBDData]:
        """
        融合最新的图像和点云数据
        
        Args:
            **fusion_kwargs: 传递给融合算法的额外参数
            
        Returns:
            RGBD数据或None
        """
        # 获取最新数据
        image_data = self.client.get_latest_image()
        pc_data = self.client.get_latest_point_cloud()
        
        if image_data is None:
            self.log.warning("No image data available")
            return None
        
        if pc_data is None:
            self.log.warning("No point cloud data available")
            return None
        
        rgb_image, image_ts = image_data
        point_cloud, pc_ts = pc_data
        
        # 时间同步检查
        if not self._sync_data(rgb_image, image_ts, point_cloud, pc_ts):
            self.log.warning("Data not synchronized, attempting fusion anyway")
        
        # 确保图像尺寸匹配
        h, w = rgb_image.shape[:2]
        if h != self.camera_intrinsics.height or w != self.camera_intrinsics.width:
            self.log.warning(
                f"Image size mismatch: got ({h}, {w}), "
                f"expected ({self.camera_intrinsics.height}, {self.camera_intrinsics.width})"
            )
            # 更新内参尺寸
            self.camera_intrinsics.height = h
            self.camera_intrinsics.width = w
        
        # 执行融合
        try:
            depth_map, confidence_map = self.fusion_algorithm.fuse(
                rgb_image=rgb_image,
                point_cloud=point_cloud,
                camera_intrinsics=self.camera_intrinsics,
                extrinsics=self.extrinsics,
                **fusion_kwargs
            )
            
            # 创建RGBD数据
            rgbd = RGBDData(
                rgb_image=rgb_image,
                depth_map=depth_map,
                timestamp=(image_ts + pc_ts) / 2.0,
                point_cloud=point_cloud,
                confidence_map=confidence_map
            )
            
            return rgbd
            
        except Exception as e:
            self.log.error(f"Fusion failed: {e}", exc_info=True)
            return None
    
    def fuse_from_data(
        self,
        rgb_image: np.ndarray,
        point_cloud: np.ndarray,
        **fusion_kwargs
    ) -> Optional[RGBDData]:
        """
        从给定的图像和点云数据融合
        
        Args:
            rgb_image: RGB图像
            point_cloud: 点云
            **fusion_kwargs: 传递给融合算法的额外参数
            
        Returns:
            RGBD数据或None
        """
        try:
            depth_map, confidence_map = self.fusion_algorithm.fuse(
                rgb_image=rgb_image,
                point_cloud=point_cloud,
                camera_intrinsics=self.camera_intrinsics,
                extrinsics=self.extrinsics,
                **fusion_kwargs
            )
            
            rgbd = RGBDData(
                rgb_image=rgb_image,
                depth_map=depth_map,
                timestamp=time.time(),
                point_cloud=point_cloud,
                confidence_map=confidence_map
            )
            
            return rgbd
            
        except Exception as e:
            self.log.error(f"Fusion failed: {e}", exc_info=True)
            return None
    
    def visualize_rgbd(self, rgbd: RGBDData, save_path: Optional[str] = None) -> np.ndarray:
        """
        可视化RGBD数据
        
        Args:
            rgbd: RGBD数据
            save_path: 保存路径（可选）
            
        Returns:
            可视化图像
        """
        h, w = rgbd.rgb_image.shape[:2]
        
        # 归一化深度图用于显示
        depth_vis = rgbd.depth_map.copy()
        valid_mask = depth_vis > 0
        if np.any(valid_mask):
            depth_min = np.min(depth_vis[valid_mask])
            depth_max = np.max(depth_vis[valid_mask])
            if depth_max > depth_min:
                depth_vis[valid_mask] = (depth_vis[valid_mask] - depth_min) / (depth_max - depth_min)
        
        depth_vis = (depth_vis * 255).astype(np.uint8)
        depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # 创建组合图像
        vis_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
        vis_image[:h, :w] = rgbd.rgb_image
        vis_image[:h, w:] = depth_colored
        
        # 如果有置信度图，也显示
        if rgbd.confidence_map is not None:
            conf_vis = (rgbd.confidence_map * 255).astype(np.uint8)
            conf_colored = cv2.applyColorMap(conf_vis, cv2.COLORMAP_VIRIDIS)
            vis_image[h:, :w] = conf_colored
        
        # 显示统计信息
        if np.any(valid_mask):
            stats_text = (
                f"Depth: [{depth_min:.2f}, {depth_max:.2f}]m\n"
                f"Valid pixels: {np.sum(valid_mask)}/{h*w}"
            )
            cv2.putText(
                vis_image,
                stats_text,
                (w + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        if save_path:
            cv2.imwrite(save_path, vis_image)
        
        return vis_image


# ==================== 使用示例 ====================

def example_usage():
    """使用示例"""
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("RGBDFusionExample")
    
    # 1. 连接ROS客户端
    connection_url = "ws://192.168.27.152:9090"
    client = RosClient(connection_url)
    client.connect_async()
    
    # 等待连接
    import time
    time.sleep(2)
    
    if not client.is_connected():
        logger.error("Failed to connect to ROS")
        return
    
    # 2. 配置相机参数（需要根据实际相机标定结果设置）
    camera_intrinsics = CameraIntrinsics(
        fx=525.0,  # 焦距x（像素）
        fy=525.0,  # 焦距y（像素）
        cx=320.0,  # 主点x（像素）
        cy=240.0,  # 主点y（像素）
        width=640,
        height=480
    )
    
    # 3. 配置外参（点云到相机的变换，默认单位变换）
    extrinsics = ExtrinsicParams.identity()
    # 如果需要，可以设置实际的变换：
    # extrinsics = ExtrinsicParams(
    #     rotation=np.eye(3),
    #     translation=np.array([[0.0], [0.0], [0.0]])
    # )
    
    # 4. 创建融合器（使用默认投影算法）
    fusion = RGBDFusion(
        client=client,
        camera_intrinsics=camera_intrinsics,
        extrinsics=extrinsics,
        time_sync_threshold=0.1
    )
    
    # 5. 可选：使用外部算法
    # interpolation_algorithm = InterpolationFusionAlgorithm()
    # fusion.set_fusion_algorithm(interpolation_algorithm)
    
    # 6. 执行融合
    logger.info("Starting RGBD fusion...")
    rgbd = fusion.fuse_latest(fill_holes=True)
    
    if rgbd is not None:
        logger.info(f"Fusion successful! Depth map shape: {rgbd.depth_map.shape}")
        
        # 7. 可视化
        vis_image = fusion.visualize_rgbd(rgbd, save_path="rgbd_result.png")
        logger.info("Visualization saved to rgbd_result.png")
        
        # 8. 可以转换为彩色点云
        colored_pc = rgbd.to_colored_pointcloud(camera_intrinsics)
        logger.info(f"Colored point cloud shape: {colored_pc.shape}")
    else:
        logger.error("Fusion failed")
    
    # 9. 清理
    client.terminate()


if __name__ == "__main__":
    example_usage()

