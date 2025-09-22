import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import math


class SARDataset(Dataset):
    """SAR目标检测数据集类"""
    
    def __init__(self, images_dir, annotations_dir, transform=None, image_size=(512, 512)):
        """
        初始化数据集
        
        Args:
            images_dir (str): 图片文件夹路径
            annotations_dir (str): 标注文件夹路径
            transform: 图像变换（可选）
            image_size (tuple): 目标图片尺寸 (width, height)，默认(512, 512)
        """
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.transform = transform
        self.image_size = image_size
        
        # 建立图片和标注文件的对应列表
        self.image_paths = []
        self.annotation_paths = []
        
        # 遍历图片文件夹，找到对应的标注文件
        for img_file in self.images_dir.glob("*.png"):
            img_name = img_file.stem  # 获取不带扩展名的文件名
            ann_file = self.annotations_dir / f"{img_name}.txt"
            
            if ann_file.exists():
                self.image_paths.append(img_file)
                self.annotation_paths.append(ann_file)
        
        print(f"数据集初始化完成，共找到 {len(self.image_paths)} 个有效样本")
        
        # 定义类别映射
        self.class_to_id = {
            'ship': 0,
            'bridge': 1,
            'aircraft': 2,
            'tank': 3,
            'vehicle': 4,
            'harbor': 5,
            'building': 6
        }
        
        # 反向映射
        self.id_to_class = {v: k for k, v in self.class_to_id.items()}
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            tuple: (image_tensor, targets)
        """
        # 获取图片和标注文件路径
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]
        
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        # 转换为RGB格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 记录原始图片尺寸
        original_height, original_width = image.shape[:2]
        
        # 将图片resize到统一尺寸
        target_width, target_height = self.image_size
        image = cv2.resize(image, (target_width, target_height))
        
        # 读取并解析标注文件，传入缩放比例
        scale_x = target_width / original_width
        scale_y = target_height / original_height
        targets = self._parse_annotations(annotation_path, (target_height, target_width), idx, scale_x, scale_y)
        
        # 转换图片为tensor
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换：归一化到[0,1]并转换为tensor
            image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        return image, targets
    
    def _parse_annotations(self, annotation_path, image_shape, idx, scale_x=1.0, scale_y=1.0):
        """
        解析标注文件
        
        Args:
            annotation_path (Path): 标注文件路径
            image_shape (tuple): 图片尺寸 (height, width)
            idx (int): 样本索引
            scale_x (float): x方向缩放比例
            scale_y (float): y方向缩放比例
            
        Returns:
            dict: 包含所有目标信息的字典
        """
        height, width = image_shape
        
        boxes = []  # 旋转框参数 [cx, cy, w, h, angle]
        labels = []  # 类别标签
        
        with open(annotation_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 9:  # 至少需要8个坐标 + 1个类别
                continue
            
            # 提取8个坐标点
            try:
                coords = [float(x) for x in parts[:8]]
                class_name = parts[8]
                
                # 应用缩放比例调整坐标
                scaled_coords = []
                for i in range(0, 8, 2):
                    scaled_x = coords[i] * scale_x
                    scaled_y = coords[i+1] * scale_y
                    scaled_coords.extend([scaled_x, scaled_y])
                
                # 将8个坐标转换为4个点
                points = [(scaled_coords[i], scaled_coords[i+1]) for i in range(0, 8, 2)]
                
                # 转换为旋转框表示
                rotated_box = self._convert_points_to_rotated_box(points)
                
                # 归一化坐标（相对于图片尺寸）
                normalized_box = [
                    rotated_box[0] / width,   # cx
                    rotated_box[1] / height,  # cy
                    rotated_box[2] / width,   # w
                    rotated_box[3] / height,  # h
                    rotated_box[4]            # angle (不需要归一化)
                ]
                
                boxes.append(normalized_box)
                
                # 映射类别名称到ID
                if class_name in self.class_to_id:
                    labels.append(self.class_to_id[class_name])
                else:
                    # 未知类别，可以添加到映射中或者跳过
                    print(f"警告: 未知类别 '{class_name}'，将映射为0")
                    labels.append(0)
                    
            except (ValueError, IndexError) as e:
                print(f"解析标注行出错: {line}, 错误: {e}")
                continue
        
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.long),
            'image_id': torch.tensor([idx], dtype=torch.long)
        }
    
    def _convert_points_to_rotated_box(self, points):
        """
        将4个角点坐标转换为旋转框表示 (cx, cy, width, height, angle)
        
        Args:
            points (list): 4个点的坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            
        Returns:
            list: [cx, cy, width, height, angle] 其中angle以弧度为单位
        """
        # 将点转换为numpy数组
        pts = np.array(points, dtype=np.float32)
        
        # 使用OpenCV的minAreaRect函数获取最小外接矩形
        # 这个函数会返回旋转框的标准表示
        rect = cv2.minAreaRect(pts)
        
        # rect格式: ((cx, cy), (width, height), angle)
        (cx, cy), (width, height), angle = rect
        
        # OpenCV的angle范围是[-90, 0]，我们转换为弧度
        # 注意：OpenCV的角度定义可能需要根据具体数据调整
        angle_rad = math.radians(angle)
        
        return [cx, cy, width, height, angle_rad]
    
    def get_class_name(self, class_id):
        """根据类别ID获取类别名称"""
        return self.id_to_class.get(class_id, 'unknown')
    
    def get_class_id(self, class_name):
        """根据类别名称获取类别ID"""
        return self.class_to_id.get(class_name, 0)


def collate_fn(batch):
    """
    DataLoader的collate函数，用于处理不同大小的标注
    
    Args:
        batch: 一个批次的数据
        
    Returns:
        tuple: (images, targets)
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 堆叠图片张量
    images = torch.stack(images, dim=0)
    
    return images, targets


if __name__ == "__main__":
    # 测试代码
    dataset = SARDataset(
        images_dir="../data/train/images",
        annotations_dir="../data/train/annfiles"
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"类别映射: {dataset.class_to_id}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        image, targets = dataset[0]
        print(f"图片形状: {image.shape}")
        print(f"目标框数量: {len(targets['boxes'])}")
        print(f"目标类别: {targets['labels']}")
