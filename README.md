# SAR Target Detection Project

## 📋 项目概述

本项目基于YOLOv8-OBB实现SAR（合成孔径雷达）图像的目标检测，支持旋转边界框检测。项目已完成环境配置、数据预处理、训练流程搭建和numpy兼容性修复等核心工作。

### 🎯 检测目标类别
- `ship` - 船舶
- `aircraft` - 飞机  
- `car` - 汽车
- `tank` - 坦克
- `bridge` - 桥梁
- `harbor` - 港口

## 🏗️ 项目结构

```
SAR_Detection/
├── configs/                    # 配置文件
│   └── sar_config.yaml        # SAR检测配置文件
├── notebooks/                  # 数据分析笔记本
│   └── 01_data_exploration.ipynb
├── src/                       # 源代码
│   └── dataset.py            # 数据集处理工具
├── outputs/                   # 训练输出(本地)
├── data/                     # 数据集(本地，已忽略)
│   ├── train/
│   │   ├── images/          # 训练图像
│   │   └── labelTxt/        # 标注文件(DOTA格式)
│   └── test_A/
│       ├── images/          # 测试图像  
│       └── annfiles/        # 测试标注
├── Yolov8_obb_Prune_Track/   # YOLOv8-OBB训练代码(本地，已忽略)
└── README.md                 # 项目说明文档
```

## 🛠️ 环境配置

### 基础环境
- **Python**: 3.9+
- **CUDA**: 支持CUDA的GPU环境
- **操作系统**: Windows/Linux

### 依赖安装

1. **克隆YOLOv8-OBB项目**（本地配置）
```bash
git clone https://github.com/hukaixuan19970627/yolov8_obb.git Yolov8_obb_Prune_Track
cd Yolov8_obb_Prune_Track
```

2. **安装Python依赖**
```bash
# 创建conda环境
conda create -n sar_det python=3.9
conda activate sar_det

# 安装依赖（已修复numpy兼容性问题）
pip install -r requirements.txt
pip install shapely  # OBB功能必需
```

### 🔧 已修复的关键问题

#### Numpy 2.0兼容性修复
已修复以下文件中的numpy兼容性问题：
- `utils/datasets.py`: 修复`np.int`已弃用问题
- `utils/general.py`: 修复数据类型转换问题
- 解决了混合数据类型（数字+字符串）的numpy数组创建问题

#### OpenCV版本冲突修复
- 修复`requirements.txt`中opencv-python版本固定问题
- 改为兼容版本：`opencv-python>=4.5.4`

## 📊 数据准备

### 数据格式
- **图像格式**: PNG/JPG，640x640推荐
- **标注格式**: DOTA格式旋转边界框
- **标注内容**: `x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty`

### 数据结构要求
```
data/
├── train/
│   ├── images/              # 训练图像 (44,544张)
│   └── labelTxt/           # 标注文件 (45,736个)
└── test_A/
    ├── images/             # 测试图像 (4,860张)  
    └── annfiles/           # 测试标注 (4,860个)
```

### 数据配置
配置文件位置: `configs/sar_config.yaml`
```yaml
path: ./data                 # 数据根目录
train: train/images         # 训练图像路径
val: train/images           # 验证图像路径(当前复用训练集)
nc: 6                       # 类别数量
names: ['ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor']
```

## 🚀 训练流程

### 快速开始
```bash
cd Yolov8_obb_Prune_Track

# 基础训练命令
python train.py \
    --data "../configs/sar_config.yaml" \
    --epochs 100 \
    --batch-size 8 \
    --imgsz 640 \
    --name "sar_detection" \
    --cfg "models/yaml/yolov8n.yaml" \
    --project "../outputs"
```

### 训练参数说明
- `--data`: 数据配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批量大小（根据GPU内存调整）
- `--imgsz`: 输入图像尺寸
- `--name`: 实验名称
- `--cfg`: 模型配置文件
- `--project`: 输出目录

### 已验证的训练配置
✅ **环境验证**: 训练流程已成功启动，无数据加载或模型维度错误  
✅ **数据流验证**: 模型能正确读取SAR数据并开始训练  
✅ **输出验证**: 在`outputs/`目录成功生成权重文件和日志

## 📈 训练监控

### TensorBoard监控
```bash
tensorboard --logdir outputs/sar_detection
```

### 输出文件
训练完成后在`outputs/sar_detection/`目录下生成：
- `weights/best.pt` - 最佳模型权重
- `weights/last.pt` - 最终模型权重  
- `hyp.yaml` - 超参数配置
- `opt.yaml` - 训练选项
- `results.png` - 训练曲线图

## ⚠️ 注意事项

### 数据标签问题
- 部分图像存在负坐标值，训练时会自动跳过
- 建议在正式训练前清理数据标注

### 性能优化建议
- **GPU内存**: 建议8GB+显存，可适当调整batch_size
- **数据预处理**: 可考虑实现train/val数据划分
- **模型选择**: 可尝试yolov8s/m/l获得更好性能

### 兼容性说明
- 代码基于PyTorch 2.8.0+测试
- 已修复Numpy 2.0兼容性问题
- Windows和Linux环境均已验证

## 🔄 后续工作建议

### 短期任务
1. **数据清理**: 修复负坐标标注，提升数据质量
2. **数据划分**: 实现train/val数据集划分
3. **超参数调优**: 针对SAR数据特点调整训练参数
4. **模型评估**: 实现完整的mAP评估流程

### 中期任务  
1. **数据增强**: 针对SAR图像特点设计增强策略
2. **多尺度训练**: 实现多尺度SAR目标检测
3. **模型压缩**: 部署优化和模型剪枝
4. **推理优化**: TensorRT/ONNX转换

### 长期规划
1. **自动标注**: 基于训练好的模型实现半自动标注
2. **在线推理**: 构建实时SAR目标检测服务
3. **模型集成**: 多模型融合提升检测精度

## 📞 技术支持

### 关键文件说明
- `configs/sar_config.yaml` - 核心配置文件，包含所有训练参数
- `Yolov8_obb_Prune_Track/utils/datasets.py` - 已修复numpy兼容性
- `Yolov8_obb_Prune_Track/utils/general.py` - 已修复数据类型问题

### 问题排查
1. **训练中断**: 检查GPU内存和batch_size设置
2. **数据加载失败**: 确认数据路径和标注格式
3. **依赖冲突**: 重新安装requirements.txt和shapely

### 项目状态
🟢 **环境配置**: 完成  
🟢 **数据流验证**: 完成  
🟢 **训练流程**: 验证通过  
🟡 **模型训练**: 待完整训练  
🟡 **性能评估**: 待实施  

---

**最后更新**: 2025年9月22日  
**项目负责人**: [当前负责人]  
**技术栈**: YOLOv8-OBB + PyTorch + CUDA
