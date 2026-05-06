# SubtitleRemover

GPU 加速视频去字幕工具。集成 VSR（Video Super Resolution）模块，在去除字幕的同时恢复画面质量。

## 工作流程

```
输入视频
  → ffmpeg 解码（提取帧 + 音频）
  → 字幕区域检测
      ├── OCR 文字检测（EasyOCR / PaddleOCR）
      └── 固定位置检测（底部黑边 / 指定 ROI）
  → 逐帧去字幕
      ├── OpenCV inpainting（快速）
      └── LaMa / MAT（高质量）
  → VSR 画质增强（恢复 inpainted 区域细节）
      ├── BasicVSR++ / Real-BasicVSR
      └── Real-ESRGAN（可选增强）
  → 音视频合并输出
```

## 目录结构

```
SubtitleRemover/
├── src/
│   ├── detection/        # 字幕区域检测
│   │   ├── __init__.py
│   │   ├── ocr_detector.py      # OCR 文字检测
│   │   └── roi_detector.py      # 固定区域检测
│   ├── inpainting/       # 字幕去除
│   │   ├── __init__.py
│   │   ├── opencv_inpaint.py    # OpenCV 快速去除
│   │   └── deep_inpaint.py      # 深度学习去字幕
│   ├── vsr/              # 视频超分辨率（核心模块）
│   │   ├── __init__.py
│   │   ├── basicvsr.py          # BasicVSR++ 集成
│   │   ├── real_esrgan.py       # Real-ESRGAN 增强
│   │   └── models/              # 预训练模型下载
│   ├── pipeline.py       # 完整处理管线编排
│   ├── config.py         # 全局配置
│   └── api_server.py     # FastAPI 服务
├── tests/
├── requirements.txt
└── README.md
```

## 安装

```bash
# 创建虚拟环境
python3.12 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 下载 VSR 预训练模型（首次运行自动）
python -c "from src.vsr.basicvsr import download_model; download_model()"
```

## VSR 模块

### BasicVSR++

当前推荐的 VSR 模型：
- **模型**: BasicVSR++ (CVPR 2022)
- **输入**: 低分辨率帧序列
- **输出**: 高分辨率增强帧
- **框架**: MMEditing / mmedit
- **显存**: ~4GB（1080p 输入）

```python
from src.vsr.basicvsr import BasicVSRPlusPlus

vsr = BasicVSRPlusPlus(device="cuda")
frames = vsr.enhance(input_frames)
```

### Real-ESRGAN（可选增强）

用于最终输出的画质锐化：
```python
from src.vsr.real_esrgan import RealESRGAN

enhancer = RealESRGAN(device="cuda")
result = enhancer.enhance(frame)
```

## API 端点（规划）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/remove-subtitle` | 上传视频，去字幕 + VSR |
| `GET` | `/status/{task_id}` | 任务状态查询 |
| `GET` | `/download/{task_id}` | 下载结果 |

## 依赖

- Python 3.12+
- NVIDIA GPU（建议 8GB+ 显存）
- CUDA 12.x
- ffmpeg
- OpenCV
- PyTorch (CUDA)

## GPU 需求

| 任务 | 显存占用 | 4070 Ti Super 16GB |
|------|:-------:|:------------------:|
| 去字幕 (inpainting) | ~2GB | ✅ |
| VSR (BasicVSR++) | ~4GB | ✅ |
| 人声分离 (同期执行) | ~4GB | ✅ 共用 16GB 够 |
