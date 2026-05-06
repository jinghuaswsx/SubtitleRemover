# SubtitleRemover

GPU 加速视频去字幕工具。集成 VSR（Video Super Resolution）模块，在去除字幕的同时恢复画面质量。

## 支持分辨率

| 分辨率 | 像素 | 处理方式 | 显存占用 |
|--------|------|---------|:-------:|
| **1080p** | 1920×1080 | 全帧 VSR | ~3GB |
| **2K** | 2560×1440 | Tile 分块 VSR | ~4-5GB |

## 当前状态

- LocalServer 生产服务：`subtitle-remover.service`
- 生产端口：`84`
- 生产路径：`/home/cjh/code/SubtitleRemover`
- 版本：`0.2.0-phaseC`
- API 文档：[docs/api.md](docs/api.md)

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

此外，独立的 VACE 编辑路由（`POST /vace-edit`，Wan2.1-VACE-1.3B）与上述去字幕流水线
**共享同一个进程内 QUEUE**，串行排队执行。规范见
[docs/vace_integration_plan.md](docs/vace_integration_plan.md)。

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

### Real-ESRGAN（生产推荐）

当前生产推荐用 Real-ESRGAN 增强去字幕区域，保留原视频分辨率：

```python
from src.vsr.real_esrgan import RealESRGAN

enhancer = RealESRGAN(model_dir="models", device="cuda")
result = enhancer.enhance_region(frame, region=(0, 860, 1920, 220))
```

### BasicVSR++

BasicVSR++ 代码路径已接入 basicsr 架构：
- **模型**: BasicVSR++ (CVPR 2022)
- **输入**: 低分辨率帧序列
- **输出**: 高分辨率增强帧（通常 x4）
- **框架**: BasicSR
- **显存**: ~4GB（1080p 输入）

```python
from src.vsr.basicvsr import BasicVSRPlusPlus

vsr = BasicVSRPlusPlus(device="cuda", model_dir="models")
frames = vsr.enhance(input_frames)
```

注意：公开 mmediting BasicVSR++ checkpoint 与 basicsr SPyNet 通道宽度不兼容。
生产请求推荐使用 `vsr=real-esrgan`；只有提供 basicsr-native checkpoint 时再启用
`vsr=basicvsr++`。

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 存活探针 |
| `GET` | `/info` | 模式与限制信息 |
| `POST` | `/remove-subtitle` | 上传视频，去字幕 + VSR |
| `POST` | `/vace-edit` | VACE 视频编辑（独立路由，与 `/remove-subtitle` 串行）|
| `GET` | `/status/{task_id}` | 任务状态查询（subtitle / vace 通用）|
| `GET` | `/download/{task_id}` | 下载结果（subtitle / vace 通用）|

任务异步执行，`POST /remove-subtitle` 与 `POST /vace-edit` 返回 `task_id` 后轮询
`/status/{task_id}`。两条路由**共享单一 QUEUE 与 worker**，同一时间只跑一个任务。
完整参数矩阵见 [docs/api.md](docs/api.md)；VACE 接入实施规范见
[docs/vace_integration_plan.md](docs/vace_integration_plan.md)。

## 依赖

- Python 3.12+
- NVIDIA GPU（建议 8GB+ 显存）
- CUDA 12.x
- ffmpeg
- OpenCV
- PyTorch (CUDA)

生产 venv 的当前冻结结果见 `requirements.lock`。该 lock 反映 2026-05-06
服务器环境状态，用于重启前对账和必要时重建环境。

## GPU 需求

| 任务 | 显存占用 | 4070 Ti Super 16GB |
|------|:-------:|:------------------:|
| 1080p 去字幕 + VSR | ~3-4GB | ✅ |
| 2K 去字幕 + VSR（tile）| ~5-6GB | ✅ |
| AudioSeparator 同时运行 | ~4GB | ✅ 共用 16GB 够 |
| 总占用 | ~10GB | 余 ~6GB 缓冲 |
