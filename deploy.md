# SubtitleRemover 部署文档

## 项目信息

- **GitHub**: https://github.com/jinghuaswsx/SubtitleRemover
- **服务端口**: 8082（规划，可通过 `SR_PORT` 更改）
- **支持分辨率**: 1080p (1920×1080) / 2K (2560×1440)
- **技术栈**: FastAPI + OpenCV + PyTorch CUDA + BasicVSR++

> ⚠️ **开发中项目** — 以下部署流程基于架构设计，实际部署需先完成模块开发。

## 架构概览

```
┌──────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐
│ 输入视频 │ → │ ffmpeg解码 │ → │ 字幕检测  │ → │ 去字幕   │ → │ VSR增强  │
│ (MP4/MKV)│   │ 提取帧+音频│   │ OCR/ROI   │   │ Inpaint  │   │ BasicVSR++│
└──────────┘   └───────────┘   └───────────┘   └──────────┘   └──────────┘
                                                    ↓
                                               ┌──────────┐
                                               │ 合并输出 │
                                               │ 视频+音频│
                                               └──────────┘
```

## 环境要求

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.12+ | |
| CUDA | 12.x | |
| NVIDIA 驱动 | >= 525 | |
| 显存 | >= 8GB | 2K + VSR 建议 16GB |
| ffmpeg | 任意 | 视频解码编码 |
| OpenCV | >= 4.8 | 图像处理 + inpainting |

## 模块开发进度

| 模块 | 状态 | 文件 | 说明 |
|------|:----:|------|------|
| 配置管理 | ✅ | `src/config.py` | 全局配置，支持环境变量覆盖 |
| VSR 框架 | ⏳ | `src/vsr/basicvsr.py` | BasicVSR++ 接口已定义，待对接 MMEditing |
| 字幕区域检测 | ❌ | `src/detection/` | 待开发：OCR 检测 / ROI 固定区域 |
| 逐帧去字幕 | ❌ | `src/inpainting/` | 待开发：OpenCV inpainting / LaMa |
| FastAPI 服务 | ❌ | `src/api_server.py` | 待开发：任务队列 + 状态轮询 |
| 测试 | ❌ | `tests/` | 待编写 |

## 快速开发启动

### 1. 克隆项目

```bash
git clone https://github.com/jinghuaswsx/SubtitleRemover.git /opt/SubtitleRemover
cd /opt/SubtitleRemover
```

### 2. 创建虚拟环境

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 安装 VSR 依赖

```bash
# BasicVSR++ 需要 MMEditing
pip install mmedit

# 或 Real-ESRGAN
pip install real-esrgan
```

### 4. 下载预训练模型

```bash
python -c "
from src.vsr.basicvsr import download_model
download_model('basicvsr_plusplus_reds4')
# 下载到 models/basicvsr_plusplus_reds4.pth
"
```

### 5. 配置环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SR_PORT` | `8082` | 服务端口 |
| `SR_GPU_FRACTION` | `0.4` | GPU 显存限制 |
| `SR_DEVICE` | `cuda` | 计算设备 |
| `SR_MAX_WIDTH` | `2560` | 最大输入宽度 |
| `SR_MAX_HEIGHT` | `1440` | 最大输入高度 |
| `SR_DETECTION_MODE` | `auto` | 字幕检测模式 |
| `SR_INPAINT_METHOD` | `opencv` | 去字幕方法 |
| `SR_VSR_MODEL` | `basicvsr++` | VSR 模型 |
| `SR_VSR_UPSCALE` | `2` | 超分倍数 |

## 开发指引

### 字幕检测模块 (`src/detection/`)

```python
# src/detection/ocr_detector.py
# 使用 EasyOCR 检测字幕区域
# 返回字幕边界框 (x, y, w, h)

# src/detection/roi_detector.py
# 固定区域检测：底部 20%、底部黑边等
# 适用于固定字幕位置的视频
```

### 去字幕模块 (`src/inpainting/`)

```python
# src/inpainting/opencv_inpaint.py
# 快速去字幕：cv2.inpaint() 或 cv2.createInpaint()
# 1080p 单帧 ~50ms

# src/inpainting/deep_inpaint.py
# 高质量：LaMa / MAT 深度学习模型
# 2K 单帧 ~500ms（需 tile 分块）
```

### VSR 模块 (`src/vsr/`)

```python
# src/vsr/basicvsr.py
# ✅ 框架已搭好，需对接 MMEditing 推理 API
# 支持全帧（1080p）和 Tile（2K）两种处理模式

# src/vsr/real_esrgan.py
# 可选锐化增强
```

### 视频管线 (`src/pipeline.py`)

完整处理流程编排：

```
1. ffmpeg 提取帧 → 临时目录 frames/
2. ffmpeg 提取音频 → temp_audio.wav
3. 字幕检测 → 获取字幕 ROI
4. 逐帧 inpainting → 去除字幕
5. VSR 增强 → 恢复字幕区域画质
6. ffmpeg 合并视频+音频 → 输出
7. 清理临时文件
```

## 与 AudioSeparator 共存部署

两台服务共用一张 4070 Ti Super 16GB：

```ini
# AudioSeparator: AS_GPU_FRACTION=0.4 → 用 ~4GB
# SubtitleRemover:  SR_GPU_FRACTION=0.4 → 用 ~6GB (含 VSR)
# 余量: ~4GB 缓冲
```

### systemd 双服务编排

`/etc/systemd/system/subtitle-remover.service`:

```ini
[Unit]
Description=Subtitle Remover API
After=network.target audio-separator.service
Requires=audio-separator.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/SubtitleRemover
Environment=SR_PORT=8082
Environment=SR_GPU_FRACTION=0.4
Environment=SR_MAX_WIDTH=2560
Environment=SR_MAX_HEIGHT=1440
ExecStart=/opt/SubtitleRemover/venv/bin/python /opt/SubtitleRemover/src/api_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### 统一反向代理 (Caddy)

```caddyfile
:80 {
    handle_path /separate/* {
        reverse_proxy localhost:80
    }

    handle_path /remove-subtitle/* {
        reverse_proxy localhost:8082
    }

    handle {
        root * /var/www/html
        file_server
    }
}
```

## 性能预估

### 4070 Ti Super 16GB — 单路串行

| 分辨率 | 去字幕 | + VSR | 合计（10分钟视频） |
|--------|:-----:|:-----:|:------------------:|
| 1080p | ~5min | ~8min | ~13min |
| 2K (tile) | ~8min | ~15min | ~23min |

### 显存占用明细

| 阶段 | 1080p | 2K (tile=512) |
|------|:-----:|:-------------:|
| 帧解码缓存 | ~500MB | ~1GB |
| OpenCV inpainting | ~200MB | ~500MB |
| BasicVSR++ 推理 | ~2.5GB | ~4GB |
| 临时帧存储 (批) | ~500MB | ~1GB |
| **合计** | **~3.7GB** | **~6.5GB** |

## 故障排查

### 2K 视频显存不足

```bash
# 降低批大小
export SR_BATCH_SIZE=2

# 启用 tile 处理
export SR_VSR_TILE_SIZE=512
```

### VSR 模型加载失败

```bash
# 检查 mmedit 安装
python -c "import mmedit; print(mmedit.__version__)"

# 手动下载模型
wget -P models/ https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_reds4_20220916-a3d03b54.pth
```

### ffmpeg 问题

```bash
# 确保 ffmpeg 支持需要的编码格式
ffmpeg -encoders | grep h264
ffmpeg -decoders | grep h264
```

---

## 相关文档

- [localserver.md](localserver.md) — 服务器部署信息
- [README.md](README.md) — 项目概述
- [docs/deploy_guide.md](docs/deploy_guide.md) — 4070 Ti Super 整体部署方案
- [src/config.py](src/config.py) — 配置项详情
