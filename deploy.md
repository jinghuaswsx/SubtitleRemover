# SubtitleRemover 部署文档

最后核实：2026-05-06

## 项目信息

- **GitHub**: https://github.com/jinghuaswsx/SubtitleRemover
- **生产状态**: `0.2.0-phaseC`，已由 `subtitle-remover.service` 上线
- **生产端口**: `84`（systemd 设置 `SR_PORT=84`；代码默认值仍为 `8082`）
- **生产路径**: `/home/cjh/code/SubtitleRemover`
- **生产 venv**: `/home/cjh/code/SubtitleRemover/venv`
- **支持分辨率**: 1080p (1920×1080) / 2K (2560×1440)
- **技术栈**: FastAPI + OpenCV + PyTorch CUDA + EasyOCR / PaddleOCR + LaMa + Real-ESRGAN / BasicSR

> `/home/cjh/code/SubtitleRemover` 是生产工作目录，不要作为临时开发副本使用。
> 常规开发应在隔离 worktree 完成，提交后再由服务器侧拉取。

## 架构概览

```
输入视频
  -> ffmpeg 解码
  -> 字幕区域检测（ROI / OCR）
  -> 去字幕（OpenCV / LaMa）
  -> 可选画质增强（Real-ESRGAN ROI / BasicVSR++ 全帧）
  -> ffmpeg 合并音视频
  -> 输出 mp4
```

## 当前模块状态

| 模块 | 状态 | 文件 | 说明 |
|---|:---:|---|---|
| 配置管理 | 已实现 | `src/config.py` | `SR_*` 环境变量覆盖 |
| 字幕区域检测 | 已实现 | `src/detection/roi_detector.py`, `src/detection/ocr_detector.py` | 固定 ROI / EasyOCR / PaddleOCR |
| 逐帧去字幕 | 已实现 | `src/inpainting/opencv_inpaint.py`, `src/inpainting/deep_inpaint.py` | OpenCV TELEA / LaMa |
| VSR / 增强 | 已实现 | `src/vsr/real_esrgan.py`, `src/vsr/basicvsr.py` | 生产推荐 `real-esrgan` |
| 视频管线 | 已实现 | `src/pipeline.py` | 串行处理，保留音频 |
| FastAPI 服务 | 已实现 | `src/api_server.py` | 异步任务队列、状态轮询、下载 |
| API 文档 | 已实现 | `docs/api.md` | phase C 接口说明 |

`vsr=basicvsr++` 的代码路径已存在，但公开 mmediting checkpoint 与 basicsr SPyNet
通道宽度不兼容；生产请求推荐 `vsr=real-esrgan`。

## 生产 systemd 配置

实际单元 `/etc/systemd/system/subtitle-remover.service`：

```ini
[Unit]
Description=SubtitleRemover API
After=network.target

[Service]
Type=simple
User=cjh
Group=cjh
WorkingDirectory=/home/cjh/code/SubtitleRemover
Environment=SR_PORT=84
Environment=PYTHONUNBUFFERED=1
Environment=SR_GPU_FRACTION=0.9
ExecStart=/home/cjh/code/SubtitleRemover/venv/bin/python -m src.api_server
Restart=on-failure
RestartSec=5
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
```

`SR_PORT=84` 会被 `src/config.py` 读取为 `config.SERVICE_PORT`，并传给
`uvicorn.run(app, host="0.0.0.0", port=config.SERVICE_PORT)`。

## 重启前验证

由于生产 venv 曾被外部 pip 操作变更，重启前必须先验证，不要直接 restart：

```bash
cd /home/cjh/code/SubtitleRemover

sudo -u cjh /home/cjh/code/SubtitleRemover/venv/bin/python \
  -c "from src import api_server; print('import ok')"

curl http://127.0.0.1:84/health
```

当前 venv 的冻结结果已提交为 `requirements.lock`。需要重建或回滚环境时，先对比
`requirements.lock`，再决定是否调整依赖。

## API

详见 [docs/api.md](docs/api.md)。常用端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/health` | 存活探针 |
| `GET` | `/info` | 模式与限制信息 |
| `POST` | `/remove-subtitle` | 上传视频并创建任务 |
| `GET` | `/status/{task_id}` | 查询任务状态 |
| `GET` | `/download/{task_id}` | 下载结果 mp4 |

## 与 AudioSeparator 共存

LocalServer 上 AudioSeparator 也已上线：

| 服务 | 端口 | GPU 配额 | 工作目录 |
|---|---:|---:|---|
| AudioSeparator | `83` | `AS_GPU_FRACTION=0.9` | `/home/cjh/code/AudioSeparator` |
| SubtitleRemover | `84` | `SR_GPU_FRACTION=0.9` | `/home/cjh/code/SubtitleRemover` |

两个服务都使用 `<1024` 端口，systemd unit 保留 `CAP_NET_BIND_SERVICE`。

## 相关文档

- [README.md](README.md) — 项目概览
- [localserver.md](localserver.md) — LocalServer 实际部署信息
- [docs/deploy_guide.md](docs/deploy_guide.md) — 4070 Ti Super 双服务部署指南
- [docs/handoff-2026-05-06-server-state.md](docs/handoff-2026-05-06-server-state.md) — 服务器状态交接
