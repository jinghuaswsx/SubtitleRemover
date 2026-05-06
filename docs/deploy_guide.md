# LocalServer 4070 Ti Super 部署指南

最后核实：2026-05-06

## 硬件环境

```
显卡: RTX 4070 Ti Super 16GB
系统: Ubuntu 24.04 LTS
CUDA: 12.x / 13.x wheel 均可能存在于当前 venv
Python: 3.12
```

## 当前上线服务

| 服务 | systemd unit | 工作目录 | 端口 | 用户 | GPU 配额 | 状态 |
|---|---|---|---:|---|---:|---|
| AudioSeparator | `audio-separator.service` | `/home/cjh/code/AudioSeparator` | `83` | `cjh` | `AS_GPU_FRACTION=0.9` | 已上线 |
| SubtitleRemover | `subtitle-remover.service` | `/home/cjh/code/SubtitleRemover` | `84` | `cjh` | `SR_GPU_FRACTION=0.9` | `0.2.0-phaseC` |

健康检查：

```bash
curl http://127.0.0.1:83/health
curl http://127.0.0.1:84/health
```

`83` 和 `84` 都是 `<1024` 特权端口，两个 unit 均通过
`AmbientCapabilities=CAP_NET_BIND_SERVICE` 允许非 root 服务用户绑定端口。

## SubtitleRemover Phase C

| 模块 | 状态 | 文件 | 说明 |
|---|:---:|---|---|
| 配置管理 | 已实现 | `src/config.py` | 支持 `SR_*` 环境变量覆盖 |
| 字幕区域检测 | 已实现 | `src/detection/roi_detector.py`, `src/detection/ocr_detector.py` | ROI / EasyOCR / PaddleOCR |
| 逐帧去字幕 | 已实现 | `src/inpainting/opencv_inpaint.py`, `src/inpainting/deep_inpaint.py` | OpenCV TELEA / LaMa |
| VSR / 画质增强 | 已实现 | `src/vsr/real_esrgan.py`, `src/vsr/basicvsr.py` | 推荐 `real-esrgan`；`basicvsr++` 需 basicsr-native checkpoint |
| 视频管线 | 已实现 | `src/pipeline.py` | ffmpeg decode/encode、音频保留、串行任务 |
| FastAPI 服务 | 已实现 | `src/api_server.py` | `/health`、`/info`、上传、状态轮询、下载 |
| API 文档 | 已实现 | `docs/api.md` | phase C 接口和参数矩阵 |

生产路径 `/home/cjh/code/SubtitleRemover` 是 systemd 正在使用的工作目录，
不要把它当作空白开发副本。常规修改应先在隔离 worktree 中完成、验证、提交并推送，
再在服务器侧按需拉取。

## SubtitleRemover systemd 单元

实际 unit：

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

关键点：

- `src/api_server.py` 通过 `config.SERVICE_PORT` 调用 `uvicorn.run(..., port=...)`，因此尊重 `SR_PORT=84`。
- `SR_GPU_FRACTION=0.9` 会在进程启动早期调用 `torch.cuda.set_per_process_memory_fraction`。
- 当前 venv 冻结在仓库 `requirements.lock`，来源为 `/home/cjh/code/SubtitleRemover/venv/bin/pip freeze`。

## 重启门禁

不要直接重启生产服务。重启前先完成：

```bash
cd /home/cjh/code/SubtitleRemover

# 1. import dry-run
sudo -u cjh /home/cjh/code/SubtitleRemover/venv/bin/python \
  -c "from src import api_server; print('import ok')"

# 2. 确认当前线上健康状态
curl http://127.0.0.1:84/health
```

如果导入失败、健康检查异常，或依赖版本与 `requirements.lock` 不一致，先停止并排查；
不要用 restart 覆盖仍在内存中正常运行的旧进程。

## 环境变量

| 变量 | 代码默认值 | 生产值 | 说明 |
|---|---:|---:|---|
| `SR_PORT` | `8082` | `84` | FastAPI 监听端口 |
| `SR_GPU_FRACTION` | `0.4` | `0.9` | 进程级显存上限 |
| `SR_DEVICE` | `cuda` | 未覆盖 | 计算设备 |
| `SR_MAX_WIDTH` | `2560` | 未覆盖 | 最大输入宽度 |
| `SR_MAX_HEIGHT` | `1440` | 未覆盖 | 最大输入高度 |
| `SR_MAX_DURATION` | `60` | 未覆盖 | 最大视频时长，分钟 |
| `SR_DETECTION_MODE` | `auto` | 未覆盖 | `auto` / `roi` / `ocr` |
| `SR_INPAINT_METHOD` | `opencv` | 未覆盖 | `opencv` / `lama` |
| `SR_VSR_MODEL` | `basicvsr++` | 请求参数通常覆盖 | 推荐请求使用 `real-esrgan` |
| `SR_VSR_TILE_SIZE` | `0` | 未覆盖 | BasicVSR++ 空间分块 |

## 统一反向代理示例

如果需要在 80/443 前面统一暴露两个服务，可按实际端口反代：

```caddyfile
:80 {
    handle_path /separate/* {
        reverse_proxy localhost:83
    }

    handle_path /remove-subtitle/* {
        reverse_proxy localhost:84
    }

    handle {
        root * /var/www/html
        file_server
    }
}
```

## 相关文档

- [../localserver.md](../localserver.md) — LocalServer 服务器信息
- [api.md](api.md) — SubtitleRemover phase C API
- [handoff-2026-05-06-server-state.md](handoff-2026-05-06-server-state.md) — 2026-05-06 服务器状态交接
- [../README.md](../README.md) — 项目概览
