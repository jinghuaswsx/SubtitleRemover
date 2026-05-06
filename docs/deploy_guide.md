# 4070 Ti Super 部署指南

## 硬件环境

```
显卡: RTX 4070 Ti Super 16GB
系统: Ubuntu 24.04 LTS (推荐) / Windows
CUDA: 12.4+
Python: 3.12
```

## 两个独立项目

### 1. AudioSeparator — 音频人声分离

```
仓库: https://github.com/jinghuaswsx/AudioSeparator
端口: 80 (默认)
依赖: python-audio-separator, FastAPI, PyTorch CUDA
```

#### 部署步骤

```bash
git clone https://github.com/jinghuaswsx/AudioSeparator.git
cd AudioSeparator
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 配置
export AS_PORT=80
export AS_GPU_FRACTION=0.5        # 留一半显存给 SubtitleRemover
export AS_DEFAULT_PRESET=vocal_balanced

# 启动（自动下载模型 ~3.5GB + 预热）
python api_server.py
```

#### 验证

```bash
curl http://localhost/health
# → {"status":"ok","cuda_available":true,"cuda_device":"RTX 4070 Ti Super"}
```

---

### 2. SubtitleRemover — 视频去字幕

```
仓库: https://github.com/jinghuaswsx/SubtitleRemover
端口: 8082 (规划)
模块: VSR (BasicVSR++)
```

#### 待开发模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 字幕区域检测 | ❌ 待开发 | EasyOCR 或固定 ROI 检测 |
| 逐帧去字幕 | ❌ 待开发 | OpenCV inpainting / LaMa |
| VSR 画质增强 | ⏳ 框架已搭 | BasicVSR++ 集成，需对接 MMEditing |
| FastAPI 服务 | ❌ 待开发 | 任务队列 + 状态轮询 |

#### 开发起点

```bash
git clone https://github.com/jinghuaswsx/SubtitleRemover.git
cd SubtitleRemover

# VSR 模块开发参考:
# - src/vsr/basicvsr.py → BasicVSR++ 集成占位
# - 需安装 MMEditing: pip install mmedit
# - 预训练模型自动下载
```

---

## 服务编排 (systemd, Ubuntu)

两个服务共用一张 4070 Ti Super 16GB，通过 `GPU_MEMORY_FRACTION` 分工：

| 服务 | 显存配额 | 实际占用 |
|------|:-------:|:--------:|
| AudioSeparator | 50% (8GB) | ~4-5GB |
| SubtitleRemover | 40% (6.4GB) | ~6GB (含 VSR) |
| 余量 | 10% | ~2GB 缓冲 |

### AudioSeparator systemd 单元

```
/etc/systemd/system/audio-separator.service
```

```ini
[Unit]
Description=Audio Separator API
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/AudioSeparator
Environment=AS_PORT=80
Environment=AS_GPU_FRACTION=0.5
ExecStart=/opt/AudioSeparator/venv/bin/python api_server.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable audio-separator
systemctl start audio-separator
```

---

## 并发策略

4070 Ti Super 16GB 预估：

| 服务 | 并发模式 | 吞吐 |
|------|---------|------|
| AudioSeparator | 2路 ProcessPool | ~800 任务/小时 (1min 音频) |
| SubtitleRemover | 串行（单个视频任务） | ~15min/视频 (10min 视频) |

**建议**：两个服务分开部署在不同端口，通过 Caddy/Nginx 统一反向代理到 80 端口。

---

## 统一反向代理 (Caddy)

```caddyfile
:80 {
    # 音频分离
    handle_path /separate/* {
        reverse_proxy localhost:80
    }

    # 去字幕（开发中）
    handle_path /remove-subtitle/* {
        reverse_proxy localhost:8082
    }

    # 默认
    handle {
        root * /var/www/html
        file_server
    }
}
```
