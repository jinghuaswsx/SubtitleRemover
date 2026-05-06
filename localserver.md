# LocalServer 部署信息

## 服务器

- **IP**: `172.30.254.14`
- **服务用户**: `cjh`
- **systemd 管理**: 需要 root / sudo
- **SSH Key**: `C:\Users\admin\.ssh\CC.pem`

## 服务

### SubtitleRemover

- **仓库**: https://github.com/jinghuaswsx/SubtitleRemover
- **状态**: 已上线，`0.2.0-phaseC`
- **systemd**: `subtitle-remover.service`
- **端口**: `84`
- **健康检查**: `http://127.0.0.1:84/health`
- **部署路径**: `/home/cjh/code/SubtitleRemover`
- **执行环境**: `/home/cjh/code/SubtitleRemover/venv`
- **GPU 配额**: `SR_GPU_FRACTION=0.9`，约 14GB / 16GB
- **支持分辨率**: 1080p / 2K
- **特权端口**: `84 < 1024`，systemd unit 保留 `CAP_NET_BIND_SERVICE`

> `/home/cjh/code/SubtitleRemover` 是生产工作目录，不要当作空白开发副本使用。
> 重启 `subtitle-remover.service` 前必须先完成 import dry-run 和健康检查。

### AudioSeparator

- **仓库**: https://github.com/jinghuaswsx/AudioSeparator
- **状态**: 已上线
- **systemd**: `audio-separator.service`
- **端口**: `83`
- **健康检查**: `http://127.0.0.1:83/health`
- **部署路径**: `/home/cjh/code/AudioSeparator`
- **GPU 配额**: `AS_GPU_FRACTION=0.9`
- **默认 preset**: `vocal_balanced`

### 依赖

- Python 3.12+
- CUDA 12.x
- ffmpeg
- PyTorch (CUDA)
- OpenCV / FastAPI / Uvicorn
- EasyOCR / PaddleOCR
- LaMa / Real-ESRGAN / BasicSR

当前生产 venv 已冻结到仓库 `requirements.lock`。如果需要重建环境，优先以该 lock 对齐，再评估是否收敛回更小的 pinned runtime 集合。

---

## 旧远程服务器（已弃用）

- **IP**: `14.103.220.208`
- **登录用户**: `root`
- **SSH Key**: `C:\Users\admin\.ssh\openclaw-noobird.pem`
