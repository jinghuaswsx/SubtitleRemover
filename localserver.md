# LocalServer 部署信息

## 服务器

- **IP**: `172.30.254.14`
- **登录用户**: `root`
- **SSH Key**: `C:\Users\admin\.ssh\CC.pem`

## 服务

### SubtitleRemover (规划中)

- **仓库**: https://github.com/jinghuaswsx/SubtitleRemover
- **端口**: 8082 (规划)
- **部署路径**: `/opt/SubtitleRemover`
- **GPU 配额**: 40% (6.4GB)
- **支持分辨率**: 1080p / 2K

### 依赖

- Python 3.12+
- CUDA 12.x
- ffmpeg
- PyTorch (CUDA)
- MMEditing (BasicVSR++)

---

## 旧远程服务器（已弃用）

- **IP**: `14.103.220.208`
- **登录用户**: `root`
- **SSH Key**: `C:\Users\admin\.ssh\openclaw-noobird.pem`
