# SubtitleRemover API

服务地址：`http://172.30.254.14:84`（本机回环：`http://127.0.0.1:84`）
版本：`0.2.0-phaseC`
部署方式：systemd 单元 `subtitle-remover.service`（开机自启）

---

## 端点速查

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/health` | 存活探针 |
| GET | `/info` | 列出所有可选模式与推荐组合 |
| POST | `/remove-subtitle` | 上传视频，返回 `task_id` |
| POST | `/vace-edit` | VACE 视频编辑（独立路由，与 `/remove-subtitle` 共享 QUEUE） |
| GET | `/status/{task_id}` | 查询任务进度（subtitle / vace 任务通用） |
| GET | `/download/{task_id}` | 下载结果 mp4（subtitle / vace 任务通用） |

任务**异步**执行：POST 后立即返回 `task_id`，需轮询 `/status` 直到 `state=done` 再 `/download`。
单进程单 worker，串行处理，**同时只跑一个任务**，多余请求排队。
`/remove-subtitle` 与 `/vace-edit` 共享同一个 QUEUE 与 worker，因此两类任务**严格串行**，
互相之间也按入队顺序排队。

---

## GET `/health`

```json
{ "status": "ok", "phase": "C", "queued": 0, "tasks_in_memory": 2 }
```

---

## GET `/info`

返回所有合法参数取值与推荐最佳质量组合：

```json
{
  "version": "0.2.0-phaseC",
  "detection": ["auto", "ocr", "roi"],
  "ocr_engine": ["easyocr", "paddle"],
  "inpaint": ["lama", "opencv"],
  "vsr": ["basicvsr", "basicvsr++", "basicvsrpp", "none", "off",
          "real-esrgan", "realesrgan"],
  "max_resolution": [2560, 1440],
  "max_duration_min": 60,
  "supported_inputs": ["mp4", "mkv", "avi", "mov", "webm"],
  "recommended_best_quality": {
    "detection": "ocr",
    "ocr_engine": "easyocr",
    "inpaint": "lama",
    "vsr": "real-esrgan"
  }
}
```

---

## POST `/remove-subtitle`

`multipart/form-data` 上传一个视频文件，返回 `task_id`。

### 表单字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|:---:|---|---|
| `file` | file | ✅ | — | 视频文件 |
| `detection` | string | ❌ | `auto` | 字幕区域检测方式：`auto` / `roi` / `ocr` |
| `ocr_engine` | string | ❌ | `easyocr` | OCR 引擎：`easyocr` / `paddle`，仅 `detection=ocr/auto` 时生效 |
| `inpaint` | string | ❌ | `opencv` | 去字幕方法：`opencv`（快）/ `lama`（高画质）|
| `vsr` | string | ❌ | `off` | 画质增强：`off` / `real-esrgan` / `basicvsr++` |
| `roi` | string | ❌ | `bottom_20%` | 字幕区域提示，详见下文 |

### `roi` 取值格式

- `bottom_<N>%` — 从底部起 N% 高的横向条带（最常用，`bottom_20%`）
- `top_<N>%`    — 顶部 N% 条带
- `x,y,w,h`     — 像素坐标矩形（左上 + 宽高），自动 clamp 到画面内

`detection=ocr` 时 `roi` 作为**约束提示**：OCR 检测到的文字框若完全落在该 ROI 内才采纳，避免把画面中部的非字幕文字误删。

### 模式组合矩阵

| 用例 | detection | inpaint | vsr | 显存（1080p）| 时间（3min 1080p）|
|---|---|---|---|:---:|:---:|
| 极速预览 | `roi` | `opencv` | `off` | ~0.3 GB | ~2 min |
| 平衡（OCR 自动定位 + 快速去除）| `ocr` | `opencv` | `off` | ~1.5 GB ↓ ~0.5 GB | ~3 min |
| **推荐最佳** ⭐ | `ocr` | `lama` | `real-esrgan` | **~3-4 GB** | **~12-18 min** |
| 中文密集字幕 | `ocr` + `ocr_engine=paddle` | `lama` | `real-esrgan` | ~4 GB | ~13-19 min |
| 固定位置 + 高画质 | `roi` | `lama` | `real-esrgan` | ~3-4 GB | ~12-18 min |

### 错误码

| HTTP | 触发条件 |
|---|---|
| 400 | 未知 `detection` / `ocr_engine` / `inpaint` / `vsr` 取值 |
| 415 | 文件扩展名不在 `supported_inputs`（mp4/mkv/avi/mov/webm）|

### 响应

```json
{ "task_id": "8c38aaf3c39d4e21b53932f274ce7430", "state": "queued" }
```

---

## POST `/vace-edit`

VACE 视频编辑（Wan2.1-VACE-1.3B）独立路由。详细实施规范见
[docs/vace_integration_plan.md](vace_integration_plan.md)。

**默认禁用**：`SR_VACE_ENABLED=False`（默认）时整路由返回 `503`。生产端口 `84`
当前阶段 1 状态下，本路由处于**未启用**状态；阶段 2 部署后才开启。

### 表单字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|:---:|---|---|
| `file` | file | ✅ | — | 源视频（沿用 `supported_inputs`）|
| `prompt` | string | ✅ | — | VACE 文本编辑指令 |
| `negative_prompt` | string | ❌ | `""` | 负面提示 |
| `mask_mode` | string | ❌ | `none` | `none` / `roi` / `mask_file` |
| `roi` | string | ❌ | — | `mask_mode=roi` 时必填，格式同 `/remove-subtitle` |
| `mask_file` | file | ❌ | — | `mask_mode=mask_file` 时必填，二值 mask 视频 |
| `profile` | string | ❌ | `rtx4070tis_balanced` | `rtx4070tis_fast` / `rtx4070tis_balanced` / `rtx4070tis_quality` |
| `seed` | int | ❌ | `-1` | 随机种子，`-1` = 随机 |

### 错误码

| HTTP | 触发条件 |
|---|---|
| 400 | 未知 `mask_mode` / 未知 `profile` |
| 415 | 不支持的输入格式 |
| 422 | `prompt` 缺失；`mask_mode=roi` 缺 `roi`；`mask_mode=mask_file` 缺 `mask_file` |
| 503 | 本服务未启用 VACE（`SR_VACE_ENABLED=False`）|

### 响应

```json
{ "task_id": "8c38aaf3c39d4e21b53932f274ce7430", "state": "queued" }
```

后续轮询 `/status/{task_id}` 与下载 `/download/{task_id}` 与 `/remove-subtitle`
完全一致。

---

## GET `/status/{task_id}`

```json
{
  "task_id": "8c38aaf3c39d4e21b53932f274ce7430",
  "state": "running",          // queued | running | done | failed
  "progress": 0.62,            // 0.0 - 1.0
  "error": null,
  "created_at": 1778051335.46,
  "finished_at": null
}
```

`progress` 每 30 帧上报一次；`state=failed` 时 `error` 字段含异常信息。

| HTTP | 触发条件 |
|---|---|
| 404 | `task_id` 不存在（进程重启后任务表会清空）|

---

## GET `/download/{task_id}`

返回 `Content-Type: video/mp4`，文件名 `{task_id}.mp4`。

| HTTP | 触发条件 |
|---|---|
| 404 | task 不存在 |
| 409 | task 未完成（state ≠ done）|
| 410 | 输出文件已被清理 |

---

## 完整示例

### 1. 推荐最佳质量调用

```bash
HOST="http://172.30.254.14:84"

# 上传
TASK=$(curl -s -F "file=@input.mp4" \
  -F "detection=ocr" \
  -F "ocr_engine=easyocr" \
  -F "inpaint=lama" \
  -F "vsr=real-esrgan" \
  $HOST/remove-subtitle | jq -r .task_id)
echo "task=$TASK"

# 轮询
while :; do
  S=$(curl -s $HOST/status/$TASK)
  STATE=$(echo "$S" | jq -r .state)
  PROG=$(echo "$S" | jq -r .progress)
  echo "$STATE $PROG"
  [[ "$STATE" == "done" || "$STATE" == "failed" ]] && break
  sleep 5
done

# 下载
[[ "$STATE" == "done" ]] && curl -s -o output.mp4 $HOST/download/$TASK
```

### 2. 已知字幕在底部 25% 区域，固定 ROI 加速

```bash
curl -s -F "file=@input.mp4" \
     -F "detection=roi" \
     -F "roi=bottom_25%" \
     -F "inpaint=lama" \
     -F "vsr=real-esrgan" \
     http://172.30.254.14:84/remove-subtitle
```

### 3. 中文密集字幕用 PaddleOCR

```bash
curl -s -F "file=@input.mp4" \
     -F "detection=ocr" \
     -F "ocr_engine=paddle" \
     -F "inpaint=lama" \
     -F "vsr=real-esrgan" \
     http://172.30.254.14:84/remove-subtitle
```

### 4. 自定义像素矩形（已知字幕在 (100, 900) 起 1700×120 区域）

```bash
curl -s -F "file=@input.mp4" \
     -F "detection=roi" \
     -F "roi=100,900,1700,120" \
     -F "inpaint=lama" \
     http://172.30.254.14:84/remove-subtitle
```

### 5. Python 客户端

```python
import time, requests, sys

HOST = "http://172.30.254.14:84"

def remove_subtitle(path: str, **opts) -> str:
    with open(path, "rb") as f:
        r = requests.post(f"{HOST}/remove-subtitle",
                          files={"file": f}, data=opts)
    r.raise_for_status()
    task = r.json()["task_id"]
    while True:
        s = requests.get(f"{HOST}/status/{task}").json()
        print(f"{s['state']:>8} {s['progress']:.1%}")
        if s["state"] == "done":
            out = path.rsplit(".", 1)[0] + "_clean.mp4"
            with open(out, "wb") as fout:
                fout.write(requests.get(f"{HOST}/download/{task}").content)
            return out
        if s["state"] == "failed":
            raise RuntimeError(s["error"])
        time.sleep(5)

remove_subtitle(sys.argv[1],
                detection="ocr", inpaint="lama", vsr="real-esrgan")
```

---

## 限制与注意事项

| 项 | 限制 |
|---|---|
| 最大分辨率 | 2560×1440（2K）— 通过 `SR_MAX_WIDTH/HEIGHT` 调整 |
| 最大时长 | 60 分钟（`SR_MAX_DURATION` 调整）|
| 输入格式 | mp4 / mkv / avi / mov / webm |
| 输出格式 | mp4（H.264 / yuv420p / preset=fast）|
| 并发 | **串行**，多任务排队（`queued` 状态）|
| 任务持久化 | 内存 only — 服务重启丢失任务记录，但 `uploads/` 与 `output/` 文件保留 |
| GPU 显存上限 | 进程级 90% × 16GB = ~14 GB（`SR_GPU_FRACTION`）|
| `vsr=basicvsr++` | **运行不可用**：mmediting 与 basicsr 的 SPyNet 通道宽度不一致；使用 `real-esrgan` 替代 |

---

## 服务运维

生产进程当前由 systemd 管理。重启前先做 import dry-run 和健康检查；如果失败，
不要用 restart 覆盖仍在内存中正常运行的进程。

```bash
# 查看状态 / 日志
sudo systemctl status subtitle-remover
sudo journalctl -u subtitle-remover -f

# 重启前验证
sudo -u cjh /home/cjh/code/SubtitleRemover/venv/bin/python \
  -c "from src import api_server; print('import ok')"
curl http://127.0.0.1:84/health

# 验证通过后再重启 / 停止 / 启动
sudo systemctl restart subtitle-remover
sudo systemctl stop subtitle-remover
sudo systemctl start subtitle-remover

# 改环境变量后必须 reload + restart
sudo systemctl edit subtitle-remover     # 改 unit 覆写
sudo systemctl daemon-reload
sudo systemctl restart subtitle-remover
```

### 关键环境变量（systemd unit 中已设）

| 变量 | 当前值 | 说明 |
|---|---|---|
| `SR_PORT` | `84` | 监听端口（绑定特权端口靠 `CAP_NET_BIND_SERVICE`）|
| `SR_GPU_FRACTION` | `0.9` | 进程级显存占比（`torch.cuda.set_per_process_memory_fraction`）|
| `SR_MAX_WIDTH` | `2560` | 最大输入宽度 |
| `SR_MAX_HEIGHT` | `1440` | 最大输入高度 |
| `SR_MAX_DURATION` | `60` | 最大输入时长（分钟）|
| `SR_BATCH_SIZE` | `4` | VSR 批大小 / 滑窗 |
| `SR_VSR_TILE_SIZE` | `0` | VSR 空间分块（0 = 全帧）|

模型权重缓存：

```
~/.EasyOCR/model/                              # EasyOCR ~100MB
~/.cache/torch/hub/checkpoints/big-lama.pt     # LaMa  ~200MB
~/.paddlex/official_models/                    # PaddleOCR PP-OCRv5 ~600MB
/home/cjh/code/SubtitleRemover/models/         # Real-ESRGAN ~64MB, BasicVSR++ ~30MB
```
