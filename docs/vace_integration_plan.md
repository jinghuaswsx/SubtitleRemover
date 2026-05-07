> 📌 **本仓库实施规范**：本文档是 SubtitleRemover 内 VACE 接入工作的**权威锚点**。
> 上游知识来自 [docs/vace_4070ti_super_plan.md](vace_4070ti_super_plan.md)（同步自
> AutoVideoSrtLocal）；本规范在 §2 列出本仓库相对于上游的明确偏离。代码改动以本规范为准。

# VACE 接入实施规范（SubtitleRemover）

## 1. 范围与目标

在现有 SubtitleRemover 服务（`/home/cjh/code/SubtitleRemover`，systemd 单元
`subtitle-remover.service`，端口 `84`）中接入 Wan2.1-VACE-1.3B，作为**独立的视频
编辑/重绘路由** `POST /vace-edit`。VACE 与现有 `/remove-subtitle` 共享同一个
进程内 `QUEUE` + 单 worker 线程，从而保证**同一时间只跑一个任务**（VACE 与
VSR 严格串行）。

**目标硬件**：本机 NVIDIA GeForce RTX 4070 Ti SUPER 16 GB，已与 AudioSeparator
（:83）共卡。VACE 仅做 1.3B / 480p 主路径，可选 720p（仅 GPU 独占模式）；不引入
14B（显存装不下）。

**非目标**：不改 `/remove-subtitle` 既有契约；不升 PyTorch；不引入跨语言/跨容器
的微服务化拆分。

## 2. 与上游 plan 的偏离（必读）

| 项 | 上游方案（AutoVideoSrtLocal） | 本仓库方案 | 原因 |
|---|---|---|---|
| 平台 | Windows，`msvcrt.locking` | **Linux**，`fcntl.flock` | 生产部署在 Linux |
| 服务边界 | 三独立进程（audio / subtitle / VACE） | **VACE 在 SubtitleRemover 进程内** | 复用单 worker QUEUE 即可串行；不引入新服务进程降低运维面 |
| 进程内串行 | 不涉及 | **同一 QUEUE + Task.kind 分发** | 见 §4 |
| 跨进程 GPU 锁 | "必装" | **阶段 2 才接入** | 阶段 1 dry-run 不真吃显存；阶段 2 实跑 + 与 :83 audio 共卡时再加 |
| profile 命名 | `rtx4070tis_{fast,balanced,quality}` | **同名沿用** | 保持跨仓库可引用 |
| `appcore/vace_subtitle/config.py` | 在 PROFILES 增加 3 档 | 本仓库新建 `src/vace/config.py`，**只放 4070 Ti Super 三档**，不复制 `rtx3060_*` | 本仓库不部署 3060；保持精简 |
| `tools/audio_separator/api_server.py` 改 env | 是（fraction 0.5→env） | **不动** | 不在本仓库范围；audio :83 由独立仓库管 |

### 2.1 与本仓库 Y 路径共存

主分支已合入并行实现 [src/inpainting/vace_adapter.py](../src/inpainting/vace_adapter.py)
（commit `9625ce0`），把 VACE 作为 `/remove-subtitle?inpaint=vace-1.3b` 的
**inpaint 后端**（subprocess 调外部 VACE checkout，env: `SR_VACE_SCRIPT` /
`SR_VACE_CKPT_DIR` 等）。本规范定义的 X 路径（独立路由 `/vace-edit`）与之**并存**：

| 维度 | Y 路径（`inpaint=vace-1.3b`） | X 路径（`/vace-edit`） |
|---|---|---|
| 调用入口 | 既有 `/remove-subtitle` 路由 | 新增 `/vace-edit` 路由 |
| 触发方式 | 表单字段 `inpaint=vace-1.3b` | 整路由专属 |
| 用途定位 | "用 VACE 去字幕"（mask 由 OCR/ROI 自动产出） | "用 VACE 视频编辑"（用户提供 prompt + 可选 mask）|
| Prompt | 来自 env `SR_VACE_PROMPT` 默认值 | 路由必填 `prompt` |
| Mask | OCR / ROI 自动 mask 视频 | `mask_mode=none/roi/mask_file` |
| 时长上限 | 默认 6s（POC 限制） | 由 profile 的 `chunk_seconds × N` 决定 |
| Profile 选择 | 不可选（用 `SR_VACE_*` 静态参数） | 路由参数 `profile` 三档可选 |
| 阶段 2 后端 | 已用 subprocess 桥（env `SR_VACE_SCRIPT`） | **复用**同一 subprocess 桥（见 §7.2） |

阶段 2 实现 X 路径真跑时，`src/vace/runner.py:_run_real()` 直接复用
`src/inpainting/vace_adapter.VaceConfig.build_command()` 思路（或直接调用一个
共享 `_invoke_vace()` 函数），保持单一外部依赖配置 `SR_VACE_SCRIPT` /
`SR_VACE_CKPT_DIR`，避免重复在 SR venv 装 wan2.1。

## 3. 路由设计：`POST /vace-edit`

### 3.1 表单字段

| 字段 | 类型 | 必填 | 默认 | 说明 |
|---|---|:---:|---|---|
| `file` | file | ✅ | — | 源视频（mp4/mkv/avi/mov/webm，沿用 `SUPPORTED_INPUTS`） |
| `prompt` | string | ✅ | — | VACE 文本编辑指令（≥ 1 字符）|
| `negative_prompt` | string | ❌ | `""` | 负面提示 |
| `mask_mode` | string | ❌ | `none` | `none` / `roi` / `mask_file` |
| `roi` | string | ❌ | — | `mask_mode=roi` 时必填，格式同 `/remove-subtitle`（`bottom_N%` / `top_N%` / `x,y,w,h`） |
| `mask_file` | file | ❌ | — | `mask_mode=mask_file` 时必填，与 `file` 同帧数同分辨率的二值视频 |
| `profile` | string | ❌ | `rtx4070tis_balanced` | 取值见 §5 |
| `seed` | int | ❌ | `-1` | 随机种子，`-1` = 随机 |

### 3.2 错误码

| HTTP | 触发条件 |
|---|---|
| 400 | 未知 `mask_mode`、未知 `profile` |
| 415 | `file` 扩展名不支持 |
| 422 | `prompt` 缺失 / 空字符串；`mask_mode=roi` 但缺 `roi`；`mask_mode=mask_file` 但缺 `mask_file` |
| 503 | `SR_VACE_ENABLED=False`（默认）→ 路由整体返回 503 |

### 3.3 状态机

`POST /vace-edit` 返回 `{task_id, state}`，与 `/remove-subtitle` **完全一致**。
后续轮询 `GET /status/{task_id}`、下载 `GET /download/{task_id}` 两个**已存在**
路由共用，按 `Task.kind` 内部分发输出文件。

## 4. 串行机制

`src/api_server.py` 中已有的 `QUEUE: Queue[str]` 和 `_worker()` 单线程**复用**：

- `Task` dataclass 增加 `kind: str = "subtitle"`，取值 `"subtitle"` / `"vace"`，并增加 VACE 专属字段（`prompt` / `mask_mode` / `mask_path` / `profile` / `seed` / `negative_prompt`）。
- `_worker()` 取出 `task_id` 后按 `kind` 分发：
  - `kind="subtitle"` → `from .pipeline import process_video`（既有逻辑，零改动）
  - `kind="vace"` → `from .vace.runner import process_vace`
- 由于 worker 是**单线程**，FIFO 入队即天然串行，**不需要额外的进程内锁**。
- 同时，所有任务 `state` 流转、`progress` 上报、错误处理共用现有路径。

## 5. Profile 表

写在 `src/vace/config.py`。`size` 字段直接使用 Wan2.1 上游 `SIZE_CONFIGS` 的键：

| Profile | model | size | frame_num | sample_steps | offload_model | t5_cpu | chunk_seconds | 适用场景 |
|---|---|---|---|---|---|---|---|---|
| `rtx4070tis_fast` | vace-1.3B | `832*480` | 17 | 20 | False | False | 1.0 | 共卡可用（VAE 峰值 <300 MiB）|
| `rtx4070tis_balanced` *(默认)* | vace-1.3B | `832*480` | 81 | 25 | False | False | 4.8 | 与 audio :83 共卡也稳 |
| `rtx4070tis_quality` | vace-1.3B | `832*480` | 81 | 30 | False | False | 4.8 | 仅靠更高 sample_steps 提质 |

**重要约束**（来自 Wan2.1 `SUPPORTED_SIZES`）：
- `vace-1.3B` 仅支持 `480*832` / `832*480`（无 720p / 1080p 档位）
- `vace-14B` 才支持 `720*1280` / `1280*720`，但需 24+ GB 显存，本卡禁用
- 因此 `quality` 仍用 1.3B + 832*480，仅以更高 `sample_steps=30` 换更稳画质，
  并不真正切换到 720p 模型

**OOM fallback chain**（`fallback_profile()`）：
`rtx4070tis_quality` → `rtx4070tis_balanced` → `rtx4070tis_fast` → `None`

**`frame_num` 约束**：必须满足 `4n+1`，上限 81。

## 6. 显存预算与跨进程协调

### 6.1 当前状态（共卡）

| 服务 | 端口 | 进程 fraction | 实际显存上限 |
|---|---:|---:|---:|
| AudioSeparator | 83 | `AS_GPU_FRACTION=0.9` | 14.4 GB |
| SubtitleRemover | 84 | `SR_GPU_FRACTION=0.9` | 14.4 GB |

两家**单独**跑都没问题；**同时**跑（audio + VACE 1.3B）瞬时显存 ~20 GB > 16 GB 必爆。

### 6.2 阶段 1：进程内串行已足够

阶段 1 走 dry-run，不真实分配显存，**不引入跨进程锁**。

### 6.3 阶段 2：fcntl 跨进程文件锁（Linux）

`src/gpu_lock/file_lock.py` 提供 `cross_process_gpu_lock(timeout=300)`
context manager，使用 `fcntl.flock(LOCK_EX|LOCK_NB)` + 轮询 + 超时。锁文件路径
取自 env `SR_GPU_LOCK_FILE`（默认 `/tmp/gpu.lock`）。

接入点：
- VACE：`src/vace/runner.py` 在调 wan 推理前 `with cross_process_gpu_lock():`
- VSR / `/remove-subtitle`：**暂不接入**，理由：本仓库进程内 worker 单线程，
  不会自我撞车；audio 服务侧若也启用同一锁文件，则 audio 推理段也会被串行化。
  **是否在 audio 侧接入由 audio 服务方自行决定**，本仓库只保证 VACE 侧拿锁。

env 开关 `SR_GPU_LOCK_ENABLED`（默认 `True`，阶段 2 启用时）。设为 `False`
等价禁用（用于压测 / 单服务独占场景）。

## 7. 阶段切分与 commit 计划

### 7.1 阶段 1 — 文档 + 骨架 + dry-run + 测试（commit 1, 2）

**commit 1**：本文档 + `docs/api.md` 增 VACE 章节 + `README.md` 端点表更新。**零代码改动**。

**commit 2**（已实现，待部署）：
- 新建 `src/vace/{__init__,config,runner}.py`
- 改 `src/config.py`：增 VACE env 配置
- 改 `src/api_server.py`：`Task.kind` + 新路由 + worker dispatch + `/info`/`/health` 增字段
- 新建 `tests/test_vace_profile.py`、`tests/test_vace_route_dry_run.py`
- 默认 `SR_VACE_ENABLED=False`，`SR_VACE_DRY_RUN=True`
- **生产 :84 行为零变化**（路由返回 503，`/remove-subtitle` 完全不变）
- 测试基线：`pytest`/`unittest discover` ≥ 17 passed（3 既有 + 5 profile + 9 route dry-run）

阶段 1 部署：合并到 master、push、生产 `cd /home/cjh/code/SubtitleRemover && git pull`、
import dry-run、`systemctl restart`、`/health` 与 `/info` 验证。

### 7.2 阶段 2 — 真跑接入（commit 3，已实现）

代码层面：
- 新建 `src/gpu_lock/{__init__,file_lock}.py`（Linux `fcntl.flock` 实现）
- `src/vace/runner.py` 实现 `_run_real()`，**不在 SR venv 装 wan2.1**——通过
  `subprocess` 调用外部 VACE checkout（与 Y 路径 `src/inpainting/vace_adapter.py`
  共用同一桥），env: `SR_VACE_PYTHON` / `SR_VACE_SCRIPT` / `SR_VACE_CKPT_DIR`
- profile 的 `model_name` / `size` / `frame_num` / `sample_steps` / `offload_model`
  / `t5_cpu` 全部传给外部 VACE 命令行
- mask 处理：`mask_mode=roi` 复用 `vace_adapter._write_mask_video`；
  `mask_mode=mask_file` 直传上传文件；`mask_mode=none` 在 real 模式拒绝
- `SR_GPU_LOCK_ENABLED=1` 时，`subprocess.run` 包在 `cross_process_gpu_lock`
  里，避免与 audio :83 服务同时占用 GPU
- 测试：3 GPU lock + 3 runner real（mock subprocess）

**未做**（保持 minimal）：
- 不在本仓库 `requirements.txt` 加 wan2.1（外部 venv 管理）
- 不写权重下载脚本（外部 VACE checkout 自管模型）
- 不接入 Y 路径的 GPU lock（不在 X scope；如果 audio + Y 共卡也撞车，
  由 Y 路径维护方在 `vace_adapter.VaceSubtitleRemover.remove` 加同名锁）

阶段 2 部署步骤详见 `deploy.md` 的 "VACE 阶段 2 启用" 一节。

## 8. 测试矩阵

| 测试文件 | case 数 | 阶段 |
|---|---:|---|
| `tests/test_phase_c_contracts.py` | 3 *(已有，不破)* | 1 |
| `tests/test_vace_profile.py` | 4 | 1 |
| `tests/test_vace_route_dry_run.py` | 4-6 | 1 |
| `tests/test_gpu_lock.py` | 3 | 2 |
| `tests/test_vace_runner_real.py` *(可选)* | 1（标记 `@pytest.mark.slow`，CI 跳过） | 2 |

## 9. 验收标准

### 9.1 阶段 1（不实跑 VACE）

- 所有单元测试通过：`pytest tests/ -v` ≥ 11 passed。
- 生产 `/health` 仍返回 `phase=C, status=ok`，新增 `vace_enabled` 字段。
- 生产 `/info` 含 `vace_profile: [rtx4070tis_fast, rtx4070tis_balanced, rtx4070tis_quality]`。
- `POST /vace-edit` 在默认 env 下返回 503（disabled）。
- `POST /remove-subtitle` 行为零变化（既有调用方无回归）。

### 9.2 阶段 2（实跑）

- `SR_VACE_ENABLED=1` 下 `POST /vace-edit` 入队、worker 处理、`/status` 100% 完成、`/download` 拉到结果 mp4。
- 1080p / 5s 视频 + `profile=rtx4070tis_balanced` 端到端无 OOM。
- 三服务并发（audio + VACE）显存峰值 ≤ 15.5 GB（fcntl 锁工作）。
- 速度数据回填上游 [vace_4070ti_super_plan.md](vace_4070ti_super_plan.md) §2.2。

## 10. 不要做的事

- ❌ 不引入 14B 模型（4070 Ti Super 装不下）。
- ❌ 不升 PyTorch 到 nightly。
- ❌ 不修改 `/remove-subtitle` 既有契约。
- ❌ 不在阶段 1 真实下载 VACE 权重（避免 ~10 GB 网络 + 磁盘开销）。
- ❌ 不在生产 venv 直接 `pip install` 未在 worktree 验证过的依赖。
- ❌ 不复制 `rtx3060_*` profile（本仓库不部署 3060）。

## 10.0 当前状态（2026-05-07，暂停）

> **`/vace-edit` 已临时关闭**（`SR_VACE_ENABLED=0`，路由返 503）。
> 代码、外部 venv、Wan2.1 18 GB 权重、systemd drop-in 全部保留；待解决
> §10.1 的 VAE 显存协调问题后切回 `SR_VACE_ENABLED=1` + `SR_VACE_DRY_RUN=0`
> 即可恢复，无需重装。

恢复步骤（一次到位）：
```bash
sudo sed -i 's/SR_VACE_ENABLED=0/SR_VACE_ENABLED=1/; s/SR_VACE_DRY_RUN=1/SR_VACE_DRY_RUN=0/' \
  /etc/systemd/system/subtitle-remover.service.d/vace.conf
sudo systemctl daemon-reload && sudo systemctl restart subtitle-remover
```

## 10.1 已知问题（阶段 2 实测 2026-05-07）

阶段 2 完整部署后，**路由层全部上线**（`/health.vace_enabled=true`、systemd
drop-in 生效、外部 venv + 模型权重就位、subprocess 能正常拉起 generate.py），
但**VACE 1.3B 实际推理在本机 16 GB 显存下 OOM**：

| 配置 | 实际 GPU 峰值 | 结果 |
|---|---|---|
| `frame_num=41` + audio 共卡（2 GB） + LOW_MEM 关 | ~12.1 GB / 13.5 GB 可用 | OOM |
| `frame_num=41` + audio 共卡 + LOW_MEM 开（offload+t5_cpu） | ~11.7 GB / 13.5 GB 可用 | OOM at VAE |
| `frame_num=17` + audio 停 + LOW_MEM 开 | ~13.5 GB / 14.5 GB 可用 | **OOM at VAE** |

后者是**纯 VACE 单跑**（audio + SR 都停），只剩 autovideosrt gunicorn 持有
570 MiB——VACE 1.3B + offload + t5_cpu + frame_num=17 仍要 13.5 GB，远超
上游文档承诺的 6-8 GB。

可能根因（待 TBD 验证）：
- torch 2.11 + cu13 nightly 与 Wan2.1 上游测试矩阵（torch 2.4/2.7）的内存行为差异
- VACE adapter 通道与 1.3B 主干在 fp32 加载（diffusion_pytorch_model.safetensors
  6.7 GB → 实际激活 fp16 × 倍数）
- Wan2.1 VAE.forward 一次性吃整段视频，未分 sub-chunk

**当前 /vace-edit 请求会以 `state=failed` 完成，error=CalledProcessError exit 1。**
不影响 `/remove-subtitle`（contracts test 仍 3/3 通过，phase 1 验收依旧有效）。

后续修复路径（commit 6+，需要新一轮调试）：
1. 降级到 torch 2.4 或 2.7（Wan2.1 测试矩阵）
2. 切到 vace-1.3B 的 fp8 量化分支或社区 fork
3. 在 SR 进程内集成 audio :83 协调（暂停/恢复）以独占 GPU
4. 用更小 chunk_seconds + frame_num=17 + 手动 VAE 分段

## 11. 相关文档

- 上游知识：[docs/vace_4070ti_super_plan.md](vace_4070ti_super_plan.md)
- API：[docs/api.md](api.md)
- 部署：[deploy.md](../deploy.md)
- 服务交接：[docs/handoff-2026-05-06-server-state.md](handoff-2026-05-06-server-state.md)
