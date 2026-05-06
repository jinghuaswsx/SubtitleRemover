# 服务器状态交接 — 2026-05-06

> 来源：本机 Windows 端协作 agent（操作 `/home/cjh/code/SubtitleRemover` 的远程开发副本）
> 接收方：服务器（172.30.254.14, LocalServer）上的开发 agent / cjh
> 状态确认时间：2026-05-06 22:19 CST

## 背景

本机 agent 在主机 `/home/cjh/code/SubtitleRemover` 下执行了一些「初始化开发环境」的操作（创建 venv、装 PyTorch、装 requirements），随后发现该目录其实**已经是 systemd 在用的生产工作目录**，并非空白开发副本。截至确认时刻，服务运行未受影响（PID 841940 4 小时未变，`/health` 返回 200），但留下了若干隐患和待办，请你逐项处理。

**处理前请用只读命令核实所有现状再决定动作。如有任何不确定，先停下、问回主仓库 agent。**

## 现状（已只读核实，仅供参考）

```
service:  subtitle-remover.service (User=cjh, :84, PID 841940,
          active 4h+, version 0.2.0-phaseC)            ✅ 正常对外服务
workdir:  /home/cjh/code/SubtitleRemover
venv:     ./venv  (Python 3.12.3, torch 2.11.0+cu130, CUDA OK)
git HEAD: 52e8074  (与 origin/master 同步)
其他:     audio-separator.service (:83) 也在跑，本次未触碰
          sub2api docker (:82) healthy，与本项目无关
```

## 问题清单

### 问题 1【最高优先级，防代码丢失】未提交的真实开发代码

服务器 working tree 上有大量未提交/未追踪的真实代码，但 `origin/master` 只是占位框架（`src/vsr/basicvsr.py` 还是 `raise NotImplementedError`）。`git status --porcelain`：

```
 M src/vsr/basicvsr.py
?? docs/api.md
?? src/api_server.py
?? src/detection/ocr_detector.py
?? src/detection/roi_detector.py
?? src/inpainting/deep_inpaint.py
?? src/inpainting/opencv_inpaint.py
?? src/pipeline.py
?? src/vsr/real_esrgan.py
```

**请处理：**

1. `git diff` / `git diff src/vsr/basicvsr.py` 把改动 review 一遍
2. 按合理粒度拆 commit（建议至少分组：`detection/` / `inpainting/` / `vsr/` / `api+pipeline` / `docs/api.md`）
3. push 到 `origin/master`
4. 不确定的文件不要直接 add，先和主仓库 agent 对账

### 问题 2【中风险】venv 已被外部 pip 操作污染，重启服务前必须验证

service 用的 venv 就是 `/home/cjh/code/SubtitleRemover/venv`，本机 agent 在不知情的情况下对它做过：

- `pip install --upgrade pip wheel setuptools` → setuptools 从 82.0.1 被 PyTorch 依赖 downgrade 到 70.2.0
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124` → 实际 PyTorch 已是 2.11.0+cu130，命令大多 already-satisfied，但触发了 setuptools 替换
- `pip install -r requirements.txt` → 新装/确认了 `opencv-python 4.13.0`、`av 17.0.1` 等

**当前进程 PID 841940 已加载内存，运行不受影响。** 但下次 `systemctl restart` 或机器重启时，新进程会用新的 site-packages 重新 import，**可能因依赖版本变化导致行为变化或启动失败**。

**请处理：**

1. 立刻冻结当前 venv：
   ```bash
   /home/cjh/code/SubtitleRemover/venv/bin/pip freeze > /tmp/venv-after-pollution.txt
   ```
2. 检查 git 里有没有 `requirements.lock` / `requirements-freeze` / 之前的 pinned 版本，对比差异
3. 如果有 lock，评估是否需要 rollback；如果没有，建议立刻基于运行中的进程 site-packages 生成一份 lock 提交进去
4. **不要直接 `systemctl restart subtitle-remover.service`**。先在 tmux 或独立 venv 跑一次 dry-run：
   ```bash
   sudo -u cjh /home/cjh/code/SubtitleRemover/venv/bin/python -c "from src import api_server; print('import ok')"
   ```
   import 成功 + 端口能起 + `/health` 通过后，再决定是否切换；切换前最好先 `pip install <pinned versions>` 把 setuptools 等回滚到原状。

### 问题 3【低优先】文档与实际部署严重不符

| 文档 | 写的 | 实际 |
|---|---|---|
| `localserver.md` | 端口 `8082 (规划)`，部署路径 `/opt/SubtitleRemover` | 端口 `84`，工作目录 `/home/cjh/code/SubtitleRemover`，已上线 |
| `docs/deploy_guide.md` | 端口 8082 + `/opt/SubtitleRemover` | 同上 |
| `docs/deploy_guide.md` | GPU 配额 40% (6.4GB) | systemd unit 实际 `SR_GPU_FRACTION=0.9` |
| `docs/deploy_guide.md` | 状态「待开发」 | 已是 `0.2.0-phaseC`，detection/inpaint/vsr 多种方案均实现 |

请同步更新这几份文档以反映 phase C 的真实部署。

### 问题 4【信息确认，不一定要改】

- service 用 `Environment=SR_PORT=84`，但 `src/api_server.py` 实际监听端口逻辑是否真的尊重这个 env？请核实，避免 env 改了但代码没读。
- service 设置了 `AmbientCapabilities=CAP_NET_BIND_SERVICE`：理论上是为了绑 <1024 端口；目前 84 < 1024，所以确实需要这个 cap，保持现状即可。

## 处理原则

- 每一步动作前先用只读命令确认现状
- **不要重启 `subtitle-remover.service` / `audio-separator.service`**，除非问题 2 验证通过、且和主仓库 agent 对账
- 完成后请反馈：哪些 commit 推上去了、lock 有没有生成、文档改了哪几行
- 如有任何不确定，stop and ask
