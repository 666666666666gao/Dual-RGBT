**条件边界均值 CPU 裁剪实验源码审查**

2026-10-13。结论：**SOURCE_ONLY PASS，源码阻塞项 0。** 当前脚本可用于限定的三臂 CPU 裁剪诊断。新 `interior_crop` 的实际像素一致性、80-case 完成情况和速度均为 **NOT RUN**；本结论不代表 native 验收或生产接入通过。

```yaml
review_scope: source-only, isolated three-arm CPU crop diagnostic
review_independence: same-family
acceptance_status: provisional
requested_model: gpt-6-astra
requested_reasoning_effort: max
requested_metadata_source: local-codex-policy.md
actual_model_id: UNAVAILABLE
actual_reasoning_effort: UNAVAILABLE
actual_metadata_note: runtime model and effort routing not independently attested
source_verdict: PASS
source_blocking_issues: 0
interior_native_pixel_check: NOT_RUN
interior_native_speed_measurement: NOT_RUN
neural_execution: NOT_RUN
ssh_or_gpu_queries: NOT_RUN
python37_ast_check: PASS
ast_check_host_python: 3.13.12
local_pipeline_source_hashes_checked: 11
local_pipeline_source_hashes_equal_saved_manifest: true
remote_running_source_state: NOT_CHECKED
```

本审查在独立任务上下文中读取源码与既有结果，遵循项目 `AGENTS.md` 和 `C:/Users/gb/.codex/skills/shared-references/local-codex-policy.md`。所请求的 Astra/max 不构成实际路由证明。唯一项目写入为本报告；没有导入或执行实验脚本、调用项目裁剪函数、加载权重、运行神经网络、SSH、GPU 查询或启动任务。

**读取的证据**

| 文件 | SHA256 |
|---|---|
| `scripts/profile_interior_crops.py` | `89abe94cbd0c6504bb0528dca4842d2c5350b9c6f021b243068493e9c823dc6e` |
| `scripts/profile_contiguous_crops.py` | `cbe7c7d6a7115d4ee383bec5ddac84b0de8179d4f2f085b628b8523e421e700f` |
| `dual_rgbt/data.py` | `858e41c8e5218c7f5323a6e18cec00997a48d0323101384b71a73d61ea0280cb` |
| `reports/completed_crop_boundary_audit.json` | `9f33df5231b7faffe715b30066d91f814cdb65b513c58c8b71403ba2d0db656b` |
| `reports/contiguous_crop_cpu_profile.json` | `01ed9ba77bda8c79aa6591c62c77e80a017268293e2747cc934ef5c128cc4727` |
| `reports/contiguous_crop_cpu_native.json` | `fe62c5d43c9d31f2b1afa2247e0b64ea1dea55da832ab0c4fd4aa948cdb4aa5d` |
| `reports/profile_tracking_default.json` | `6356735a99423b03654cecf6ed3929454cd64003a461fb9b635e430fc8ef22f8` |
| `reports/full_pipeline_source_manifest.json` | `9984345d9f0a4377ee45cbc8747b83866a30c91fcb6652f2054166214deca50b` |

M14 的既有 cProfile 记录 `crop_pair` 73 次累计 4.824751 秒，NumPy `ndarray.mean` 146 次累计 3.478360 秒；默认 PyTorch/OpenCV 线程为 32/64。这是调查均值开销的实际依据，不能直接换算为本候选的收益。既有 contiguous native 记录退出 0，40 cases 的 tensor、目标和几何通过精确相等检查，120 对裁剪调用均值为 46.766314/37.609052 ms。其内嵌 report 与单独 profile JSON 完全相同，本审查未重跑它。

重数既有边界审计得到 40 条记录、25 条严格图内和 15 条越界；其 `(sequence, frame, size)` 顺序与已完成 contiguous 40 cases 一致。记录图像尺寸仅为 481×631 和 576×960。这份报告重建的是旧 40 cases 的几何，不是新 80 cases 的像素检查；新增搜索扰动改变后续 identity/seed，不能把 25/40 直接当作新实验跳过均值的比例。

**几何与 bilinear 边界判定**

`profile_interior_crops.py:25–32` 与原 `crop_pair`、已审 `contiguous_crop` 保留相同的中心、面积平方根边长、扰动及 float32 仿射矩阵。守卫在 33 行将这张已经舍入的 float32 矩阵提升为 float64 再求逆；真正 `warpAffine` 仍使用原 float32 矩阵。它没有用未舍入的 `left/top/side` 近似实际映射，也没有更改用于裁剪的矩阵。

OpenCV 文档说明，不设置 `WARP_INVERSE_MAP` 时会先求逆，再由目标坐标映射到源图；所查 OpenCV 4.x 通用 CPU 实现先把输入矩阵转为 float64。该调用方向与守卫一致。[OpenCV 几何变换文档](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)、[OpenCV imgwarp.cpp](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/src/imgwarp.cpp)。

守卫投影 `(0,0)`、`(size-1,0)`、`(0,size-1)`、`(size-1,size-1)` 四个实际输出像素角点。仿射函数各坐标的极值在矩形角点取得；当前变换更是无旋转的正尺度平移。因此四角在 `[1,W-2]×[1,H-2]` 内，足以约束整个输出网格。双线性插值只使用相邻两行两列；OpenCV 通用 CPU 路径以 10 位定点构造坐标、5 位插值表量化，其舍入扰动远小于这里保留的一像素余量，取样支持不会跨出图像。由此推断，将未被使用的常数边界值改为零不会改变裁剪像素。[OpenCV 插值常量](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/include/opencv2/imgproc.hpp)、[OpenCV 坐标构造与 bilinear 实现](https://raw.githubusercontent.com/opencv/opencv/4.x/modules/imgproc/src/imgwarp.cpp)。

这是依据 API 语义和通用实现作出的源码判定。上游 4.x 分支于本审查日期读取，未独立证明它与目标服务器的 OpenCV 二进制、HAL/优化后端完全对应；实际输出仍由脚本的三臂精确相等断言验收，不能把上述推导记为 native pixel PASS。

`interior=False` 时，两路仍分别计算同一全图均值，沿用 `INTER_LINEAR`、`BORDER_CONSTANT` 和相同通道顺序。该分支也覆盖守卫不能证明安全的贴边裁剪，并不只代表已越界。`read_pair` 已断言 RGB/TIR 等尺寸，两个连续数组来自同一六通道图像，所以以 `images[0]` 的高宽判断两路成立。源码没有将某一路均值复用于另一路，也没有改变越界填充规则。

**实验流程核对**

| 检查项 | 判定与依据 |
|---|---|
| 三臂差异 | original 使用 frozen `data.py` 的六通道图像视图；contiguous 使用预先复制的两张连续图像；interior 在相同连续输入上增加 guard 并有条件跳过两次均值。guard 成本会计入候选调用。 |
| 几何、目标与归一化 | 标准库 AST 对照确认三函数前 5 条几何/扰动语句完全一致；`crop` 拼接、`normalized_box`、`tensor` 赋值及前三项返回表达式也一致。逆矩阵、角点和判定只生成新对象，不修改 affine、box 或 RNG。实际新输出是否相等尚未运行验证。 |
| RNG | 69–77 行每次调用都新建 `RandomState(2027 + identity)`；128 模板传 `None`，256 搜索均先抽一次 normal 尺度再抽两维 uniform 中心扰动。三臂及三次重复的 RNG 起点相同，guard 不消耗随机数。这是固定配对样本，不声称复现训练加载器此前已推进的 RNG 历史。 |
| 闭包 | 每个 case 的三函数在其循环变量改变前立即调用，暖机、断言、全部计时均在当前循环内完成；records 不保存闭包。不存在延迟执行后统一引用最后一帧或最后一个 seed 的问题。 |
| 样本数 | 57–66 行选 dev 前 10 序列，每序列首个及中间有效帧，每帧 1 个 `(2,128)` 模板和 3 个 `(4,256)` 搜索，共计划 80 cases，即模板 20、搜索 60。既有数据审计记录 97 条 dev，清单生成器要求每序列至少 3 个正宽高有效帧。本地没有 `data/manifest.json`，未重新核验远端实际清单或解码。 |
| 一致性门槛 | 78–82 行每个 case 先各调用一次作为暖机，并对两候选逐项要求 `torch.equal` 和两次 `np.array_equal`，之后才计时；失败会中止正常报告写入。80 cases 计划产生 160 组三项比较。新实验尚未通过这些检查。 |
| 顺序平衡 | 三次 repeat 的 `(identity + repeat) % 3` 覆盖 0、1、2；每个 case 中每臂各处于第一、第二、第三位一次。因此计划每臂计时 240 次，共 720 次；另有 240 次未计时的暖机/检查调用。标准库枚举三种 identity 余数确认位置计数均为 1。 |
| 计时边界 | `perf_counter` 包含 RNG 构造、函数调用、guard、所需均值、仿射裁剪、拼接、归一化与目标/几何生成；不包含三臂输出比较。解码、六通道拼接、两张连续数组准备均在计时外，报告字段如实披露。 |
| CPU 范围 | 54 行要求 `CUDA_VISIBLE_DEVICES == ''`；脚本没有模型、Tracker、权重加载或 CUDA API。输入为 NumPy，`torch.from_numpy` 与 `torch.equal` 操作 CPU tensor。导入 contiguous helper 不会触发其 `main`，包初始化仅为文档字符串。 |
| 输出范围 | 只读取 dev 图像/框并写新报告；56 行要求输出尚不存在。保存每 case 的 sequence/frame/size/trial/seed、guard 结果及全部三次计时，统计均值/P95；不计算 PR/NPR/SR，不生成轨迹。 |

三臂固定暖机顺序仍可能带来短时缓存影响；每 case 三次计时及混合尺寸/图像的均值/P95只是组件描述，不能证明统计显著性或全系统延迟。连续数组准备被排除，尚未实现生产解码直接保留两张数组；不能把 crop 调用降幅直接称为在线提速或精度保持。实际均值保留/跳过数应按新 report 核验。

**语法与活动源码隔离**

在本机现有 uv 管理的 Python 3.13.12 上，仅用标准库 `ast.parse(..., feature_version=(3,7))` 检查新 helper、contiguous helper 和 `dual_rgbt/data.py`，三者通过。该检查没有导入 NumPy/OpenCV/Torch 或项目模块，不等同 Python 3.7 环境的运行验证。首次 PATH Python 返回 `No pyvenv.cfg file`，未执行检查；后续明确使用已存在的 uv interpreter 完成上述 AST 检查，没有安装或修改环境。

逐项重算 `reports/full_pipeline_source_manifest.json` 中 11 个本地文件的 SHA256，11/11 与保存值相等：`train_policy.py`、`check_policy_checkpoint.py`、`train_trajectory_policy.py`、`run_task.py`、`run_tracking.py`、`score_tracking.py`、`complete_pipeline.py`、`model.py`、`data.py`、`tracker.py`、`losses.py`。源码搜索未发现生产入口引用新 helper。

隔离边界应准确写为：**不修改 11 项源码，不调用训练/跟踪/评价入口；复用 frozen `data.py` 的 `read_pair/crop_pair` 作为 CPU 数据入口和基线。** `data.py` 本身在 11 项清单内，因此不能写成“实验完全不调用这 11 项源码”。本审查只读取这些函数，没有调用；没有查询远端活动进程或重新证明远端源码状态。

没有需要修改源码的阻塞项，也没有要求添加 fallback、额外异常处理或兼容分支。后续成功运行至多支持其实际完成的 dev 裁剪输出一致性及隔离 CPU 调用计时；新 interior native pixel/speed、完整系统速度、轨迹等价及精度结论均未由本审查取得。
