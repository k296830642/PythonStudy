# GPU 固定人脸视频剪辑说明

此目录是 GPU 版本的本地副本，对应服务器 `/home/ymqai/test/videoCut`。本地文件不会自动覆盖服务器，待现有任务结束并确认后再部署。

## 功能与规则

- 从 `faces` 读取同一目标孩子的样本人脸，生成平均特征。
- 扫描 `input` 中的 MP4、MOV、MKV、AVI，按目标出现片段分别导出 MP4。
- 导出保留源视频分辨率、帧率、音频；视频使用 NVIDIA NVENC `h264_nvenc` 编码。
- 目标孩子不打码；所有其他检测到的人脸覆盖 `dama/cat_kawaii_blink.png`，自动缩放并跟随位置移动。
- 导出阶段逐帧检测所有脸，不为速度降低打码检测频率。
- 目标脸直接识别成功后，若相似度短暂不足，按位置、运动方向、脸框重叠度与尺寸进行保守轨迹关联，防止目标孩子被误打码。不能严格关联的脸仍打码。
- `buffalo_l` 的检测、识别、年龄、性别、关键点模型均保留。
- 输出 InsightFace 检测日志：检测脸数、目标直接识别数、目标轨迹关联数、打码数和相似度。

## 处理流程

1. 加载 InsightFace `buffalo_l` 与目标样本人脸特征。
2. 顺序读取视频帧，仅在 `sample_fps` 时间点扫描。顺序读取避免频繁跳转关键帧造成 CPU 解码变慢。
3. 合并目标出现段，按前后补时生成剪辑区间。
4. 导出时逐帧检测：目标脸不覆盖，其他脸全部叠加猫咪打码图。
5. 在输出目录写入剪辑视频和 `gpu_face_clip_report.json`。

## 目录与运行

```text
videoCut/
├── faces/                         # 目标人脸样本
├── input/                         # 输入视频
├── output/                        # 本地默认输出目录
├── dama/cat_kawaii_blink.png      # 非目标脸打码图
├── .models/                       # InsightFace 模型缓存
├── config.json                    # 运行配置
└── gpu_face_clip.py               # 主程序
```

服务器建议将 `output_dir` 设为 `/ssd-data/output`，不要把大视频输出到根盘。

```bash
cd /home/ymqai/test/videoCut
.venv/bin/python gpu_face_clip.py

# 只扫描，不导出
.venv/bin/python gpu_face_clip.py --scan-only

# 只处理一个视频
.venv/bin/python gpu_face_clip.py --video /home/ymqai/test/videoCut/input/example.mp4
```

## 日志说明

```text
[export] t=12.00s faces=3 target_recognized=1 target_tracked=0 masks=2 similarities=[0.841, 0.214, 0.168]
```

- `faces`：检测到的人脸数。
- `target_recognized`：本帧由相似度阈值直接确认的目标脸数。
- `target_tracked`：相似度短暂不足、但由连续轨迹确认而不打码的目标脸数。
- `masks`：需覆盖猫咪图的人脸数。
- `similarities`：各脸与样本人脸的余弦相似度。

GPU 正常初始化必须包含：

```text
InsightFace active providers: ['CPUExecutionProvider', 'CUDAExecutionProvider']
```

只有 CPU Provider 时，`require_gpu=true` 会让程序报错退出，避免静默回退 CPU。

## config.json 全部参数

### 输入、输出与模型

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `reference_faces_dir` | 目标人脸样本目录。 | 放同一孩子的清晰、多角度图片，不要混入其他人。 |
| `input_dir` | 输入视频目录。 | 服务器建议放 SSD。 |
| `output_dir` | 视频和报告输出目录。 | 服务器使用 `/ssd-data/output`。 |
| `model_cache_dir` | InsightFace 模型缓存。 | 解压后的模型目录必须保留；已解压的 zip 可删除。 |
| `model_pack` | InsightFace 模型包。 | 当前为精度优先的 `buffalo_l`。 |
| `onnx_providers` | ONNX 推理 Provider 顺序。 | 保持 CUDA 在 CPU 前。 |
| `require_gpu` | 是否强制 GPU。 | 服务器保持 `true`。 |
| `cuda_library_dirs` | CUDA 动态库目录。 | 当前服务器需包含 cublas、cudnn、cuda_runtime、cuda_nvrtc、curand、cufft 的 `lib` 目录。 |

### 识别与扫描

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `recognition_mode` | 使用哪组相似度阈值。 | 默认 `balanced`；实际效果取决于下方数值。 |
| `mode_thresholds.strict` | strict 模式阈值。 | 调低更易命中，也更可能误识别。 |
| `mode_thresholds.balanced` | balanced 模式阈值。 | 推荐默认值；调低提高召回，调高降低误识别。 |
| `mode_thresholds.sensitive` | sensitive 模式阈值。 | 以数值为准：调低更易认定目标，调高更严格。 |
| `detection_size` | 人脸检测输入边长。 | 调大提升远处小脸检测，但更慢、更占显存；调小更快。 |
| `detection_threshold` | 人脸检测置信度阈值。 | 调低检出更多弱脸但误检可能增加；调高更保守。 |
| `sample_fps` | 扫描阶段每秒识别帧数。 | 调大不易漏短暂出现但更慢；调小更快。导出打码不受影响。 |
| `minimum_face_size` | 忽略过小人脸的最小边长。 | 调低可检出远处小脸但更慢；调高更快。 |

### 剪辑片段

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `before_seconds` | 目标出现前保留秒数。 | 调大保留更多上下文，文件更长。 |
| `after_seconds` | 目标最后出现后保留秒数。 | 调大保留更多上下文，文件更长。 |
| `clip_overlap_strategy` | 相邻补时片段重叠策略。 | `trim` 中点裁开、不重复；`allow` 保留重叠、可能重复。 |
| `minimum_appearance_seconds` | 有效出现的最短时长。 | 调低保留短暂出现；调高过滤偶发误识别。 |
| `maximum_gap_seconds` | 同一出现段允许的未命中间隔。 | 调大可跨过遮挡；过大可能合并两次独立出现。 |

### 打码

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `mask_other_faces` | 是否打码所有非目标脸。 | 隐私需求下保持 `true`。 |
| `mask_image_path` | 猫咪打码 PNG 路径。 | 建议使用带透明通道的正方形 PNG。 |
| `mask_scale` | 打码图相对脸部最大边的倍数。 | 调大覆盖更安全；调小更贴合。 |
| `mask_minimum_size` | 打码图最小像素尺寸。 | 调大更能遮小脸，但遮挡更多。 |
| `mask_alpha` | 打码图不透明度，范围 `(0, 1]`。 | 隐私要求下保持 `1.0`。 |

### 目标轨迹与日志

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `target_tracking_enabled` | 启用目标轨迹关联。 | 保持 `true`，防止目标脸瞬时识别波动后被打码。 |
| `target_tracking_max_gap_frames` | 直接识别未命中后最多关联帧数。 | 调大可应对模糊/快速运动；过大增加误去码风险。 |
| `target_tracking_minimum_iou` | 候选脸与预测目标框的最小重叠比例。 | 调大更严格；调小更适应移动。 |
| `target_tracking_maximum_center_distance_ratio` | 候选中心与预测中心的最大距离，按目标脸对角线比例计。 | 调大适应快速镜头移动；调小更严格。 |
| `target_tracking_maximum_size_ratio` | 候选脸与预测目标脸允许的最大尺寸变化倍数。 | 调大适应推拉镜头；调小降低误关联。 |
| `insightface_detection_log` | 是否打印检测日志。 | 调试设 `true`；稳定批处理可设 `false`。 |
| `detection_log_interval_seconds` | 日志时间间隔。 | 调小更详细但日志更多；调大减少输出。 |
| `detection_log_include_similarity` | 是否打印每张脸相似度。 | 调阈值时设 `true`，稳定运行可设 `false`。 |

### 编码与写入

| 参数 | 功能 | 调整建议 |
| --- | --- | --- |
| `video_codec` | FFmpeg 视频编码器。 | 服务器使用 `h264_nvenc`。 |
| `video_preset` | NVENC 预设。 | `p1` 最快、质量较低；`p7` 质量更高、较慢；`p5` 为平衡值。 |
| `video_cq` | NVENC 恒定质量参数。 | 数值调小，画质更高、文件更大；调大则相反。 |
| `audio_codec` | 输出音频编码器。 | 当前 `aac`，兼容性较好。 |
| `overwrite_output` | 已有同名输出是否覆盖。 | 常规保持 `false`；需要重跑同一任务时改 `true`。 |

## CUDA 与 ONNX Runtime

服务器驱动支持 CUDA 13.2，但项目实际使用 CUDA 12 的 Python 动态库，因此固定 `onnxruntime-gpu==1.26.0` 与 CUDA 12/cuDNN 9 配套。

启动时脚本会按绝对路径预加载 CUDA Runtime、NVRTC、cuBLAS、cuBLASLt、CURAND、CUFFT、cuDNN，解决 Linux 在 Python 进程启动后修改 `LD_LIBRARY_PATH` 仍无法加载 CUDA Provider 的问题。

不要把 `.so.12` 伪造软链接成 `.so.13`，这不能解决 ABI 兼容性问题。

## 后续优化项

1. 用真实样本回归检查快速镜头移动、多人重叠、转头、远距离小脸和短暂遮挡；根据检测日志校准阈值与轨迹参数。
2. 使用 FFmpeg/NVDEC 硬件解码，进一步降低当前 OpenCV 解码的 CPU 占用。
3. 扫描阶段采用批量 GPU 推理，提高 RTX 4090 利用率；需要验证显存、顺序与模型 API。
4. 增加独立日志文件和“轨迹去码帧”清单，便于人工复核多人场景。
5. Windows EXE 优先封装本地 CPU 版；GPU 版更适合服务器，因为 CUDA、ONNX Runtime 和模型文件较大。

## 当前验证范围

- 已验证 GPU 服务器上 ONNX Runtime 与 InsightFace 全部模型实际启用 `CUDAExecutionProvider`。
- 已验证 `h264_nvenc` 编码器存在。
- 本地 GPU 副本只做语法检查，未自动启动本地视频处理。目标轨迹逻辑需要真实视频人工复核后再部署到服务器。
