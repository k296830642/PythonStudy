# 固定人脸视频剪辑工具

本工具使用 `faces` 目录中的目标人脸图片，识别 `input` 目录视频中该人物出现的时间，并把每次出现分别导出为 MP4 文件。

默认规则：人物出现前 2 秒、后 2 秒；相邻片段如有时间重叠，会自动在重叠区中点裁开，避免不同输出视频重复播放同一段画面。

## 目录说明

| 目录或文件 | 用途 |
| --- | --- |
| `faces` | 目标人物的人脸样本图片。建议使用清晰、正面、无遮挡的图片。 |
| `input` | 需要处理的视频。 |
| `output_2s_no_overlap` | 当前 2 秒且无重复规则的输出目录。 |
| `output` | 以前 5 秒上下文规则生成的历史结果，保留不修改。 |
| `config.json` | 识别、分段和导出配置。 |

## 安装环境

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

首次执行会将 YuNet 人脸检测模型和 SFace 人脸特征模型下载到 `.models` 目录。

## 使用方法

处理全部视频：

```powershell
.\.venv\Scripts\python.exe .\face_clip.py
```

只识别时间段、不导出视频：

```powershell
.\.venv\Scripts\python.exe .\face_clip.py --scan-only
```

只处理一个视频：

```powershell
.\.venv\Scripts\python.exe .\face_clip.py --video .\input\example.mp4
```

运行单元测试：

```powershell
.\.venv\Scripts\python.exe -m unittest -v .\test_face_clip.py
```

### 误识别优化后的推荐命令

```powershell
python face_clip.py --config config.json --scan-only
python face_clip.py --config config.json
```

`config.json` 当前使用 strict 模式、较高检测阈值（0.80）、最小人脸 56 像素、最短出现 1 秒和 1.5 秒最大断档，优先压低误识别。若漏检明显，可先把 `recognition_mode` 改为 `balanced`，或将 strict 阈值从 `0.52` 小幅降到 `0.48`，每次只改一个参数。

参考脸现在按“每张样本分别计算相似度并取最大值”匹配，适合目录中包含正脸、侧脸、不同光线的多张图片。建议保留 10-20 张清晰样本，并删除严重模糊、遮挡、多人同框或检测不到脸的图片。

### buffalo_sc / buffalo_l CPU 对比

基准脚本不会修改现有 YuNet/SFace 流程，需单独安装 InsightFace CPU 依赖：

```powershell
python -m pip install insightface onnxruntime psutil
python benchmark_insightface.py --config config.json --video input\example.mp4 --seconds 30 --sample-fps 2
```

脚本会依次输出两个模型的平均单帧耗时、P95 耗时、吞吐 FPS、检测脸数和进程 RSS 内存。首次运行会自动下载模型包；CPU 环境务必确认输出中的 provider 为 `CPUExecutionProvider`。通常 `buffalo_sc` 更快、更省内存，`buffalo_l` 精度和鲁棒性更强但占用更高，最终应以同一批视频的误识别/漏识别和实测耗时决定。

### InsightFace buffalo_sc 剪辑版

`face_clip_buffalo_sc.py` 是保留原有剪辑、红框、打码、轨迹保护和 JSON 报告功能的 InsightFace CPU 版本；原来的 `face_clip.py` 保持 YuNet + SFace，不会被替换。

```powershell
python -m pip install -r requirements.txt
python face_clip_buffalo_sc.py --config config_buffalo_sc.json --scan-only
python face_clip_buffalo_sc.py --config config_buffalo_sc.json
python face_clip_buffalo_sc.py --config config_buffalo_sc.json --video input\example.mp4
```

该版本固定使用 `buffalo_sc` 和 `CPUExecutionProvider`。`config_buffalo_sc.json` 的 `insightface_detector_size` 默认是 `[640, 640]`，优先保证检测效果；若长视频处理速度不足，可试 `[480, 480]`，但要重新抽查远处脸和侧脸的漏检情况。识别阈值需以 `--scan-only` 的真实结果校准；当前 strict 值 `0.52` 是保守起点，误识别仍存在时可按 `0.02` 逐步提高。

### 新手微调学习路线

1. 先不要训练模型：整理 20-50 个短视频片段，标记“目标/非目标”，记录每个误识别和漏检案例。
2. 先做阈值校准：用 `--scan-only` 报告统计相似度，分别估计目标脸与他人脸的分布，再决定 strict/balanced 阈值。
3. 若仍不够，再学习 InsightFace 推理、ONNX 模型输入输出和人脸对齐；先替换预训练模型，不改网络结构。
4. 最后再做微调：准备不同光线、角度、距离的目标脸，以及数量相近的非目标脸；按训练/验证集 8:2 划分，保存验证集上的 FAR（误识别率）和 FRR（漏检率）。
5. 实践顺序建议：Python/NumPy -> OpenCV 人脸检测 -> 人脸 embedding 与余弦相似度 -> PyTorch 基础 -> 分类/度量学习 -> ONNX 导出与部署。每次只改变一个变量，并保留可回滚的配置和测试视频集。

## 关键配置

- `recognition_mode`：`strict` 减少误识别，`balanced` 为默认平衡模式，`sensitive` 更容易检出侧脸或远脸，但误识别风险更高。
- `sample_fps`：每秒检测的帧数。数值越高，短暂出现越不容易漏掉，但处理时间更长。
- `before_seconds` 与 `after_seconds`：每段视频前后保留的上下文秒数，当前均为 `2.0`。
- `clip_overlap_strategy`：`trim` 自动裁掉相邻片段重叠部分；`allow` 允许不同输出视频有重复画面。
- `annotate_matched_face`：设为 `true` 时，命中的目标人脸会用红框标出；默认 `false`，使用更快的 FFmpeg 导出路径。
- `mask_other_faces`：设为 `true` 时，除目标孩子外的所有检测到的人脸都会使用 `mask_image_path` 打码。启用后导出会逐帧检测，速度会降低，但可避免漏打码。
- `mask_image_path`、`mask_scale`、`mask_minimum_size`、`mask_alpha`：分别设置打码图片、相对人脸大小、最小尺寸和透明度。隐私场景建议 `mask_alpha=1.0`。
- `target_tracking_enabled`：目标脸轨迹保护开关。目标脸在连续少量帧中相似度波动时，按位置、运动方向、重叠度和尺寸变化继续判定为目标，避免目标脸被误打码。
- `target_tracking_max_gap_frames`、`target_tracking_minimum_iou`、`target_tracking_maximum_center_distance_ratio`、`target_tracking_maximum_size_ratio`：控制轨迹保护的容错范围。调大可适应快速移动、转头和短暂模糊，但误把其他脸当目标的风险也会增加；建议从默认值开始，逐项小幅调整。
- `insightface_detection_log`、`detection_log_interval_seconds`、`detection_log_include_similarity`：控制逐帧检测日志。日志会显示检测脸数、直接识别目标数、轨迹目标数、打码数及相似度。
- `annotation_show_similarity`：设为 `true` 时，在红框上方显示本帧人脸相似度。
- `annotation_box_color_bgr`：标注颜色，OpenCV 使用 BGR 顺序；`[0, 0, 255]` 是红色。
- `annotation_box_thickness`、`annotation_label_font_scale`、`annotation_label_thickness`：分别控制脸框粗细、相似度文字大小和文字粗细。
- `maximum_gap_seconds`：两次命中之间允许间隔的最大秒数。当前为 3 秒，用于容忍短暂转头、遮挡或检测丢失。
- `overwrite_output`：为 `false` 时，已存在的同名结果不会重复生成，便于中断后继续处理。

当两次出现的间隔小于“前置秒数 + 后置秒数”时，无法同时满足每段完整上下文且完全不重复。`trim` 模式优先保证没有重复画面，因此会适当缩短相邻片段靠近交界处的一侧上下文。

开启 `annotate_matched_face` 后，导出阶段会逐帧重新识别人脸并绘制红框，因此比普通导出明显更慢；只在需要人工核对识别效果时建议开启。

启用 `mask_other_faces` 后也会进入逐帧检测导出路径，即使关闭红框仍会执行打码。目标脸保护优先于打码：只有直接识别为目标，或通过已确认轨迹严格关联的脸不打码；其余检测到的人脸全部打码。

## 输出与报告

每个 MP4 文件名包含源视频名、片段序号和起止时间。处理报告位于输出目录的 `face_clip_report.json`，其中记录命中时间、相似度、出现区间和导出文件路径。

源视频不会被修改。当前输出使用 H.264 视频和 AAC 音频，保留原始分辨率和帧率。

## 模型许可

后续发布 EXE 前，需要保留并审查 YuNet、SFace、OpenCV 与 FFmpeg 的相关许可文件。



## 合并的视频运行脚本
python merge_videos_by_folder.py "C:\Users\steven\IdeaProjects\pythonStudy\videoEdit\server_gpu\output" --output "C:\Users\steven\IdeaProjects\pythonStudy\videoEdit\hebing\merged.mp4" --include-root-videos
## YuNet + SFace CPU 版本
python face_clip.py --config config.json --scan-only

## InsightFace buffalo_sc CPU 版本
..\ .venv\Scripts\python.exe face_clip_buffalo_sc.py --config config_buffalo_sc.json --scan-only
..\ .venv\Scripts\python.exe face_clip_buffalo_sc.py --config config_buffalo_sc.json

python face_clip_buffalo_sc.py --config config_buffalo_sc.json --scan-only
python face_clip_buffalo_sc.py --config config_buffalo_sc.json --video input/your_video.mp4
python face_clip_buffalo_sc.py --config config_buffalo_sc.json

## buffalo_sc buffalo_l对比
python benchmark_insightface.py --config config.json --seconds 30 --sample-fps 2
## 结果
{
"model": "buffalo_sc",
"video": "C:\\Users\\steven\\IdeaProjects\\pythonStudy\\videoEdit\\input\\0aa6a65e6aa03b24feb40141028ebe60_raw.mp4",
"samples": 60,
"faces_detected": 146,
"mean_ms": 79.85,
"p95_ms": 171.24,
"throughput_fps": 12.52,
"rss_before_mb": 101.9,
"rss_after_load_mb": 133.2,
"rss_after_run_mb": 196.4
},
{
"model": "buffalo_l",
"video": "C:\\Users\\steven\\IdeaProjects\\pythonStudy\\videoEdit\\input\\0aa6a65e6aa03b24feb40141028ebe60_raw.mp4",
"samples": 60,
"faces_detected": 165,
"mean_ms": 2276.95,
"p95_ms": 4458.0,
"throughput_fps": 0.44,
"rss_before_mb": 126.5,
"rss_after_load_mb": 368.8,
"rss_after_run_mb": 562.5
}
