# 192.168.4.137 GPU 服务端改造与模型选型方案

## 目标

本地电脑或未来 Windows EXE 作为操作端；`192.168.4.137` 的 RTX 4090 服务器负责视频识别、剪辑和结果保存。所有任务文件放到 `/ssd-data`，不要写入空间紧张的系统根分区。

```text
Windows EXE / 本地桌面程序
        |
        | 内网 HTTP(S) + Token
        v
FastAPI 服务
        |
        +-- 创建任务、上传文件、查询状态、下载结果
        +-- 任务队列
        +-- RTX 4090 GPU Worker
        +-- FFmpeg NVDEC/NVENC Worker
        v
/ssd-data/jobs/<任务ID>/input work output report.json worker.log
```

## 当前本地模型为什么是 YuNet + SFace

| 模型 | 作用 | 选择原因 |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | 检测画面中人脸的位置和关键点。 | 轻量、OpenCV 可直接调用、CPU 可运行。 |
| `face_recognition_sface_2021dec.onnx` | 把人脸转为特征向量，并和目标孩子样本比较。 | 与 YuNet 配合完成固定人脸识别，部署简单。 |

人脸检测回答“脸在哪里”；人脸识别回答“是不是目标孩子”。两者都需要。当前组合适合本机先验证业务规则，但在远脸、强侧脸、小脸、多人密集或暗光场景中，上限不如更大的专用模型。

## 为什么当前没有使用 YOLO

YOLO 是目标检测框架，标准 YOLO 并不输出人脸身份特征。即使采用 YOLO-Face，它也主要解决“脸在哪里”；要判断是不是目标孩子，仍然需要 SFace、ArcFace 或其他人脸识别模型。

因此，YOLO 不是人脸识别的替代品，而是“人脸检测器”的可选替代方案。对于固定人脸检索，直接使用 YuNet/SFace 的本地依赖更少、调试链路更短。YOLO-Face 适合已经统一采用 YOLO 平台，或确实需要更强检测、跟踪和复杂场景能力的项目。

## 服务端模型路线

| 路线 | 检测 | 识别 | 优点 | 代价 | 建议 |
| --- | --- | --- | --- | --- | --- |
| A：保持本地一致 | YuNet | SFace | 最易和本地结果对齐。 | GPU 加速需额外验证 OpenCV CUDA 或 ONNX GPU。 | 服务端第一阶段。 |
| B：推荐 GPU 路线 | SCRFD 或 RetinaFace | ArcFace | 对小脸、侧脸、多人画面通常更有余量。 | 必须重新校准阈值，并审查模型许可。 | 正式服务端优先路线。 |
| C：YOLO-Face 路线 | YOLO-Face | ArcFace | 可复用 YOLO 检测与跟踪生态。 | 仍需第二个识别模型；模型来源、许可和阈值更复杂。 | 仅在已决定采用 YOLO 平台时使用。 |

推荐两阶段迁移：先在服务器复现 YuNet + SFace，建立本地输出基线；再部署 SCRFD/RetinaFace + ArcFace，使用同一批视频人工比较命中片段、漏检、误识别和处理时间。不同识别模型的相似度分布不同，不能直接把本地 SFace 的 `0.40` 阈值用于 ArcFace。

## 服务端实现调整

### 本次服务器报错分析

日志中的关键错误是：`libcublasLt.so.13: cannot open shared object file`。这说明 NVIDIA 驱动和 RTX 4090 可见，但 ONNX Runtime CUDA Provider 的动态依赖没有找到。随后 InsightFace 自动显示 `Applied providers: ['CPUExecutionProvider']`，所以当时实际使用的是 CPU，不是 GPU。

`nvidia-smi` 显示的 `CUDA Version: 13.2` 表示驱动支持的 CUDA 兼容上限，不等于系统已经安装了 `libcublasLt.so.13`。同时，`nvcc` 不存在也不一定是问题：运行预编译推理服务通常不需要完整 CUDA 编译工具链，但必须有与 ONNX Runtime GPU 包匹配的 CUDA/cuBLAS/cuDNN runtime 动态库。

修复顺序：

1. 先执行 `ldconfig -p | grep -E 'libcublasLt|libcudnn'`，确认库是否存在。
2. 检查 `/ssd-data/ai/venv` 是否带有 NVIDIA Python runtime，并把真实的 `nvidia/cublas/lib`、`nvidia/cudnn/lib` 目录加入 `server_gpu/config.json` 的 `cuda_library_dirs`。
3. 临时 `export LD_LIBRARY_PATH=...` 验证 `onnxruntime.get_available_providers()` 是否出现 `CUDAExecutionProvider`。
4. 如果库不存在，再按当前 `onnxruntime-gpu` 版本的官方兼容矩阵安装匹配 runtime；不要仅凭 `nvidia-smi` 的 13.2 字样安装任意 CUDA Toolkit。
5. `require_gpu=true` 时，脚本会在启动阶段拒绝 CPU 回退；必须看到 `InsightFace active providers` 中包含 `CUDAExecutionProvider` 才继续处理。

1. 使用 `onnxruntime-gpu`，优先 `CUDAExecutionProvider`，保留 CPU Provider 作为回退。
2. 初期每张 GPU 同时只处理一个任务，避免显存、磁盘和 NVENC 编码器竞争。
3. 输入端验证 NVDEC 硬件解码；输出端测试 `h264_nvenc`，并比较与 `libx264` 的画质、文件大小和兼容性。
4. 扫描阶段可低频检测；候选区间再用高频或逐帧精查，提高起止时间精度。
5. 红框模式仍需逐帧识别，应作为人工审核模式，而不是最快的批量模式。
6. 可加入 ByteTrack 或 BoT-SORT 跟踪器，用于连接相邻帧同一张脸、平滑短暂漏检；跟踪器不能替代人脸身份识别。

建议的服务器独立配置：

```json
{
  "execution_mode": "server_gpu",
  "gpu_device_id": 0,
  "onnx_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
  "detector_backend": "scrfd_10g",
  "recognizer_backend": "arcface_r100",
  "gpu_batch_size": 16,
  "scan_fps": 2.0,
  "candidate_refine_fps": 10.0,
  "video_decoder": "nvdec",
  "video_encoder": "h264_nvenc",
  "job_root": "/ssd-data/jobs",
  "max_concurrent_gpu_jobs": 1,
  "result_retention_days": 7
}
```

数值调节：`gpu_batch_size` 调大可提高吞吐但更占显存，建议从 8、16、32 逐档测试；`candidate_refine_fps` 调大能让片段边界更精确但增加 GPU 时间，通常 8-15；`max_concurrent_gpu_jobs` 初期保持 1；`result_retention_days` 调大保留结果更久，但更占 `/ssd-data`。

## 接口、存储和安全

提供创建任务、上传参考人脸、上传视频或访问共享目录、查询进度、下载 ZIP、读取报告、取消任务等接口。每个任务使用独立目录。

即使仅在内网，也应使用 Token 或账号鉴权。服务器密码、SSH 密码、人脸图片不得写入代码、配置、EXE、接口返回值或日志。完成任务可保留 7 天后清理 `work` 和 `output`，运行中任务不得清理。

## 上线验收

在同一批视频、同一张人脸样本与相同业务规则下，对比本地和服务器：

| 指标 | 验收标准 |
| --- | --- |
| 命中片段 | 数量一致，或差异能够人工解释。 |
| 误识别和漏检 | 不高于本地基线。 |
| 输出规格 | 保持原分辨率、帧率和音频。 |
| 起止时间 | 与人工检查一致，且不重复。 |
| 处理时间 | 端到端显著快于本地 CPU。 |
| 磁盘使用 | 工作文件均位于 `/ssd-data`，根分区不显著增长。 |

先确认模型识别正确，再提高并发或追求更高速度。对本项目而言，识别准确性优先于吞吐量。
