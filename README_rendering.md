# BlenderProc 2 工业场景渲染说明

从 **models_simple** 中的 PLY 零件渲染复杂工业场景，目标为在 2D 图像中识别 **obj_000001.ply**。

## 场景设计（增强版）

- **工业背景**：地面 + 箱子、管道等杂物；使用 [Haven](https://polyhaven.com/) 纹理（金属、混凝土、铁皮）和 HDRI 环境光。
- **密集堆叠**：零件在小范围 XY（±0.15 m）、较高处（Z 0.35–0.7 m）随机落下，物理仿真 5–15 秒，形成**密集堆叠、互相遮挡、杂乱排布**。
- **干扰因素**：前背景融合、轮廓不清晰（景深）、高度相似零件（obj_000004/000005 等）、动态光照（HDRI）。
- **目标模型**：`obj_000001.ply` 必现；保留 PLY 顶点颜色；多帧相机、instance_to_model.json。

## 1. 下载 Haven（可选，用于更真实背景）

```bash
blenderproc download haven resources/haven
```

会下载 textures 和 hdris。不下载时脚本使用简单灰色背景。

## 2. 用法

```bash
# 使用 Haven（纹理 + HDRI）
blenderproc run render_industrial_scene.py models-simple --haven-path resources/haven -o output

# 不指定 Haven
blenderproc run render_industrial_scene.py models-simple -o output

# 指定零件数、帧数、分辨率
blenderproc run render_industrial_scene.py models-simple --haven-path resources/haven -o out1 -n 12 --num-frames 12 --resolution 800 600

# 无头运行（SSH / Autodl）
xvfb-run -a blenderproc run render_industrial_scene.py models-simple --haven-path resources/haven -o output --seed 42
```

## 3. 参数

| 参数 | 说明 |
|------|------|
| model_dir | PLY 目录（如 `models-simple`），需包含 `obj_000001.ply` |
| --haven-path | Haven 资源目录（textures + hdris）；不传则简单背景 |
| --output / -o | 输出目录 |
| --num-objects / -n | 零件数量（含目标与干扰） |
| --num-frames | 渲染帧数（相机位姿数） |
| --allow-duplicates / --no-duplicates | 是否允许同模型多次出现 |
| --physics / --no-physics | 物理落体堆叠 / 表面采样 |
| --depth-of-field / --no-depth-of-field | 景深模糊（轮廓不清晰） |
| --scale-mm, --resolution, --seed | 单位、分辨率、随机种子 |

## 4. 输出

- **HDF5**：`output/*.hdf5`（colors, depth, segmap）
- **instance_to_model.json**：instance_id → model_id 映射
- **PNG**：`output/rgb/*.png`（若 imageio 可用）
