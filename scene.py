import blenderproc as bproc

import argparse
import json
import math
import random
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


TARGET_PLY = "obj_000001.ply"
SCALE_MM_TO_M = 0.001
GROUND_TEXTURE_RELATIVE_PATH = Path("picture") / "complex.png"

CATEGORY_BACKGROUND = 0
CATEGORY_TARGET = 1
CATEGORY_DISTRACTOR = 2
# 与 obj_000001.ply 零件风格一致的深蓝，用于背景地面主色（高对比干扰）
OBJ1_GROUND_BLUE = (0.03, 0.05, 0.22)
DEFAULT_MODEL_DIR_CANDIDATES = ("models_simple", "models-simple")
SIMILARITY_BOOST = {
    "obj_000004.ply": 2.2,
    "obj_000005.ply": 2.2,
    "obj_000010.ply": 1.6,
    "obj_000011.ply": 1.3,
}


# ─────────────────────────────────────────────────────────────────────────────
#  可配置参数 Cfg — 所有调优旋钮集中在此，便于后续批量调整
#  [KEY CHANGE] 新增 Cfg 类，取代原先散落在各函数里的硬编码魔法数字
# ─────────────────────────────────────────────────────────────────────────────
class Cfg:
    # ── 场景物体数量（1 目标 + 4~5 干扰物 = 每帧 5~6 个 .ply）────────────────
    N_OBJ_MIN      = 5        # 每帧含目标的最少物体数（1 目标 + 4 干扰）
    N_OBJ_MAX      = 6        # 每帧含目标的最多物体数（1 目标 + 5 干扰）

    # ── 相机 — 近距离俯视堆叠体，距离与焦距动态调节 ─────────────────────────
    CAM_DIST_MIN   = 0.30    # m，相机到堆叠体中心距离下限
    CAM_DIST_MAX   = 0.80    # m，相机到堆叠体中心距离上限
    CAM_ELEV_DEG_MIN = 20.0  # 与铅垂线夹角（度），约为人站立低头视角下限
    CAM_ELEV_DEG_MAX = 55.0  # 与铅垂线夹角（度），避免倒置
    CAM_LENS_MIN   = 18.0    # mm，动态调节焦距时的下限
    CAM_LENS_MAX   = 85.0    # mm，动态调节焦距时的上限
    CAM_ITER       = 30      # 相机位姿采样迭代次数（降低以解除 CPU 瓶颈）
    LENS_ITER      = 12      # 单次位姿下焦距二分/迭代次数

    # ── 整体占空比目标（目标+干扰物+线缆等整堆物体 bbox 占画面比例）──────────
    SCENE_OCC_LO   = 0.65    # 整体占空比下限（严格 65%）
    SCENE_OCC_HI   = 0.75    # 整体占空比上限（严格 75%）

    # ── Domain Randomization（合成-真实域差距缩减）──────────────────────────
    DR_SCALE_LO    = 0.90    # [DR] 物体缩放随机下限（模拟制造公差/不同型号）
    DR_SCALE_HI    = 1.10    # [DR] 物体缩放随机上限
    DR_ROUGH_LO    = 0.12    # [DR] 材质粗糙度随机下限
    DR_ROUGH_HI    = 0.88    # [DR] 材质粗糙度随机上限
    DR_METAL_LO    = 0.00    # [DR] 金属度随机下限
    DR_METAL_HI    = 0.95    # [DR] 金属度随机上限
    DR_LIGHT_JIT   = 0.35    # [DR] 灯光能量随机抖动 ±35%（叠加到基础能量范围）
    DR_CAM_NOISE   = 0.008   # [DR] 相机位置高斯噪声（m），模拟手持/夹具抖动
    DR_MBLUR_PROB  = 0.0     # 彻底关闭运动模糊
    DR_MBLUR_LO    = 0.04    # [DR] 运动模糊快门比例下限（帧，0=关闭）
    DR_MBLUR_HI    = 0.12    # [DR] 运动模糊快门比例上限

    # ── 物理仿真（低采样率减轻单核引擎负担，对视觉堆叠影响极小）────────────
    PHYS_T_MIN     = 3.0     # s，最短仿真时间
    PHYS_T_MAX     = 6.0     # s，最长仿真时间
    PHYS_CHK       = 0.8     # s，收敛检测间隔
    PHYS_STEPS_PER_SEC = 60  # 默认 60，降低以解除物理引擎性能瓶颈
    PHYS_SOLVER_ITERS  = 10  # 默认 10

    # ── 紧密堆叠控制（XY 范围缩小以让掉落后堆叠更紧密）──────────────────────
    STACK_FOOTPRINT_SCALE = 0.14  # 相对目标 footprint 的投放半范围，越小越紧
    STACK_XY_SIGMA_SCALE  = 0.18  # XY 高斯采样 sigma = footprint * 此值，越小越密
    STACK_LAYER_STEP      = 0.026 # m，相邻投放层高度步长
    STACK_DROP_JITTER_MIN = 0.008
    STACK_DROP_JITTER_MAX = 0.026

    # ── 工作台 / 接触面高度 ──────────────────────────────────────────────────
    TABLE_Z_TOP      = 0.86   # m，工作台顶面高度
    # ── 防穿透 / 稳定性修正 ──────────────────────────────────────────────────
    FLOOR_Z_TOP      = TABLE_Z_TOP
    GROUND_CLEARANCE = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="BlenderProc 2 工业杂乱目标识别场景渲染")
    parser.add_argument("model_dir", type=str, nargs="?", default=None,
                        help="PLY 模型目录，默认自动尝试 models_simple 或 models-simple")
    parser.add_argument("--haven-path", type=str, default="resources/haven",
                        help="Haven 资源目录（可选，用于 HDRI 环境光）")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="输出目录")
    parser.add_argument("--num-layouts", type=int, default=5,
                        help="物理堆叠场景数量（外层循环），默认 5")
    parser.add_argument("--views-per-layout", type=int, default=12,
                        help="每个场景的拍摄视角数（内层循环），默认 12；总帧数 = num_layouts × views_per_layout")
    parser.add_argument("--min-distractors", type=int, default=Cfg.N_OBJ_MIN - 1,
                        help="每帧最少干扰物数量（不含目标，默认 Cfg.N_OBJ_MIN-1）")
    parser.add_argument("--max-distractors", type=int, default=Cfg.N_OBJ_MAX - 1,
                        help="每帧最多干扰物数量（不含目标，默认 Cfg.N_OBJ_MAX-1）")
    parser.add_argument("--depth-of-field", action="store_true", default=False,
                        help="启用轻微景深")
    parser.add_argument("--physics", action="store_true", default=True,
                        help="启用物理模拟")
    parser.add_argument("--no-physics", action="store_false", dest="physics",
                        help="关闭物理模拟")
    parser.add_argument("--render-samples", type=int, default=40,
                        help="Cycles 采样数")
    parser.add_argument("--scale-mm", type=float, default=SCALE_MM_TO_M,
                        help="模型单位换算，默认 mm -> m")
    parser.add_argument("--resolution", type=int, nargs=2, default=[960, 720],
                        metavar=("W", "H"), help="图像分辨率")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    return parser.parse_args()


def resolve_model_dir(model_dir_arg: str | None) -> Path:
    if model_dir_arg is not None:
        model_dir = Path(model_dir_arg).resolve()
        if not model_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        return model_dir

    root = Path.cwd()
    for candidate in DEFAULT_MODEL_DIR_CANDIDATES:
        path = (root / candidate).resolve()
        if path.exists():
            return path
    raise FileNotFoundError("未找到模型目录，请显式传入 models_simple 或 models-simple 路径")


def collect_ply_files(model_dir: Path) -> list[Path]:
    ply_files = sorted(model_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"在 {model_dir} 中未找到 PLY 文件")
    target_path = model_dir / TARGET_PLY
    if not target_path.exists():
        raise FileNotFoundError(f"目标模型 {TARGET_PLY} 不存在于 {model_dir}")
    return ply_files


def load_models_info(model_dir: Path) -> dict:
    info_path = model_dir / "models_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"缺少模型尺寸信息: {info_path}")
    with open(info_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    mapped = {}
    for idx_str, item in raw.items():
        mapped[f"obj_{int(idx_str):06d}.ply"] = {
            "size": np.array([item["size_x"], item["size_y"], item["size_z"]], dtype=np.float32),
            "diameter": float(item["diameter"]),
        }
    return mapped


def resolve_ground_texture_path() -> Path:
    candidates = [
        Path.cwd() / GROUND_TEXTURE_RELATIVE_PATH,
        Path(__file__).resolve().parent / GROUND_TEXTURE_RELATIVE_PATH,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"未找到地面贴图: {GROUND_TEXTURE_RELATIVE_PATH}")


def build_similarity_weights(model_names: list[str], model_stats: dict) -> dict[str, float]:
    target_size = model_stats[TARGET_PLY]["size"]
    target_diameter = model_stats[TARGET_PLY]["diameter"]
    target_ratio = target_size / np.maximum(target_size[2], 1e-6)

    weights = {}
    for model_name in model_names:
        if model_name == TARGET_PLY:
            continue
        stats = model_stats[model_name]
        size = stats["size"]
        size_ratio = size / np.maximum(size[2], 1e-6)
        size_diff = np.linalg.norm((size - target_size) / np.maximum(target_size, 1e-6))
        ratio_diff = np.linalg.norm(size_ratio - target_ratio)
        diameter_diff = abs(stats["diameter"] - target_diameter) / max(target_diameter, 1e-6)
        similarity = math.exp(-(0.9 * size_diff + 0.7 * ratio_diff + 0.8 * diameter_diff))
        similarity *= SIMILARITY_BOOST.get(model_name, 1.0)
        weights[model_name] = max(0.05, similarity)
    return weights


def weighted_sample_without_replacement(items: list[str], weights: dict[str, float], k: int) -> list[str]:
    chosen = []
    pool = list(items)
    local_weights = {item: float(weights.get(item, 1.0)) for item in pool}
    while pool and len(chosen) < k:
        total = sum(local_weights[item] for item in pool)
        threshold = random.uniform(0.0, total)
        accum = 0.0
        picked = pool[-1]
        for item in pool:
            accum += local_weights[item]
            if accum >= threshold:
                picked = item
                break
        chosen.append(picked)
        pool.remove(picked)
    return chosen


def get_blender_obj(mesh_obj):
    return mesh_obj.blender_obj


def configure_rigidbody_world():
    """提高刚体求解稳定性，减少点接触和穿透。"""
    scene = bpy.context.scene
    rw = scene.rigidbody_world
    if rw is None:
        bpy.ops.rigidbody.world_add()
        rw = scene.rigidbody_world
    if rw is None:
        return
    if hasattr(rw, "steps_per_second"):
        rw.steps_per_second = Cfg.PHYS_STEPS_PER_SEC
    if hasattr(rw, "solver_iterations"):
        rw.solver_iterations = Cfg.PHYS_SOLVER_ITERS


def configure_gpu_rendering():
    """强制 Cycles 使用 GPU（OptiX/CUDA）渲染，大幅加速。"""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences
    if "cycles" in prefs.addons:
        cprefs = prefs.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA"):
            try:
                cprefs.compute_device_type = device_type
                if hasattr(cprefs, "refresh_devices"):
                    cprefs.refresh_devices()
                for d in getattr(cprefs, "devices", []):
                    if getattr(d, "type", None) == device_type:
                        d.use = True
                break
            except Exception:
                continue
    print("[Cycles] 渲染设备: GPU" if scene.cycles.device == "GPU" else "[Cycles] 使用默认设备")




# ─────────────────────────────────────────────────────────────────────────────
#  [KEY CHANGE] 工人视角相机矩阵构建器：强制 roll = 0°
# ─────────────────────────────────────────────────────────────────────────────
def build_worker_cam2world(cam_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """
    [KEY CHANGE] 构建工人俯视视角相机矩阵，roll 强制为 0°（画面地平线水平）。

    原理：以世界 Z 轴（天顶方向）为"上"的参考，显式构建正交坐标系：
      fwd   = normalize(target - camera)              # 相机朝向目标
      right = normalize(fwd × world_up)               # 画面水平方向（X）
      up    = normalize(right × fwd)                  # 画面垂直方向（Y），对齐 Z

    Blender 相机约定：本地 -Z = 观察方向，+Y = 画面上方。
    与 bproc.camera.rotation_from_forward_vec(up_axis="Z", inplane_rot=0.0) 等效，
    但本实现完全不引入任何 inplane_rot 项，确保画面绝不倒置/侧翻。
    """
    fwd = np.asarray(target_pos, dtype=float) - np.asarray(cam_pos, dtype=float)
    fwd /= (np.linalg.norm(fwd) + 1e-12)

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, world_up)
    r_norm = np.linalg.norm(right)
    if r_norm < 1e-6:
        # 相机几乎正对正下方（pitch ≈ 90°），改用 Y 轴作备用上参考
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
    right /= r_norm

    up = np.cross(right, fwd)
    up /= (np.linalg.norm(up) + 1e-12)

    # cam2world 列排列：[right | up | -fwd | pos]
    # 第三列 = -fwd，因为 Blender 相机沿本地 -Z 轴观察
    m = np.eye(4, dtype=float)
    m[:3, 0] = right
    m[:3, 1] = up
    m[:3, 2] = -fwd   # Blender 相机本地 +Z 指向相机背面
    m[:3, 3] = np.asarray(cam_pos, dtype=float)
    return m


def reset_rigidbody_state(mesh_obj):
    blender_obj = get_blender_obj(mesh_obj)
    if blender_obj.rigid_body is not None:
        if hasattr(blender_obj.rigid_body, "linear_velocity"):
            blender_obj.rigid_body.linear_velocity = (0.0, 0.0, 0.0)
        if hasattr(blender_obj.rigid_body, "angular_velocity"):
            blender_obj.rigid_body.angular_velocity = (0.0, 0.0, 0.0)


def clamp_objects_above_ground(mesh_objs: list):
    """
    对仿真后的物体做接触面穿透修正：若 min_z 低于工作台顶面则整体抬升。
    """
    for obj in mesh_objs:
        if obj is None:
            continue
        try:
            bo = get_blender_obj(obj)
            min_z = min((bo.matrix_world @ Vector(c)).z for c in bo.bound_box)
            if min_z < Cfg.FLOOR_Z_TOP + Cfg.GROUND_CLEARANCE:
                loc = bo.location.copy()
                loc.z += (Cfg.FLOOR_Z_TOP + Cfg.GROUND_CLEARANCE - min_z)
                bo.location = loc
        except Exception:
            pass


def sync_target_state_from_pose(target_obj, target_state: dict):
    """物理后同步目标位置，避免相机仍使用仿真前位置。"""
    bo = get_blender_obj(target_obj)
    target_state["location"] = np.array([bo.location.x, bo.location.y, bo.location.z], dtype=np.float32)


def setup_mesh_obj(mesh_obj, scale: float, category_id: int, model_id: str):
    mesh_obj.set_scale([scale, scale, scale])
    mesh_obj.move_origin_to_bottom_mean_point()
    mesh_obj.set_shading_mode("FLAT")
    mesh_obj.set_cp("category_id", category_id)
    mesh_obj.set_cp("model_id", model_id)


def load_scene_objects(model_dir: Path, scale: float, model_stats: dict) -> tuple:
    ply_paths = collect_ply_files(model_dir)
    cached = {}

    target_obj = None
    distractor_objs = {}
    for ply_path in ply_paths:
        loaded = bproc.loader.load_obj(str(ply_path), cached_objects=cached)
        if not loaded:
            continue
        mesh_obj = loaded[0]
        model_name = ply_path.name
        category = CATEGORY_TARGET if model_name == TARGET_PLY else CATEGORY_DISTRACTOR
        setup_mesh_obj(mesh_obj, scale, category, model_name)
        mesh_obj.set_name(f"mesh_{model_name.replace('.ply', '')}")

        # 初次加载时覆盖原始材质（PLY 顶点色太单调），赋予 PBR 材质
        randomize_part_material(mesh_obj)

        if model_name == TARGET_PLY:
            target_obj = mesh_obj
        else:
            distractor_objs[model_name] = mesh_obj

    if target_obj is None:
        raise RuntimeError("目标模型加载失败")

    # 坚守凸包碰撞 (CONVEX_HULL)，Active 刚体绝不用 MESH，保证生成效率
    for obj in [target_obj] + list(distractor_objs.values()):
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        blender_obj = get_blender_obj(obj)
        if blender_obj.rigid_body is not None:
            rb = blender_obj.rigid_body
            rb.mass = 0.45 + 0.65 * random.random()
            rb.friction = 1.2
            rb.restitution = 0.0
            rb.linear_damping = 0.8
            rb.angular_damping = 0.8
            if hasattr(rb, "use_margin"):
                rb.use_margin = True
            if hasattr(rb, "collision_margin"):
                rb.collision_margin = 0.001

    return target_obj, distractor_objs, [p.name for p in ply_paths if p.name != TARGET_PLY]


def assign_material_to_mesh(mesh_obj, material):
    blender_obj = get_blender_obj(mesh_obj)
    blender_obj.data.materials.clear()
    blender_obj.data.materials.append(material)


def make_color_rgba(rgb):
    return (rgb[0], rgb[1], rgb[2], 1.0)


def set_node_input_if_exists(node, input_name: str, value):
    socket = node.inputs.get(input_name)
    if socket is not None:
        socket.default_value = value


def random_industrial_palette():
    """
    工业地面调色板：地面深蓝与 obj1 零件完全一致（OBJ1_GROUND_BLUE），
    深灰与污渍做轻微抖动；整体可再叠加更多噪声层实现高对比干扰。
    """
    # 地面主蓝固定为与 obj1 一致的蓝色，不随机
    blue = OBJ1_GROUND_BLUE
    grays = [
        (0.11, 0.11, 0.13), (0.13, 0.14, 0.15), (0.09, 0.09, 0.11), (0.10, 0.12, 0.16),
    ]
    dirt = (0.02, 0.02, 0.04)
    gray = random.choice(grays)
    jittered_gray = tuple(max(0.0, min(1.0, c + random.uniform(-0.02, 0.02))) for c in gray)
    jittered_dirt = tuple(max(0.0, min(1.0, c + random.uniform(-0.01, 0.01))) for c in dirt)
    return [blue, jittered_gray, jittered_dirt]


def rebuild_floor_material(material, palette, scale_xy: tuple, texture_path: Path):
    """
    使用外部贴图构建地面材质，并通过高倍平铺让大尺寸地面保持
    “无限延展”的观感；叠加轻微噪声来减弱重复拼贴痕迹。
    """
    material.use_nodes = True
    nt = material.node_tree
    nt.nodes.clear()
    nt.links.clear()
    N = nt.nodes
    L = nt.links

    out = N.new("ShaderNodeOutputMaterial"); out.location = (1000, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled"); bsdf.location = (760, 0)
    set_node_input_if_exists(bsdf, "Metallic", random.uniform(0.00, 0.03))
    set_node_input_if_exists(bsdf, "Roughness", random.uniform(0.82, 0.95))
    set_node_input_if_exists(bsdf, "Specular IOR Level", 0.02)

    tc = N.new("ShaderNodeTexCoord"); tc.location = (-1100, 0)
    mp = N.new("ShaderNodeMapping"); mp.location = (-880, 0)
    mp.inputs["Scale"].default_value[0] = scale_xy[0]
    mp.inputs["Scale"].default_value[1] = scale_xy[1]
    mp.inputs["Rotation"].default_value[2] = random.uniform(0.0, math.pi * 2.0)
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    image = bpy.data.images.load(str(texture_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"

    img = N.new("ShaderNodeTexImage"); img.location = (-640, 140)
    img.image = image
    img.extension = "REPEAT"
    img.interpolation = "Smart"
    L.new(mp.outputs["Vector"], img.inputs["Vector"])

    hue_sat = N.new("ShaderNodeHueSaturation"); hue_sat.location = (-380, 140)
    hue_sat.inputs["Hue"].default_value = 0.5 + random.uniform(-0.015, 0.015)
    hue_sat.inputs["Saturation"].default_value = random.uniform(0.94, 1.08)
    hue_sat.inputs["Value"].default_value = random.uniform(0.92, 1.02)
    L.new(img.outputs["Color"], hue_sat.inputs["Color"])

    noise = N.new("ShaderNodeTexNoise"); noise.location = (-640, -180)
    noise.inputs["Scale"].default_value = random.uniform(6.0, 14.0)
    noise.inputs["Detail"].default_value = random.uniform(4.0, 8.0)
    noise.inputs["Roughness"].default_value = 0.55
    L.new(mp.outputs["Vector"], noise.inputs["Vector"])

    dirt_ramp = N.new("ShaderNodeValToRGB"); dirt_ramp.location = (-380, -180)
    ramp = dirt_ramp.color_ramp
    while len(ramp.elements) > 2:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position = 0.28
    ramp.elements[0].color = (0.82, 0.82, 0.82, 1.0)
    ramp.elements[1].position = 0.88
    ramp.elements[1].color = (0.96, 0.96, 0.96, 1.0)
    L.new(noise.outputs["Fac"], dirt_ramp.inputs["Fac"])

    mix_dirt = N.new("ShaderNodeMixRGB"); mix_dirt.location = (-120, 60)
    mix_dirt.blend_type = "MULTIPLY"
    mix_dirt.use_clamp = True
    mix_dirt.inputs["Fac"].default_value = random.uniform(0.08, 0.16)
    L.new(hue_sat.outputs["Color"], mix_dirt.inputs["Color1"])
    L.new(dirt_ramp.outputs["Color"], mix_dirt.inputs["Color2"])

    bump = N.new("ShaderNodeBump"); bump.location = (520, -180)
    bump.inputs["Strength"].default_value = random.uniform(0.03, 0.07)
    bump.inputs["Distance"].default_value = 0.003
    L.new(noise.outputs["Fac"], bump.inputs["Height"])

    L.new(mix_dirt.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(dirt_ramp.outputs["Color"], bsdf.inputs["Roughness"])
    L.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])


def rebuild_wall_material(material, scale_xy: tuple):
    """简洁深灰工业墙壁材质。"""
    material.use_nodes = True
    nt = material.node_tree
    nt.nodes.clear()
    nt.links.clear()
    N = nt.nodes
    L = nt.links

    out  = N.new("ShaderNodeOutputMaterial");  out.location  = (700, 0)
    bsdf = N.new("ShaderNodeBsdfPrincipled");  bsdf.location = (460, 0)
    set_node_input_if_exists(bsdf, "Roughness", random.uniform(0.65, 0.85))
    set_node_input_if_exists(bsdf, "Metallic",  0.0)

    tc = N.new("ShaderNodeTexCoord");  tc.location = (-700, 0)
    mp = N.new("ShaderNodeMapping");   mp.location = (-500, 0)
    mp.inputs["Scale"].default_value[0] = scale_xy[0]
    mp.inputs["Scale"].default_value[1] = scale_xy[1]
    L.new(tc.outputs["Object"], mp.inputs["Vector"])

    noise = N.new("ShaderNodeTexNoise");  noise.location = (-260, 0)
    noise.inputs["Scale"].default_value    = random.uniform(8.0, 22.0)
    noise.inputs["Detail"].default_value   = 6.0
    noise.inputs["Roughness"].default_value = 0.6
    L.new(mp.outputs["Vector"], noise.inputs["Vector"])

    cr = N.new("ShaderNodeValToRGB");  cr.location = (-40, 0)
    c  = cr.color_ramp
    while len(c.elements) > 2:
        c.elements.remove(c.elements[-1])
    g0 = random.uniform(0.06, 0.14)
    g1 = random.uniform(0.18, 0.30)
    c.elements[0].color = (g0, g0, g0, 1.0)
    c.elements[1].color = (g1, g1, g1, 1.0)
    L.new(noise.outputs["Fac"], cr.inputs["Fac"])
    L.new(cr.outputs["Color"], bsdf.inputs["Base Color"])
    L.new(bsdf.outputs["BSDF"], out.inputs["Surface"])


def rebuild_industrial_material(material, palette, scale_xy: tuple, metallic_bias: float = 0.0):
    """向后兼容入口，转发到 rebuild_floor_material。"""
    rebuild_floor_material(
        material,
        palette,
        scale_xy,
        texture_path=resolve_ground_texture_path(),
    )


def create_background_objects():
    """创建无限大平层地面（30m×30m），无墙壁，仅返回 floor。"""
    floor = bproc.object.create_primitive("CUBE", size=1.0)
    floor.set_scale([30.0, 30.0, 0.04])
    floor.set_location([0.0, 0.0, Cfg.TABLE_Z_TOP - 0.04])
    floor.set_rotation_euler([0.0, 0.0, 0.0])
    floor.set_name("industrial_workbench_top")
    floor.set_cp("category_id", CATEGORY_BACKGROUND)
    floor.enable_rigidbody(active=False, collision_shape="BOX")
    return floor


def refresh_background_materials(floor, palette):
    """每帧重新创建地面材质，仅 floor。"""
    blender_obj = get_blender_obj(floor)
    old_mats = list(blender_obj.data.materials)
    blender_obj.data.materials.clear()
    for m in old_mats:
        if m is not None:
            try:
                bpy.data.materials.remove(m, do_unlink=True)
            except Exception:
                pass

    floor_mat = bpy.data.materials.new(name="FloorMat")
    assign_material_to_mesh(floor, floor_mat)
    scale_floor = (random.uniform(10.0, 18.0), random.uniform(10.0, 18.0))
    ground_texture_path = resolve_ground_texture_path()
    rebuild_floor_material(floor_mat, palette, scale_xy=scale_floor, texture_path=ground_texture_path)


def create_lights():
    return {
        "top": bproc.types.Light(light_type="AREA"),
        "fill": bproc.types.Light(light_type="AREA"),
        "rim": bproc.types.Light(light_type="AREA"),
    }


def configure_background_and_lights(haven_path: Path, lights: dict, focus_point: np.ndarray):
    # 纯深灰世界背景（无 HDRI 风景），强度足够提供环境补光
    bproc.renderer.set_world_background([0.12, 0.12, 0.12], strength=0.45)

    fp = np.asarray(focus_point, dtype=float)
    # 顶光：质心正上方 0.8m，明亮且带柔和阴影
    lights["top"].set_location([float(fp[0]), float(fp[1]), float(fp[2]) + 0.8])
    lights["top"].set_energy(random.uniform(60.0, 80.0))
    lights["top"].set_radius(random.uniform(0.4, 0.6))
    lights["top"].set_color([1.0, 0.98, 0.95])

    # 辅光 fill：侧前方超大柔光板，专为金属提供渐变反射
    lights["fill"].set_location([float(fp[0]) + 0.6, float(fp[1]) + 0.6, float(fp[2]) + 0.3])
    lights["fill"].set_energy(random.uniform(20.0, 40.0))
    lights["fill"].set_radius(random.uniform(1.5, 2.5))
    lights["fill"].set_color([1.0, 0.98, 0.95])

    # 轮廓光 rim：侧后方超大柔光板，金属边缘高光
    lights["rim"].set_location([float(fp[0]) - 0.6, float(fp[1]) - 0.6, float(fp[2]) + 0.3])
    lights["rim"].set_energy(random.uniform(20.0, 40.0))
    lights["rim"].set_radius(random.uniform(1.5, 2.5))
    lights["rim"].set_color([1.0, 0.98, 0.95])


def hide_mesh_obj(mesh_obj):
    mesh_obj.set_location([0.0, 0.0, -8.0])
    blender_obj = get_blender_obj(mesh_obj)
    blender_obj.hide_render = True
    blender_obj.hide_viewport = True
    if blender_obj.rigid_body is not None:
        blender_obj.rigid_body.enabled = False


def show_mesh_obj(mesh_obj):
    blender_obj = get_blender_obj(mesh_obj)
    blender_obj.hide_render = False
    blender_obj.hide_viewport = False
    if blender_obj.rigid_body is not None:
        blender_obj.rigid_body.enabled = True


def place_target(target_obj, model_stats: dict, scale: float) -> dict:
    """
    将目标物设为 PASSIVE（固定锚点），平放在工作台中心，作为干扰物堆叠的稳定底座。
    PASSIVE 物体不参与物理运动，确保整个 simulation 周期内不漂移。
    """
    dr_scale = scale * random.uniform(Cfg.DR_SCALE_LO, Cfg.DR_SCALE_HI)
    target_obj.set_scale([dr_scale, dr_scale, dr_scale])
    size_m = model_stats[TARGET_PLY]["size"] * dr_scale

    # origin 已被 move_origin_to_bottom_mean_point 移到底面
    # 因此 location.z = TABLE_Z_TOP 即底面接触工作台台面
    target_loc = np.array([
        random.uniform(-0.015, 0.015),
        random.uniform(-0.015, 0.015),
        Cfg.TABLE_Z_TOP,
    ], dtype=np.float32)
    target_rot = [
        math.radians(random.uniform(-5.0, 5.0)),   # 几乎水平，小幅倾斜
        math.radians(random.uniform(-5.0, 5.0)),
        math.radians(random.uniform(0.0, 360.0)),  # yaw 随机
    ]
    target_obj.set_location(target_loc.tolist())
    target_obj.set_rotation_euler(target_rot)
    show_mesh_obj(target_obj)

    blender_obj = get_blender_obj(target_obj)
    if blender_obj.rigid_body is not None:
        blender_obj.rigid_body.type = "PASSIVE"     # 固定，作为堆叠底座
        blender_obj.rigid_body.collision_shape = "CONVEX_HULL"
        blender_obj.rigid_body.friction = 1.5
        blender_obj.rigid_body.restitution = 0.0
        if hasattr(blender_obj.rigid_body, "collision_margin"):
            blender_obj.rigid_body.collision_margin = 0.001
    reset_rigidbody_state(target_obj)
    return {"size_m": size_m, "location": target_loc, "dr_scale": dr_scale}


def arrange_distractors(active_names: list[str], distractor_objs: dict,
                        model_stats: dict, scale: float, target_size_m: np.ndarray):
    """
    分层堆叠策略：所有干扰物从目标正上方更小范围内落下，
    物理仿真后在工作台上形成更紧密的自然堆叠。

    - 动态掉落高度：按上一物体的真实 Z 尺寸累加 (layer_base_z += current_z_size + 0.02)，
      确保第 0 帧在 Z 轴上无重叠，避免初始穿模与爆炸。
    - 干扰物严格使用 CONVEX_HULL，绝不对 Active 刚体用 MESH。
    """
    hx = max(float(target_size_m[0]) * Cfg.STACK_FOOTPRINT_SCALE, 0.015)
    hy = max(float(target_size_m[1]) * Cfg.STACK_FOOTPRINT_SCALE, 0.015)
    target_h = float(target_size_m[2])

    for model_name, obj in distractor_objs.items():
        hide_mesh_obj(obj)
        blender_obj = get_blender_obj(obj)
        if blender_obj.rigid_body is not None:
            blender_obj.rigid_body.type = "ACTIVE"
            blender_obj.rigid_body.collision_shape = "CONVEX_HULL"
            blender_obj.rigid_body.friction = 1.4
            blender_obj.rigid_body.restitution = 0.0
            if hasattr(blender_obj.rigid_body, "collision_margin"):
                blender_obj.rigid_body.collision_margin = 0.001

    random.shuffle(active_names)
    # 动态高度：从目标顶面起，每个物体底面 = 上一物体顶面 + 0.02，保证 Z 向无重叠
    layer_base_z = target_h

    for idx, model_name in enumerate(active_names):
        obj = distractor_objs[model_name]
        show_mesh_obj(obj)

        dr_scale = scale * random.uniform(Cfg.DR_SCALE_LO, Cfg.DR_SCALE_HI)
        obj.set_scale([dr_scale, dr_scale, dr_scale])
        randomize_part_material(obj)

        dims_m = model_stats[model_name]["size"] * dr_scale
        current_object_z_size = float(dims_m[2])

        # 当前物体底面高度 = 工作台 + 已累加高度 + 小幅 Z 抖动
        z_jitter = random.uniform(Cfg.STACK_DROP_JITTER_MIN, Cfg.STACK_DROP_JITTER_MAX)
        drop_z = Cfg.TABLE_Z_TOP + layer_base_z + z_jitter
        # 下一物体起始高度 = 当前物体顶面 + 0.02
        layer_base_z += current_object_z_size + 0.02

        # XY：以目标中心为核做高斯采样，sigma 缩小使掉落后堆叠更紧密
        sigma_x = hx * getattr(Cfg, "STACK_XY_SIGMA_SCALE", 0.18)
        sigma_y = hy * getattr(Cfg, "STACK_XY_SIGMA_SCALE", 0.18)
        px = float(np.clip(random.gauss(0.0, sigma_x), -hx, hx))
        py = float(np.clip(random.gauss(0.0, sigma_y), -hy, hy))

        loc = np.array([px, py, drop_z], dtype=np.float64)
        rot = np.array([
            math.radians(random.uniform(-7.0, 7.0)),
            math.radians(random.uniform(-7.0, 7.0)),
            math.radians(random.uniform(0.0, 360.0)),
        ])
        obj.set_location(loc.tolist())
        obj.set_rotation_euler(rot.tolist())
        reset_rigidbody_state(obj)


def create_funnel_air_walls(target_state: dict):
    """
    在目标物体四周紧贴生成 4 面有厚度的墙（CUBE），构成方形漏斗，
    避免零厚度平面导致物理死循环；BOX 碰撞体计算最快。
    返回对象名称列表，仿真结束后按名称删除。
    """
    cx = float(target_state["location"][0])
    cy = float(target_state["location"][1])
    sx, sy = float(target_state["size_m"][0]), float(target_state["size_m"][1])
    half_width = max(sx, sy) * 0.5 + 0.05
    wall_span = 2.0 * half_width + 0.02
    wall_height = 0.5
    wall_thickness = 0.05
    z_center = Cfg.TABLE_Z_TOP + wall_height * 0.5

    names = ["funnel_air_wall_left", "funnel_air_wall_right", "funnel_air_wall_front", "funnel_air_wall_back"]
    walls = []
    # 左墙：CUBE，厚度在 X 方向，scale 第三维为厚度
    left = bproc.object.create_primitive("CUBE", size=1.0)
    left.set_scale([wall_height, wall_span, wall_thickness])
    left.set_location([cx - half_width, cy, z_center])
    left.set_rotation_euler([0.0, math.pi / 2.0, 0.0])
    walls.append(left)
    # 右墙
    right = bproc.object.create_primitive("CUBE", size=1.0)
    right.set_scale([wall_height, wall_span, wall_thickness])
    right.set_location([cx + half_width, cy, z_center])
    right.set_rotation_euler([0.0, -math.pi / 2.0, 0.0])
    walls.append(right)
    # 前墙：厚度在 Y 方向
    front = bproc.object.create_primitive("CUBE", size=1.0)
    front.set_scale([wall_span, wall_height, wall_thickness])
    front.set_location([cx, cy + half_width, z_center])
    front.set_rotation_euler([math.pi / 2.0, 0.0, 0.0])
    walls.append(front)
    # 后墙
    back = bproc.object.create_primitive("CUBE", size=1.0)
    back.set_scale([wall_span, wall_height, wall_thickness])
    back.set_location([cx, cy - half_width, z_center])
    back.set_rotation_euler([-math.pi / 2.0, 0.0, 0.0])
    walls.append(back)

    # BOX 碰撞体计算速度最快，并设置唯一名称供后续按名删除
    for wall, name in zip(walls, names):
        wall.enable_rigidbody(active=False, collision_shape="BOX")
        bo = get_blender_obj(wall)
        bo.hide_render = True
        bo.hide_viewport = True
        bo.name = name

    return names


def delete_generated_objects(objects: list):
    """支持传入对象名称（str）或 Blender 对象；按名称查找可避免仿真后引用失效。"""
    for obj in objects:
        if obj is None:
            continue
        if isinstance(obj, str):
            blender_obj = bpy.data.objects.get(obj)
        else:
            try:
                blender_obj = obj if obj.name in bpy.data.objects else None
            except ReferenceError:
                blender_obj = None
        if blender_obj is not None:
            try:
                bpy.data.objects.remove(blender_obj, do_unlink=True)
            except (ReferenceError, RuntimeError):
                pass


def get_vertex_color_layer_name(mesh_obj) -> str | None:
    """
    返回 PLY 模型第一个顶点颜色属性的名称。
    Blender 4.x 用 color_attributes；旧版用 vertex_colors。
    如果模型不含顶点色，返回 None。
    """
    blender_obj = get_blender_obj(mesh_obj)
    mesh = blender_obj.data
    if hasattr(mesh, "color_attributes") and len(mesh.color_attributes) > 0:
        return mesh.color_attributes[0].name
    if hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
        return mesh.vertex_colors[0].name
    return None


def create_part_material(obj_name: str, vcol_layer: str | None = None) -> bpy.types.Material:
    """
    为工业零件生成材质：微表面凹凸(Bump)、底色扰动，提升表面真实感。
    参数签名不变。（已移除 Bevel 节点以降低 GPU 渲染负担。）
    """
    mat = bpy.data.materials.new(name=f"PartMat_{obj_name}_{random.randint(0, 9999999)}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    nt.links.clear()

    output = nt.nodes.new("ShaderNodeOutputMaterial")
    output.location = (900, 0)

    principled = nt.nodes.new("ShaderNodeBsdfPrincipled")
    principled.location = (580, 0)

    is_metal = random.random() < 0.95
    if is_metal:
        metallic = random.uniform(0.95, 1.00)
        base_roughness = random.uniform(0.15, 0.35)
    else:
        metallic = random.uniform(0.0, 0.08)
        base_roughness = random.uniform(0.45, 0.88)

    set_node_input_if_exists(principled, "Metallic", metallic)
    set_node_input_if_exists(principled, "Specular IOR Level", random.uniform(0.25, 0.55))

    # ── 原始底色（顶点色或 Fallback）────────────────────────────────────
    if vcol_layer is not None:
        vcol_node = nt.nodes.new("ShaderNodeVertexColor")
        vcol_node.location = (-420, 160)
        vcol_node.layer_name = vcol_layer
        gamma = nt.nodes.new("ShaderNodeGamma")
        gamma.location = (-220, 160)
        gamma.inputs["Gamma"].default_value = random.uniform(0.88, 1.18)
        nt.links.new(vcol_node.outputs["Color"], gamma.inputs["Color"])
        base_color_socket = gamma.outputs["Color"]
    else:
        if is_metal:
            fallback_colors = [
                (0.52, 0.52, 0.55, 1.0), (0.40, 0.40, 0.42, 1.0), (0.62, 0.50, 0.35, 1.0),
                (0.34, 0.34, 0.36, 1.0), (0.58, 0.55, 0.52, 1.0),
            ]
        else:
            fallback_colors = [
                (0.12, 0.13, 0.15, 1.0), (0.22, 0.24, 0.28, 1.0),
                (0.30, 0.26, 0.20, 1.0), (0.18, 0.18, 0.16, 1.0),
            ]
        base_color_node = nt.nodes.new("ShaderNodeRGB")
        base_color_node.location = (-420, 160)
        base_color_node.outputs[0].default_value = (*random.choice(fallback_colors)[:3], 1.0)
        base_color_socket = base_color_node.outputs["Color"]

    # ── 底色扰动：高频噪波与原始底色混合，打破纯色 ─────────────────────
    noise_color = nt.nodes.new("ShaderNodeTexNoise")
    noise_color.location = (-420, -40)
    noise_color.inputs["Scale"].default_value = 150.0
    noise_color.inputs["Detail"].default_value = 12.0
    noise_color.inputs["Roughness"].default_value = 0.5
    cr_color = nt.nodes.new("ShaderNodeValToRGB")
    cr_color.location = (-220, -40)
    c = cr_color.color_ramp
    while len(c.elements) > 2:
        c.elements.remove(c.elements[-1])
    c.elements[0].position = 0.0
    c.elements[0].color = (0.85, 0.85, 0.88, 1.0)
    c.elements[1].position = 1.0
    c.elements[1].color = (1.0, 1.0, 1.02, 1.0)
    nt.links.new(noise_color.outputs["Fac"], cr_color.inputs["Fac"])
    mix_base = nt.nodes.new("ShaderNodeMixRGB")
    mix_base.location = (-20, 80)
    mix_base.blend_type = "MIX"
    mix_base.inputs["Fac"].default_value = 0.15
    nt.links.new(base_color_socket, mix_base.inputs["Color1"])
    nt.links.new(cr_color.outputs["Color"], mix_base.inputs["Color2"])
    nt.links.new(mix_base.outputs["Color"], principled.inputs["Base Color"])

    # ── 划痕噪声：用于 Roughness 与 Bump Height ─────────────────────────
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.location = (-420, -280)
    noise.inputs["Scale"].default_value = random.uniform(45.0, 90.0)
    noise.inputs["Detail"].default_value = random.uniform(6.0, 12.0)
    noise.inputs["Roughness"].default_value = 0.65

    noise_scale = nt.nodes.new("ShaderNodeMath")
    noise_scale.location = (-220, -280)
    noise_scale.operation = "MULTIPLY"
    noise_scale.inputs[1].default_value = random.uniform(0.08, 0.18)
    rough_add = nt.nodes.new("ShaderNodeMath")
    rough_add.location = (-40, -280)
    rough_add.operation = "ADD"
    rough_add.use_clamp = True
    rough_add.inputs[0].default_value = base_roughness
    nt.links.new(noise.outputs["Fac"], noise_scale.inputs[0])
    nt.links.new(noise_scale.outputs["Value"], rough_add.inputs[1])
    nt.links.new(rough_add.outputs["Value"], principled.inputs["Roughness"])

    # ── 表面凹凸 (Bump)：划痕噪声驱动微表面 ─────────────────────────────
    bump = nt.nodes.new("ShaderNodeBump")
    bump.location = (320, 80)
    bump.inputs["Strength"].default_value = random.uniform(0.05, 0.15)
    bump.inputs["Distance"].default_value = 0.002
    nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], principled.inputs["Normal"])

    nt.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return mat


def randomize_part_material(mesh_obj):
    """
    每帧重新应用一个新材质。
    优先使用 PLY 自带顶点色，如模型无顶点色则使用随机工业固定色。
    """
    vcol_layer = get_vertex_color_layer_name(mesh_obj)
    mat = create_part_material(mesh_obj.get_name(), vcol_layer=vcol_layer)
    assign_material_to_mesh(mesh_obj, mat)






def target_bbox_area_ratio(target_obj) -> float:
    scene = bpy.context.scene
    cam = scene.camera
    blender_obj = get_blender_obj(target_obj)
    bbox = [blender_obj.matrix_world @ Vector(corner) for corner in blender_obj.bound_box]
    projected = [world_to_camera_view(scene, cam, p) for p in bbox]

    xs = [p.x for p in projected if p.z > 0]
    ys = [p.y for p in projected if p.z > 0]
    if len(xs) < 8 or len(ys) < 8:
        return -1.0

    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))
    if max_x <= min_x or max_y <= min_y:
        return -1.0
    return (max_x - min_x) * (max_y - min_y)


def measure_scene_occupancy(scene_objs: list, cam_obj) -> float:
    """
    用包围盒投影快速估算「整堆物体」在画面中的面积占比（无需实际渲染）。
    用于动态调节焦距，使占空比严格落在 65%～75%。

    scene_objs：包含目标、干扰物以及若已加入场景的线缆等所有需计入的物体。
    - 对所有可见物体的 8 个顶点投影到相机平面，取联合包围框
    - 返回值：[0.0, 1.0]，0=全空，1=全满
    """
    scene = bpy.context.scene
    all_xs, all_ys = [], []

    for obj in scene_objs:
        if obj is None:
            continue
        try:
            blender_obj = get_blender_obj(obj)
            if blender_obj.hide_render:
                continue
            bbox_world = [blender_obj.matrix_world @ Vector(c) for c in blender_obj.bound_box]
            for p_world in bbox_world:
                p_cam = world_to_camera_view(scene, cam_obj, p_world)
                # 只统计在相机前方的点（p_cam.z > 0）
                if p_cam.z > 0:
                    all_xs.append(p_cam.x)
                    all_ys.append(p_cam.y)
        except Exception:
            pass

    if not all_xs:
        return 0.0

    # 联合包围框面积（钳制到 [0,1]）
    x_min = max(0.0, min(all_xs))
    x_max = min(1.0, max(all_xs))
    y_min = max(0.0, min(all_ys))
    y_max = min(1.0, max(all_ys))
    if x_max <= x_min or y_max <= y_min:
        return 0.0
    return (x_max - x_min) * (y_max - y_min)


def compute_scene_focus_point(scene_objs: list, target_state: dict) -> tuple[np.ndarray, float]:
    """
    在物理仿真完成后，根据最终场景几何分布计算注视点和堆叠高度。
    相机不再围绕“理想化目标中心”采样，而是围绕最终静止后的物体堆分布采样。
    """
    xs, ys, zs = [], [], []
    for obj in scene_objs:
        if obj is None:
            continue
        try:
            blender_obj = get_blender_obj(obj)
            if blender_obj.hide_render:
                continue
            for corner in blender_obj.bound_box:
                p = blender_obj.matrix_world @ Vector(corner)
                xs.append(float(p.x))
                ys.append(float(p.y))
                zs.append(float(p.z))
        except Exception:
            pass

    if not xs:
        fallback = np.array([
            float(target_state["location"][0]),
            float(target_state["location"][1]),
            float(target_state["location"][2]) + float(target_state["size_m"][2]) * 0.35,
        ], dtype=float)
        return fallback, float(target_state["size_m"][2])

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    min_z, max_z = min(zs), max(zs)
    pile_h = max(max_z - min_z, float(target_state["size_m"][2]))
    focus = np.array([
        0.5 * (min_x + max_x),
        0.5 * (min_y + max_y),
        min_z + pile_h * 0.35,
    ], dtype=float)
    return focus, float(pile_h)


def sample_worker_camera(
    target_obj,
    target_state: dict,
    scene_objs: list,
    focus_point: np.ndarray,
) -> dict:
    """
    采样工人俯视堆叠体的相机位姿。仅依赖「整体占空比」约束，不做目标单独 bbox 约束。
    focus_point 由调用方传入，避免重复计算。
    """
    scene = bpy.context.scene
    cam = scene.camera
    cam.data.sensor_width = 36.0

    focus_point = np.asarray(focus_point, dtype=float)

    best = None
    best_occ_err = 1e9

    for _ in range(Cfg.CAM_ITER):
        # 距离限制 0.3m～0.8m；俯仰为与铅垂线夹角 20°～55°（人站立低头，避免倒置）
        distance = random.uniform(Cfg.CAM_DIST_MIN, Cfg.CAM_DIST_MAX)
        azimuth = random.uniform(0.0, math.pi * 2.0)
        elev_deg = random.uniform(Cfg.CAM_ELEV_DEG_MIN, Cfg.CAM_ELEV_DEG_MAX)
        elev_rad = math.radians(elev_deg)
        # 从堆叠体中心指向相机的单位向量：相机在堆叠体上方
        dx = math.sin(elev_rad) * math.cos(azimuth)
        dy = math.sin(elev_rad) * math.sin(azimuth)
        dz = math.cos(elev_rad)
        cam_pos = focus_point + distance * np.array([dx, dy, dz], dtype=float)

        # [DR] 相机位置轻微高斯噪声
        cam_pos += np.array([
            random.gauss(0.0, Cfg.DR_CAM_NOISE),
            random.gauss(0.0, Cfg.DR_CAM_NOISE),
            random.gauss(0.0, Cfg.DR_CAM_NOISE * 0.5),
        ])
        dist_actual = np.linalg.norm(cam_pos - focus_point)
        dist_actual = max(Cfg.CAM_DIST_MIN, min(Cfg.CAM_DIST_MAX, dist_actual))
        cam_pos = focus_point + (cam_pos - focus_point) / (np.linalg.norm(cam_pos - focus_point) + 1e-12) * dist_actual

        # 对准堆叠体中心，Roll=0（build_worker_cam2world 已保证）
        cam2world = build_worker_cam2world(cam_pos, focus_point)
        cam.matrix_world = Matrix(cam2world.tolist())

        # ── 动态调节焦距：使「整堆物体」占画面比例严格落在 65%～75% ────────
        # measure_scene_occupancy 统计的是 scene_objs 中所有可见物体的 bbox 联合投影
        # （包含目标、干扰物以及若已加入场景的线缆等），无需实际渲染。
        lens_lo, lens_hi = float(Cfg.CAM_LENS_MIN), float(Cfg.CAM_LENS_MAX)
        lens = 0.5 * (lens_lo + lens_hi)
        for _l in range(Cfg.LENS_ITER):
            cam.data.lens = lens
            occ = measure_scene_occupancy(scene_objs, cam)
            if Cfg.SCENE_OCC_LO <= occ <= Cfg.SCENE_OCC_HI:
                break
            if occ < Cfg.SCENE_OCC_LO:
                lens_lo = lens
                lens = 0.5 * (lens_lo + lens_hi)
                lens = min(lens, Cfg.CAM_LENS_MAX - 0.5)
            else:
                lens_hi = lens
                lens = 0.5 * (lens_lo + lens_hi)
                lens = max(lens, Cfg.CAM_LENS_MIN + 0.5)
        occ_final = measure_scene_occupancy(scene_objs, cam)

        view_dir = focus_point - cam_pos
        horizontal = max(np.linalg.norm(view_dir[:2]), 1e-6)
        pitch_deg = math.degrees(math.atan2(-view_dir[2], horizontal))

        result = {
            "cam2world": cam2world,
            "ratio": target_bbox_area_ratio(target_obj),  # 仅用于元数据记录，不参与约束
            "scene_ratio": occ_final,
            "distance": float(np.linalg.norm(view_dir)),
            "pitch_deg": pitch_deg,
            "elevation_deg": pitch_deg,
            "azimuth_deg": math.degrees(azimuth),
            "lens_mm": float(cam.data.lens),
            "poi": focus_point.tolist(),
            "camera_height_m": float(cam_pos[2]),
        }

        if Cfg.SCENE_OCC_LO <= occ_final <= Cfg.SCENE_OCC_HI:
            occ_err = 0.0
        elif occ_final < Cfg.SCENE_OCC_LO:
            occ_err = Cfg.SCENE_OCC_LO - occ_final
        else:
            occ_err = occ_final - Cfg.SCENE_OCC_HI
        if occ_err == 0.0:
            cam.matrix_world = Matrix(cam2world.tolist())
            return result
        if occ_err < best_occ_err:
            best_occ_err = occ_err
            best = result

    if best is not None:
        cam.matrix_world = Matrix(best["cam2world"].tolist())
        cam.data.lens = best["lens_mm"]
        return best

    # 保底：固定距离与俯仰，仅做焦距调节
    distance = 0.5 * (Cfg.CAM_DIST_MIN + Cfg.CAM_DIST_MAX)
    azimuth = 0.0
    elev_rad = math.radians(0.5 * (Cfg.CAM_ELEV_DEG_MIN + Cfg.CAM_ELEV_DEG_MAX))
    cam_pos = focus_point + distance * np.array([
        math.sin(elev_rad) * math.cos(azimuth),
        math.sin(elev_rad) * math.sin(azimuth),
        math.cos(elev_rad),
    ], dtype=float)
    cam2world = build_worker_cam2world(cam_pos, focus_point)
    cam.matrix_world = Matrix(cam2world.tolist())
    lens_lo, lens_hi = float(Cfg.CAM_LENS_MIN), float(Cfg.CAM_LENS_MAX)
    for _l in range(Cfg.LENS_ITER):
        lens = 0.5 * (lens_lo + lens_hi)
        cam.data.lens = lens
        occ = measure_scene_occupancy(scene_objs, cam)
        if Cfg.SCENE_OCC_LO <= occ <= Cfg.SCENE_OCC_HI:
            break
        if occ < Cfg.SCENE_OCC_LO:
            lens_lo = lens
        else:
            lens_hi = lens
    view_dir = focus_point - cam_pos
    horizontal = max(np.linalg.norm(view_dir[:2]), 1e-6)
    pitch_deg = math.degrees(math.atan2(-view_dir[2], horizontal))
    return {
        "cam2world": cam2world,
        "ratio": target_bbox_area_ratio(target_obj),
        "scene_ratio": measure_scene_occupancy(scene_objs, cam),
        "distance": float(np.linalg.norm(view_dir)),
        "pitch_deg": pitch_deg,
        "elevation_deg": pitch_deg,
        "azimuth_deg": 0.0,
        "lens_mm": float(cam.data.lens),
        "poi": focus_point.tolist(),
        "camera_height_m": float(cam_pos[2]),
    }


def sample_camera_pose(target_obj, target_state: dict, image_size: tuple[int, int]) -> dict:
    """
    采样相机位姿，使目标物体 bbox 占图像面积约 15%~30%。

    正确理解"70% 占空比"：70% 指整个场景物体区域，目标仅是其中一部分，
    因此目标自身占 15%~30% 是合理的工人俯视工作台视角。

    参数范围：
    - 距离：0.40~0.85 m（比之前更远，能看到更多干扰物）
    - 镜头：24~50 mm（较宽，纳入更多周围零件）
    - 仰角：10~28°（人自然俯视工作台角度）
    - 目标 ratio 接受区间：0.15~0.30
    """
    scene = bpy.context.scene
    cam = scene.camera
    poi = Vector((
        float(target_state["location"][0]),
        float(target_state["location"][1]),
        float(target_state["size_m"][2] * random.uniform(0.30, 0.55)),
    ))

    best = None
    best_diff = 1e9
    desired = 0.22          # 目标占图像面积的理想值（15%~30% 中心）
    accept_lo = 0.15
    accept_hi = 0.30

    for _ in range(120):
        distance = random.uniform(0.40, 0.85)
        azimuth = random.uniform(-math.pi, math.pi)
        elevation_deg = random.uniform(10.0, 28.0)
        elevation = math.radians(elevation_deg)

        loc = np.array([
            poi.x + distance * math.cos(elevation) * math.cos(azimuth),
            poi.y + distance * math.cos(elevation) * math.sin(azimuth),
            max(0.15, poi.z + distance * math.sin(elevation)),
        ], dtype=np.float32)

        # 较宽的镜头让更多周围物体入画
        lens = random.uniform(24.0, 50.0)
        cam.data.lens = lens
        cam.data.sensor_width = 36.0

        forward = np.array([poi.x, poi.y, poi.z], dtype=np.float32) - loc
        forward /= np.linalg.norm(forward)
        rot = bproc.camera.rotation_from_forward_vec(
            forward,
            up_axis="Z",
            inplane_rot=random.uniform(-0.05, 0.05)
        )
        cam2world = bproc.math.build_transformation_mat(loc, rot)
        cam.matrix_world = Matrix(cam2world.tolist())

        ratio = target_bbox_area_ratio(target_obj)
        if ratio <= 0.0:
            continue

        diff = abs(ratio - desired)
        if accept_lo <= ratio <= accept_hi:
            best = {
                "cam2world": cam2world,
                "ratio": ratio,
                "distance": distance,
                "lens_mm": lens,
                "elevation_deg": elevation_deg,
                "poi": [poi.x, poi.y, poi.z],
            }
            break
        if diff < best_diff:
            best_diff = diff
            best = {
                "cam2world": cam2world,
                "ratio": ratio,
                "distance": distance,
                "lens_mm": lens,
                "elevation_deg": elevation_deg,
                "poi": [poi.x, poi.y, poi.z],
            }

    if best is None:
        raise RuntimeError("未能采样到有效相机位姿")

    cam.matrix_world = Matrix(best["cam2world"].tolist())
    return best


def configure_depth_of_field(camera_info: dict):
    cam = bpy.context.scene.camera.data
    cam.dof.use_dof = True
    cam.dof.focus_distance = float(camera_info["distance"])
    cam.dof.aperture_fstop = random.uniform(5.0, 8.0)


def disable_depth_of_field():
    bpy.context.scene.camera.data.dof.use_dof = False


def setup_color_management():
    """启用 Filmic 色调映射，避免高光过曝"""
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium Low Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def save_rgb_png(rgb_img, path: Path):
    import imageio.v2 as imageio
    imageio.imwrite(str(path), rgb_img)


def write_scene_metadata(output_dir: Path, metadata: list):
    with open(output_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def prepare_output_dirs(output_dir: Path) -> dict:
    rgb_dir = output_dir / "rgb"
    hdf5_dir = output_dir / "hdf5"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    return {"rgb": rgb_dir, "hdf5": hdf5_dir}


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    model_dir  = resolve_model_dir(args.model_dir)
    haven_path = Path(args.haven_path).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dirs   = prepare_output_dirs(output_dir)

    model_stats = load_models_info(model_dir)
    bproc.init()
    configure_gpu_rendering()              # 强制 Cycles 使用 GPU，加速渲染
    configure_rigidbody_world()
    setup_color_management()               # Filmic 色调，防止高光过曝
    bproc.camera.set_resolution(*args.resolution)
    bproc.renderer.set_max_amount_of_samples(args.render_samples)
    # 当前 BlenderProc 版本仅支持 "INTEL" / "OPTIX" / None，且 OptiX 在本机会偶发失败
    bproc.renderer.set_denoiser("INTEL")
    # [KEY CHANGE] 同时输出 RGB / 实例分割 / 深度图
    bproc.renderer.enable_depth_output(activate_antialiasing=True)
    bproc.renderer.enable_segmentation_output(map_by=["category_id", "instance", "name"])

    floor = create_background_objects()
    lights = create_lights()
    target_obj, distractor_objs, distractor_names = load_scene_objects(
        model_dir, args.scale_mm, model_stats
    )
    similarity_weights = build_similarity_weights(distractor_names, model_stats)

    frame_records = []
    total_frames = args.num_layouts * args.views_per_layout

    # ─────────────────────────────────────────────────────────────────────
    #  双层循环：外层 = 物理布局（一次物理多视角），内层 = 相机与渲染
    # ─────────────────────────────────────────────────────────────────────

    for layout_idx in range(args.num_layouts):
        bproc.utility.reset_keyframes()

        # ── 外层：目标放置、干扰物、漏斗、两段式物理 ───────────────────────
        randomize_part_material(target_obj)
        target_state = place_target(target_obj, model_stats, args.scale_mm)
        funnel_names = create_funnel_air_walls(target_state)

        distractor_count = random.randint(
            min(args.min_distractors, len(distractor_names)),
            min(args.max_distractors, len(distractor_names)),
        )
        chosen_distractors = weighted_sample_without_replacement(
            distractor_names, similarity_weights, distractor_count
        )
        arrange_distractors(
            chosen_distractors, distractor_objs,
            model_stats, args.scale_mm, target_state["size_m"]
        )

        if args.physics:
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=1.5,
                max_simulation_time=2.5,
                check_object_interval=Cfg.PHYS_CHK,
            )
            delete_generated_objects(funnel_names)
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=1.5,
                max_simulation_time=2.5,
                check_object_interval=Cfg.PHYS_CHK,
            )
            sync_target_state_from_pose(target_obj, target_state)
        else:
            delete_generated_objects(funnel_names)

        pile_objs = [target_obj] + [distractor_objs[n] for n in chosen_distractors]

        # ── 内层：同一堆叠场景下多视角（材质/光照/相机/单帧渲染）────────────
        for view_idx in range(args.views_per_layout):
            frame_idx = layout_idx * args.views_per_layout + view_idx

            focus_point, _ = compute_scene_focus_point(pile_objs, target_state)

            palette = random_industrial_palette()
            refresh_background_materials(floor, palette)
            for obj in pile_objs:
                randomize_part_material(obj)
            configure_background_and_lights(haven_path, lights, focus_point)

            camera_info = sample_worker_camera(target_obj, target_state, pile_objs, focus_point)

            use_mblur = random.random() < Cfg.DR_MBLUR_PROB
            if use_mblur:
                bpy.context.scene.render.use_motion_blur = True
                bpy.context.scene.render.motion_blur_shutter = random.uniform(Cfg.DR_MBLUR_LO, Cfg.DR_MBLUR_HI)
            else:
                bpy.context.scene.render.use_motion_blur = False

            # add_camera_pose 会推进 scene.frame_end；BlenderProc 要求 frame_end > frame_start 才渲染
            rendered_frame = bproc.camera.add_camera_pose(camera_info["cam2world"])
            bpy.context.scene.frame_start = rendered_frame
            bpy.context.scene.frame_end = rendered_frame + 1

            if args.depth_of_field:
                configure_depth_of_field(camera_info)
            else:
                disable_depth_of_field()

            data = bproc.renderer.render()

            rgb_path = out_dirs["rgb"] / f"{frame_idx:04d}.png"
            save_rgb_png(data["colors"][0], rgb_path)
            hdf5_frame_dir = out_dirs["hdf5"] / f"{frame_idx:04d}"
            hdf5_frame_dir.mkdir(parents=True, exist_ok=True)
            bproc.writer.write_hdf5(str(hdf5_frame_dir), data)

            frame_records.append({
                "frame_id":               frame_idx,
                "layout_id":              layout_idx,
                "view_id":                view_idx,
                "target_model":           TARGET_PLY,
                "distractor_models":      chosen_distractors,
                "camera_distance_m":      round(float(camera_info["distance"]),    4),
                "camera_height_m":        round(float(camera_info.get("camera_height_m", 0.0)), 4),
                "camera_lens_mm":         round(float(camera_info["lens_mm"]),     3),
                "camera_pitch_deg":       round(float(camera_info["pitch_deg"]),   2),
                "camera_azimuth_deg":     round(float(camera_info.get("azimuth_deg", 0.0)), 2),
                "target_bbox_area_ratio": round(float(camera_info["ratio"]),       4),
                "scene_bbox_area_ratio":  round(float(camera_info.get("scene_ratio", -1.0)), 4),
                "dr_target_scale":        round(float(target_state.get("dr_scale", args.scale_mm)), 6),
                "motion_blur":            use_mblur,
            })
            print(
                f"[frame {frame_idx:04d}] layout={layout_idx} view={view_idx} "
                f"target={camera_info['ratio']:.3f} scene={camera_info.get('scene_ratio', -1.0):.3f} "
                f"dist={camera_info['distance']:.3f}m"
                + ("  mblur" if use_mblur else "")
            )

    write_scene_metadata(output_dir, frame_records)

    with open(output_dir / "instance_to_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "target_ply":     TARGET_PLY,
            "category_id_map": {
                str(CATEGORY_BACKGROUND): "background",
                str(CATEGORY_TARGET):     TARGET_PLY,
                str(CATEGORY_DISTRACTOR): "distractor_parts",
            },
            "available_distractors": distractor_names,
            "cfg": {
                "N_OBJ_MIN":  Cfg.N_OBJ_MIN,
                "N_OBJ_MAX":  Cfg.N_OBJ_MAX,
                "SCENE_OCC_LO": Cfg.SCENE_OCC_LO,
                "SCENE_OCC_HI": Cfg.SCENE_OCC_HI,
                "DR_SCALE":   [Cfg.DR_SCALE_LO, Cfg.DR_SCALE_HI],
                "CAM_DIST":   [Cfg.CAM_DIST_MIN, Cfg.CAM_DIST_MAX],
                "CAM_LENS":   [Cfg.CAM_LENS_MIN, Cfg.CAM_LENS_MAX],
            },
        }, f, indent=2, ensure_ascii=False)

    print(f"渲染完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
