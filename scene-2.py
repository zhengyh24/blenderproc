import blenderproc as bproc

import importlib.util
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
import mathutils
from mathutils import Vector


def _load_base_scene_module():
    try:
        import scene as _base
        return _base
    except ModuleNotFoundError:
        scene_path = Path(__file__).resolve().with_name("scene.py")
        spec = importlib.util.spec_from_file_location("base_scene_module", str(scene_path))
        if spec is None or spec.loader is None:
            raise FileNotFoundError("无法加载 scene.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["base_scene_module"] = module
        spec.loader.exec_module(module)
        return module


base = _load_base_scene_module()

# 1) 覆写核心参数：32 物体（1 目标 + 31 干扰）+ 抬高相机视角
base.Cfg.N_OBJ_MIN = 32
base.Cfg.N_OBJ_MAX = 32
base.Cfg.CAM_DIST_MIN = 0.6
base.Cfg.CAM_DIST_MAX = 1.0
base.Cfg.CAM_ELEV_DEG_MIN = 15.0
base.Cfg.CAM_ELEV_DEG_MAX = 40.0

_original_randomize_part_material = base.randomize_part_material


def randomize_part_material_patched(mesh_obj):
    # 1. 先执行原版逻辑，获取底色和基础节点
    _original_randomize_part_material(mesh_obj)

    # 2. 拦截并覆写 PBR 物理参数（阳极氧化/喷砂金属）
    bo = base.get_blender_obj(mesh_obj)
    for mat in bo.data.materials:
        if mat and mat.use_nodes and mat.node_tree is not None:
            for node in mat.node_tree.nodes:
                if node.type != "BSDF_PRINCIPLED":
                    continue

                metallic_in = node.inputs.get("Metallic")
                if metallic_in is not None and float(metallic_in.default_value) < 0.5:
                    metallic_in.default_value = random.uniform(0.8, 1.0)

                rough_in = node.inputs.get("Roughness")
                if rough_in is not None:
                    rough_in.default_value = random.uniform(0.45, 0.65)

                spec_ior_in = node.inputs.get("Specular IOR Level")
                if spec_ior_in is not None:
                    spec_ior_in.default_value = random.uniform(0.2, 0.3)


def _augment_model_stats_from_info(info_path: Path, model_stats: dict):
    if not info_path.exists():
        return
    with open(info_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for idx_str, item in raw.items():
        name = f"obj_{int(idx_str):06d}.ply"
        if name not in model_stats:
            model_stats[name] = {
                "size": np.array([item["size_x"], item["size_y"], item["size_z"]], dtype=np.float32),
                "diameter": float(item.get("diameter", max(item["size_x"], item["size_y"], item["size_z"]))),
            }


def load_scene_objects_patched(model_dir: Path, scale: float, model_stats: dict) -> tuple:
    root = Path.cwd()
    simple_dir_candidates = [
        (root / "models_simple").resolve(),
        (root / "models-simple").resolve(),
    ]
    cad_dir_candidates = [
        (root / "models_cad").resolve(),
        (root / "models-cad").resolve(),
    ]
    simple_dir = next((p for p in simple_dir_candidates if p.exists()), None)
    cad_dir = next((p for p in cad_dir_candidates if p.exists()), None)

    target_candidates = []
    if simple_dir is not None:
        target_candidates.append((simple_dir / base.TARGET_PLY).resolve())
    target_candidates.extend([
        (root / "models_simple" / base.TARGET_PLY).resolve(),
        (root / "models-simple" / base.TARGET_PLY).resolve(),
    ])
    target_path = next((p for p in target_candidates if p.exists()), None)

    if target_path is None:
        raise FileNotFoundError(f"目标模型不存在，已尝试: {target_candidates}")
    if simple_dir is None:
        raise FileNotFoundError(f"models_simple 目录不存在，已尝试: {simple_dir_candidates}")
    if cad_dir is None:
        raise FileNotFoundError(f"models_cad 目录不存在，已尝试: {cad_dir_candidates}")

    # 注意：Linux 上 glob("*.ply") 不匹配 ".PLY"；这里做大小写无关收集
    simple_plys = sorted([p for p in simple_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ply"])
    cad_plys = sorted([p for p in cad_dir.iterdir() if p.is_file() and p.suffix.lower() == ".ply"])
    pipe_filenames = ["1.ply", "2.ply", "3.ply"]
    simple_pool = [p for p in simple_plys if p.name != base.TARGET_PLY and p.name.lower() not in pipe_filenames]
    pipe_paths = [p for p in simple_plys if p.name.lower() in pipe_filenames]
    if len(pipe_paths) != 3:
        print(f"警告：未找全 1.PLY, 2.PLY, 3.PLY，目前找到 {len(pipe_paths)} 个")
    cad_pool = [p for p in cad_plys if p.name != base.TARGET_PLY]
    if len(simple_pool) < 9:
        raise RuntimeError(f"models_simple 中可用干扰 .ply 不足 9，当前 {len(simple_pool)}")
    if len(cad_pool) < 18:
        raise RuntimeError(f"models_cad 中可用干扰 .ply 不足 18，当前 {len(cad_pool)}")

    selected_simple = random.sample(simple_pool, 9)
    selected_cad = random.sample(cad_pool, 18)  # 一次性抽取18个保证不重复
    selected_cad2 = selected_cad[:9]
    selected_cad3 = selected_cad[9:]

    _augment_model_stats_from_info((root / "models_simple" / "models_info.json").resolve(), model_stats)
    _augment_model_stats_from_info((root / "models-simple" / "models_info.json").resolve(), model_stats)
    _augment_model_stats_from_info((simple_dir / "models_info.json").resolve(), model_stats)
    _augment_model_stats_from_info((root / "models_cad" / "models_info.json").resolve(), model_stats)
    _augment_model_stats_from_info((cad_dir / "models_info.json").resolve(), model_stats)

    cached = {}
    loaded_objs = []
    distractor_objs = {}
    selected_names = []

    # 目标：models_simple/obj_000001.ply
    target_loaded = bproc.loader.load_obj(str(target_path), cached_objects=cached)
    if not target_loaded:
        raise RuntimeError(f"目标模型加载失败: {target_path}")
    target_obj = target_loaded[0]
    base.setup_mesh_obj(target_obj, scale, base.CATEGORY_TARGET, target_path.name)
    target_obj.set_name(f"mesh_{target_path.stem}")
    base.randomize_part_material(target_obj)
    loaded_objs.append(target_obj)

    def add_distractor(cad_path: Path, prefix: str):
        unique_name = f"{prefix}_{cad_path.name}"
        loaded = bproc.loader.load_obj(str(cad_path), cached_objects=cached)
        if not loaded:
            return
        obj = loaded[0]

        base.setup_mesh_obj(obj, scale, base.CATEGORY_DISTRACTOR, cad_path.name)
        obj.set_name(f"mesh_{prefix}_{cad_path.stem}")
        base.randomize_part_material(obj)
        distractor_objs[unique_name] = obj
        loaded_objs.append(obj)

        # 为管道等“非 obj_*.ply”模型补齐尺寸统计；同时写入小写 key，便于后续用 "1.ply" 访问
        base_key = cad_path.name.lower()
        if base_key not in model_stats:
            bo = base.get_blender_obj(obj)
            dims = np.array([float(bo.dimensions.x), float(bo.dimensions.y), float(bo.dimensions.z)], dtype=np.float32)
            model_stats[base_key] = {
                "size": dims,
                "diameter": float(np.linalg.norm(dims)),
            }
        model_stats[unique_name] = {
            "size": np.array(model_stats[base_key]["size"], dtype=np.float32),
            "diameter": float(model_stats[base_key]["diameter"]),
        }
        selected_names.append(unique_name)

    for cad_path in selected_simple:
        add_distractor(cad_path, "simple")
    for cad_path in selected_cad2:
        add_distractor(cad_path, "cad2")
    for cad_path in selected_cad3:
        add_distractor(cad_path, "cad3")
    for pipe_path in pipe_paths:
        add_distractor(pipe_path, "pipe")

    # 额外加载一根 1.ply：用于最终第五波“单根横向落下”
    pipe_1_path = next((p for p in simple_plys if p.name.lower() == "1.ply"), None)
    if pipe_1_path:
        add_distractor(pipe_1_path, "final_pipe")

    if len(distractor_objs) != 31:
        raise RuntimeError(f"干扰物加载数量异常，期望 31，实际 {len(distractor_objs)}")

    for obj in loaded_objs:
        obj.enable_rigidbody(active=True, collision_shape="CONVEX_HULL")
        bo = base.get_blender_obj(obj)
        if bo.rigid_body is not None:
            rb = bo.rigid_body
            rb.type = "ACTIVE"
            rb.collision_shape = "CONVEX_HULL"
            rb.mass = 0.45 + 0.65 * random.random()
            rb.friction = 1.2
            rb.restitution = 0.0
            rb.linear_damping = 0.8
            rb.angular_damping = 0.8
            if hasattr(rb, "use_margin"):
                rb.use_margin = True
            if hasattr(rb, "collision_margin"):
                rb.collision_margin = 0.001

    return target_obj, distractor_objs, selected_names


def _place_wave(tx: float, ty: float, names: list[str], distractor_objs: dict, model_stats: dict,
                scale: float, target_size_m: np.ndarray, base_z: float, set_damping: float | None):
    placed_circles = []
    names_sorted = sorted(names, key=lambda n: float(model_stats[n]["diameter"]))
    r_target = max(float(target_size_m[0]), float(target_size_m[1])) / 2.0

    for model_name in names_sorted:
        obj = distractor_objs[model_name]
        base.show_mesh_obj(obj)

        dr_scale = scale * random.uniform(base.Cfg.DR_SCALE_LO, base.Cfg.DR_SCALE_HI)
        obj.set_scale([dr_scale, dr_scale, dr_scale])
        base.randomize_part_material(obj)

        r_safe = (float(model_stats[model_name]["diameter"]) * dr_scale) / 2.0 + 0.005
        if r_safe > r_target * 1.1:
            start_sr = r_safe
        else:
            start_sr = 0.0

        best_x, best_y = 0.0, 0.0
        if placed_circles:
            found = False
            sr = start_sr
            while sr <= 0.8 and not found:
                theta = 0.0
                while theta < (2.0 * math.pi):
                    cx = sr * math.cos(theta)
                    cy = sr * math.sin(theta)
                    overlaps = False
                    for px, py, pr in placed_circles:
                        if math.hypot(cx - px, cy - py) < (r_safe + pr):
                            overlaps = True
                            break
                    if not overlaps:
                        best_x, best_y = cx, cy
                        found = True
                        break
                    theta += 0.2
                sr += 0.005

        placed_circles.append((best_x, best_y, r_safe))
        z_safe = base_z + r_safe + 0.015
        z_offset = random.uniform(-0.01, 0.01)

        loc = np.array([tx + best_x, ty + best_y, z_safe + z_offset], dtype=np.float64)
        rot = np.array([
            math.radians(random.uniform(-12.0, 12.0)),
            math.radians(random.uniform(-12.0, 12.0)),
            math.radians(random.uniform(0.0, 360.0)),
        ], dtype=np.float64)
        obj.set_location(loc.tolist())
        obj.set_rotation_euler(rot.tolist())
        if set_damping is not None:
            bo = base.get_blender_obj(obj)
            if bo.rigid_body is not None:
                bo.rigid_body.linear_damping = set_damping
                bo.rigid_body.angular_damping = set_damping
        base.reset_rigidbody_state(obj)


def _scene_global_max_z(distractor_objs: dict, wave_names: list[str], include_target: bool = True) -> float:
    zs = []
    if include_target:
        for bo in bpy.data.objects:
            try:
                if bo.get("category_id", None) == base.CATEGORY_TARGET:
                    for corner in bo.bound_box:
                        p = bo.matrix_world @ Vector(corner)
                        zs.append(float(p.z))
                    break
            except Exception:
                continue
    for name in wave_names:
        obj = distractor_objs.get(name)
        if obj is None:
            continue
        bo = base.get_blender_obj(obj)
        for corner in bo.bound_box:
            p = bo.matrix_world @ Vector(corner)
            zs.append(float(p.z))
    return max(zs) if zs else base.Cfg.TABLE_Z_TOP


def compute_scene_focus_point_patched(scene_objs: list, target_state: dict) -> tuple[np.ndarray, float]:
    # 强行将视觉中心锁定在目标零件的 XY 坐标以及它稍微偏上的 Z 高度
    focus = np.array(
        [
            float(target_state["location"][0]),
            float(target_state["location"][1]),
            float(target_state["location"][2]) + float(target_state["size_m"][2]) * 0.5,
        ],
        dtype=float,
    )

    # 计算场景的近似最大高度，供给相机调整焦距使用
    zs = [float(target_state["location"][2])]
    for obj in scene_objs:
        if obj is None:
            continue
        try:
            bo = base.get_blender_obj(obj)
            if not bo.hide_render:
                for corner in bo.bound_box:
                    p = bo.matrix_world @ mathutils.Vector(corner)
                    zs.append(float(p.z))
        except Exception:
            pass
    pile_h = max(zs) - min(zs) if zs else float(target_state["size_m"][2])

    return focus, float(pile_h)


def arrange_distractors_patched(active_names: list[str], distractor_objs: dict,
                                model_stats: dict, scale: float, target_size_m: np.ndarray):
    for _, obj in distractor_objs.items():
        base.hide_mesh_obj(obj)
        bo = base.get_blender_obj(obj)
        if bo.rigid_body is not None:
            bo.rigid_body.type = "ACTIVE"
            bo.rigid_body.collision_shape = "CONVEX_HULL"
            bo.rigid_body.friction = 1.4
            bo.rigid_body.restitution = 0.0
            if hasattr(bo.rigid_body, "collision_margin"):
                bo.rigid_body.collision_margin = 0.001

    wave1_names = [n for n in active_names if n.startswith("simple_")]
    wave2_names = [n for n in active_names if n.startswith("cad2_")]
    wave3_names = [n for n in active_names if n.startswith("cad3_")]
    wave4_names = [n for n in active_names if n.startswith("pipe_")]
    wave5_names = [n for n in active_names if n.startswith("final_pipe")]

    tx, ty = 0.0, 0.0
    for bo in bpy.data.objects:
        try:
            if bo.get("category_id", None) == base.CATEGORY_TARGET:
                tx = float(bo.location.x)
                ty = float(bo.location.y)
                break
        except Exception:
            continue

    base_z_wave1 = base.Cfg.TABLE_Z_TOP + float(target_size_m[2])

    # 第一波：9 个 simple，螺旋贪心 + 中途物理
    _place_wave(tx, ty, wave1_names, distractor_objs, model_stats, scale, target_size_m, base_z_wave1, 0.5)
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=2.0,
        check_object_interval=0.5,
    )

    # 第二波：9 个 cad2，铺在第一座山上方 + 强制中途物理结算
    global_max_z = _scene_global_max_z(distractor_objs, wave1_names, include_target=True)
    base_z_wave2 = global_max_z + 0.02
    _place_wave(tx, ty, wave2_names, distractor_objs, model_stats, scale, target_size_m, base_z_wave2, 0.5)
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=2.0,
        check_object_interval=0.5,
    )

    # 第三波：9 个 cad3，铺在前两座山上方 + 强制中途结算第三波
    global_max_z3 = _scene_global_max_z(distractor_objs, wave1_names + wave2_names, include_target=True)
    base_z_wave3 = global_max_z3 + 0.02
    _place_wave(tx, ty, wave3_names, distractor_objs, model_stats, scale, target_size_m, base_z_wave3, 0.5)

    # 强制中途结算第三波
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=2.0,
        check_object_interval=0.5,
    )

    # 第四波：管道（pipe_），铺在前三座山上方
    global_max_z4 = _scene_global_max_z(distractor_objs, wave1_names + wave2_names + wave3_names, include_target=True)
    base_z_wave4 = global_max_z4 + 0.02
    _place_wave(tx, ty, wave4_names, distractor_objs, model_stats, scale, target_size_m, base_z_wave4, 0.4)

    # 第四波（Wave 4）之后：保留中途物理结算
    bproc.object.simulate_physics_and_fix_final_poses(
        min_simulation_time=1.0,
        max_simulation_time=2.0,
        check_object_interval=0.5,
    )

    # 第五波（Wave 5）：场景中心 + 前四波最高点上方，横向放置唯一一根 1.ply 管道落下
    global_max_z5 = _scene_global_max_z(
        distractor_objs,
        wave1_names + wave2_names + wave3_names + wave4_names,
        include_target=True,
    )

    if len(wave5_names) == 1:
        pipe = distractor_objs[wave5_names[0]]
        base.show_mesh_obj(pipe)

        dr_scale = scale * random.uniform(base.Cfg.DR_SCALE_LO, base.Cfg.DR_SCALE_HI)
        pipe.set_scale([dr_scale, dr_scale, dr_scale])
        base.randomize_part_material(pipe)

        # 计算安全高度：最高点 + 管道半径 + 2厘米安全缓冲
        pipe_radius = (float(model_stats["1.ply"]["diameter"]) * dr_scale) / 2.0
        z_safe = global_max_z5 + pipe_radius + 0.02

        bo = base.get_blender_obj(pipe)
        if bo.rigid_body is not None:
            bo.rigid_body.linear_damping = 0.5
            bo.rigid_body.angular_damping = 0.5

        # 将唯一的一根管道放在正中心，横向放平
        pipe.set_location([tx, ty, z_safe])
        pipe.set_rotation_euler([0.0, math.pi / 2.0, 0.0])
        base.reset_rigidbody_state(pipe)

    # 为所有干扰物统一设小阻尼
    for _, obj in distractor_objs.items():
        bo = base.get_blender_obj(obj)
        if bo.rigid_body is not None:
            bo.rigid_body.linear_damping = 0.4
            bo.rigid_body.angular_damping = 0.4

def configure_background_and_lights_patched(haven_path, lights, focus_point):
    # 极限压暗：尽量切断环境泛光
    bproc.renderer.set_world_background([0.12, 0.12, 0.12], strength=0.012)
    fp = np.asarray(focus_point, dtype=float)

    lights["top"].set_location([float(fp[0]), float(fp[1]), float(fp[2]) + 0.85])
    lights["top"].set_radius(random.uniform(1.0, 2.5))
    lights["top"].set_energy(random.uniform(34.0, 108.0))
    lights["top"].set_color([1.0, 0.98, 0.95])

    lights["fill"].set_location([float(fp[0]) + 0.6, float(fp[1]) + 0.6, float(fp[2]) + 0.35])
    lights["fill"].set_radius(random.uniform(1.0, 2.5))
    lights["fill"].set_energy(random.uniform(10.0, 40.0))
    lights["fill"].set_color([1.0, 0.98, 0.95])

    lights["rim"].set_location([float(fp[0]) - 0.6, float(fp[1]) - 0.6, float(fp[2]) + 0.35])
    lights["rim"].set_radius(random.uniform(1.0, 2.5))
    lights["rim"].set_energy(random.uniform(10.0, 40.0))
    lights["rim"].set_color([1.0, 0.98, 0.95])


def create_funnel_air_walls_patched(target_state: dict):
    return []


_original_sample_worker_camera = base.sample_worker_camera


def sample_worker_camera_patched(target_obj, target_state, scene_objs, focus_point):
    camera_info = _original_sample_worker_camera(target_obj, target_state, scene_objs, focus_point)

    # 最直接的全局压亮度手段：压曝光（避免地面/金属泛白）
    try:
        bpy.context.scene.view_settings.exposure = 0.45
    except Exception:
        pass

    cam = bpy.context.scene.camera
    cam_loc_vec = cam.matrix_world.to_translation()
    cam_axes = cam.matrix_world.to_3x3()
    cam_right = cam_axes @ mathutils.Vector((1.0, 0.0, 0.0))
    cam_up = cam_axes @ mathutils.Vector((0.0, 1.0, 0.0))
    cam_back = cam_axes @ mathutils.Vector((0.0, 0.0, 1.0))

    fp = np.asarray(focus_point, dtype=float)
    area_lights = [obj for obj in bpy.data.objects if obj.type == "LIGHT" and obj.data.type == "AREA"]
    if len(area_lights) >= 3:
        l0, l1, l2 = area_lights[0], area_lights[1], area_lights[2]

        # 主光：相机左侧 1.5m，上方 3.0m
        l0.location = cam_loc_vec - cam_right * 1.5 + cam_up * 3.0
        # 辅光：相机右侧 1.5m，上方 1.0m
        l1.location = cam_loc_vec + cam_right * 1.5 + cam_up * 1.0
        # 轮廓光：目标后方 4.0m，高度 1.5m
        l2.location = mathutils.Vector((fp[0], fp[1], fp[2])) - cam_back * 4.0 + mathutils.Vector((0.0, 0.0, 1.5))

        for light in (l0, l1, l2):
            light.rotation_euler = (mathutils.Vector((fp[0], fp[1], fp[2])) - light.location).to_track_quat("-Z", "Y").to_euler()

    return camera_info


base.load_scene_objects = load_scene_objects_patched
base.arrange_distractors = arrange_distractors_patched
base.randomize_part_material = randomize_part_material_patched
base.configure_background_and_lights = configure_background_and_lights_patched
base.create_funnel_air_walls = create_funnel_air_walls_patched
base.sample_worker_camera = sample_worker_camera_patched
base.compute_scene_focus_point = compute_scene_focus_point_patched


if __name__ == "__main__":
    base.main()
