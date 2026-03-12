import blenderproc as bproc
# 固定排布：3 个零件紧密排列在底部，1 个零件靠在其中某个边上，
# 目标零件（obj_000001.ply）立在这三个零件之上。共 5 个零件，无物理模拟。

import argparse
import json
import os
from pathlib import Path

import numpy as np


TARGET_PLY = "obj_000001.ply"
SCALE_MM_TO_M = 0.001

# 模型尺寸 (mm)，用于计算布局，来自 models_info.json
MODEL_SIZES_MM = {
    "obj_000001.ply": (231, 186, 106),
    "obj_000002.ply": (394, 194, 173),
    "obj_000003.ply": (73, 50, 20),
    "obj_000004.ply": (170, 108, 123),
    "obj_000005.ply": (170, 119, 123),
    "obj_000006.ply": (146, 302, 126),
    "obj_000007.ply": (200, 90, 73),
    "obj_000008.ply": (59, 90, 59),
    "obj_000009.ply": (180, 250, 83),
    "obj_000010.ply": (136, 100, 83),
    "obj_000011.ply": (117, 110, 105),
}


def parse_args():
    parser = argparse.ArgumentParser(description="BlenderProc 2 手动布局渲染（5 零件固定排布）")
    parser.add_argument("model_dir", type=str, default="models-simple", nargs="?",
                        help="PLY 模型目录")
    parser.add_argument("--haven-path", type=str, default="resources/haven",
                        help="Haven 资源目录")
    parser.add_argument("-o", "--output", type=str, default="output_manual",
                        help="输出目录")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="渲染帧数")
    parser.add_argument("--depth-of-field", action="store_true", default=True)
    parser.add_argument("--no-depth-of-field", action="store_false", dest="depth_of_field")
    parser.add_argument("--resolution", type=int, nargs=2, default=[800, 600],
                        metavar=("W", "H"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def mm_to_m(size_mm):
    return tuple(s * SCALE_MM_TO_M for s in size_mm)


def load_models_for_layout(model_dir: Path, scale: float) -> list:
    """
    加载 5 个零件：
    - 底部 3 个：obj_000003, obj_000008, obj_000011（较小，便于紧密排列）
    - 靠边 1 个：obj_000010（靠在中间那个的边上）
    - 顶部 1 个：obj_000001（目标零件）
    """
    layout = [
        ("obj_000003.ply", "bottom_left"),
        ("obj_000008.ply", "bottom_center"),
        ("obj_000011.ply", "bottom_right"),
        ("obj_000010.ply", "leaning"),
        ("obj_000001.ply", "target_top"),
    ]
    objs = []
    cached = {}
    for ply_name, role in layout:
        fp = str(model_dir / ply_name)
        if not Path(fp).exists():
            raise FileNotFoundError(f"模型不存在: {ply_name}")
        loaded = bproc.loader.load_obj(fp, cached_objects=cached)
        for o in loaded:
            o.set_scale([scale, scale, scale])
            o.move_origin_to_bottom_mean_point()
            o.set_shading_mode("FLAT")
            o.set_cp("role", role)
            o.set_cp("model_id", ply_name)
            o.set_cp("category_id", len(objs))
            objs.append(o)
    return objs


def place_manual_layout(objs: list, scale: float):
    """
    手动放置 5 个零件，符合物理直觉：
    - 底部 3 个紧密排列
    - 1 个靠在中间零件的边上（略倾斜）
    - 目标零件立在 3 个底部零件之上
    """
    roles = {o.get_cp("role"): o for o in objs}
    sizes = {o.get_cp("model_id"): mm_to_m(MODEL_SIZES_MM.get(o.get_cp("model_id"), (100, 100, 100))) 
             for o in objs}

    # 底部 3 个：紧密排列，沿 X 轴
    # obj_000003: 73x50x20 mm -> 0.073x0.05x0.02 m
    # obj_000008: 59x90x59 mm
    # obj_000011: 117x110x105 mm
    s3 = sizes["obj_000003.ply"]  # (x, y, z)
    s8 = sizes["obj_000008.ply"]
    s11 = sizes["obj_000011.ply"]

    # 从左到右排列，间隔小
    x3, y3, z3 = s3[0], s3[1], s3[2]
    x8, y8, z8 = s8[0], s8[1], s8[2]
    x11, y11, z11 = s11[0], s11[1], s11[2]

    gap = 0.01  # 1cm 间隙
    # 底部 3 个中心 y 对齐，z=0（地面）
    left_x = -x3/2 - gap - x8/2
    center_x = 0
    right_x = x8/2 + gap + x11/2

    roles["bottom_left"].set_location([left_x, 0, 0])
    roles["bottom_left"].set_rotation_euler([0, 0, 0])

    roles["bottom_center"].set_location([center_x, 0, 0])
    roles["bottom_center"].set_rotation_euler([0, 0, 0])

    roles["bottom_right"].set_location([right_x, 0, 0])
    roles["bottom_right"].set_rotation_euler([0, 0, 0])

    # 底部最高点（用于放置目标）
    h_bottom = max(z3, z8, z11)

    # 靠边的零件：靠在中间零件（obj_000008）的右侧，略倾斜
    # 绕 Y 轴旋转约 25°，使其“靠”在中间零件上
    lean_obj = roles["leaning"]
    lean_sz = sizes["obj_000010.ply"][2]
    lean_sx = sizes["obj_000010.ply"][0]
    lean_x = center_x + x8/2 + lean_sx/2 * 0.6  # 部分重叠
    lean_z = h_bottom * 0.3  # 底部略高于地面，靠在中间零件侧面
    lean_obj.set_location([lean_x, 0.05, lean_z])
    lean_obj.set_rotation_euler([0, np.deg2rad(25), 0])  # 向后倾

    # 目标零件：立在 3 个底部零件之上，居中
    target_obj = roles["target_top"]
    target_sz = sizes["obj_000001.ply"][2]
    target_obj.set_location([center_x, 0, h_bottom])
    target_obj.set_rotation_euler([0, 0, np.deg2rad(15)])  # 略旋转增加真实感


def create_floor(haven_path: Path):
    floor = bproc.object.create_primitive("PLANE", size=4.0)
    floor.set_location([0, 0, -0.02])
    floor.set_rotation_euler([0, 0, 0])
    floor.set_cp("category_id", 999)
    floor.set_name("floor")
    if haven_path.exists():
        try:
            mats = bproc.loader.load_haven_mat(str(haven_path), used_assets=["blue_metal_plate"])
            if mats:
                floor.replace_materials(mats[0])
        except Exception:
            pass
    return floor


def setup_lighting(haven_path: Path):
    INDUSTRIAL_BG = "blue_metal_plate_diff_2k.jpg"
    bg_path = haven_path / "textures" / "blue_metal_plate" / INDUSTRIAL_BG
    if bg_path.exists():
        bproc.world.set_world_background_hdr_img(
            str(bg_path),
            strength=np.random.uniform(1.4, 1.8),
            rotation_euler=[0, 0, np.random.uniform(0, np.pi * 2)]
        )
    else:
        bproc.renderer.set_world_background([0.3, 0.32, 0.35], strength=0.7)
    light = bproc.types.Light(light_type="AREA")
    light.set_location([0, 0, 1.2])
    light.set_energy(220)
    light.set_radius(1.5)


def add_camera_poses(num_frames: int, poi: np.ndarray):
    for i in range(num_frames):
        angle = 2 * np.pi * i / num_frames + np.random.uniform(-0.2, 0.2)
        radius = np.random.uniform(0.5, 0.9)
        height = np.random.uniform(0.15, 0.5)
        loc = np.array([
            poi[0] + radius * np.cos(angle),
            poi[1] + radius * np.sin(angle),
            poi[2] + height
        ])
        forward = poi - loc
        forward /= np.linalg.norm(forward)
        rot = bproc.camera.rotation_from_forward_vec(forward, up_axis="Z", inplane_rot=np.random.uniform(-0.1, 0.1))
        cam2world = bproc.math.build_transformation_mat(loc, rot)
        bproc.camera.add_camera_pose(cam2world)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    model_dir = Path(args.model_dir).resolve()
    haven_path = Path(args.haven_path).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bproc.init()
    bproc.camera.set_resolution(*args.resolution)

    # 1. 地面
    floor = create_floor(haven_path)

    # 2. 光照
    setup_lighting(haven_path)

    # 3. 加载并放置 5 个零件（无物理）
    part_objs = load_models_for_layout(model_dir, SCALE_MM_TO_M)
    place_manual_layout(part_objs, SCALE_MM_TO_M)

    # 4. 相机
    poi = np.array([0, 0, 0.12])
    add_camera_poses(args.num_frames, poi)

    if args.depth_of_field:
        focal_empty = bproc.object.create_empty("dof_focus")
        focal_empty.set_location(poi)
        bproc.camera.add_depth_of_field(focal_empty, fstop_value=4.0)

    # 5. 渲染
    bproc.renderer.enable_depth_output(activate_antialiasing=True)
    bproc.renderer.enable_segmentation_output(map_by="category_id")
    data = bproc.renderer.render()

    # 6. 写入
    bproc.writer.write_hdf5(str(output_dir), data)

    instance_to_model = {"target_ply": TARGET_PLY, "instance_to_model": {}}
    for i, obj in enumerate(part_objs):
        instance_to_model["instance_to_model"][str(i)] = obj.get_cp("model_id")
    with open(output_dir / "instance_to_model.json", "w", encoding="utf-8") as f:
        json.dump(instance_to_model, f, indent=2, ensure_ascii=False)

    rgb_dir = output_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    try:
        import imageio
        for i, img in enumerate(data["colors"]):
            imageio.imwrite(str(rgb_dir / f"{i:04d}.png"), img)
    except Exception:
        pass

    print(f"渲染完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
