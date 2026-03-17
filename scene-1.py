import blenderproc as bproc
import importlib.util
import random
import sys
from pathlib import Path

import bpy


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
_ORIGINAL_RANDOMIZE_PART_MATERIAL = base.randomize_part_material


def refresh_background_materials_patched(floor, palette):
    blender_obj = base.get_blender_obj(floor)
    old_mats = list(blender_obj.data.materials)
    blender_obj.data.materials.clear()
    for m in old_mats:
        if m is not None:
            try:
                bpy.data.materials.remove(m, do_unlink=True)
            except Exception:
                pass

    floor_mat = bpy.data.materials.new(name="FloorMat")
    base.assign_material_to_mesh(floor, floor_mat)

    ground_texture_path = Path("picture") / "blue.png"
    scale_floor = (random.uniform(60.0, 63.0), random.uniform(60.0, 63.0))
    base.rebuild_floor_material(
        floor_mat,
        palette,
        scale_xy=scale_floor,
        texture_path=ground_texture_path,
    )

    # 地面去反光：仅覆盖 floor 材质，不影响零件材质
    if floor_mat.use_nodes and floor_mat.node_tree is not None:
        for node in floor_mat.node_tree.nodes:
            if node.type != "BSDF_PRINCIPLED":
                continue
            roughness = node.inputs.get("Roughness")
            if roughness is not None:
                roughness.default_value = 1.0
            specular = node.inputs.get("Specular")
            if specular is not None:
                specular.default_value = 0.0
            spec_ior = node.inputs.get("Specular IOR Level")
            if spec_ior is not None:
                spec_ior.default_value = 0.0
            metallic = node.inputs.get("Metallic")
            if metallic is not None:
                metallic.default_value = 0.0


def _degloss_material(material):
    if material is None or not material.use_nodes or material.node_tree is None:
        return
    for node in material.node_tree.nodes:
        if node.type != "BSDF_PRINCIPLED":
            continue
        metallic = node.inputs.get("Metallic")
        if metallic is not None:
            metallic.default_value = 0.0
        roughness = node.inputs.get("Roughness")
        if roughness is not None:
            roughness.default_value = 1.0
        spec_ior = node.inputs.get("Specular IOR Level")
        if spec_ior is not None:
            spec_ior.default_value = 0.0
        specular = node.inputs.get("Specular")
        if specular is not None:
            specular.default_value = 0.0


def randomize_part_material_patched(mesh_obj):
    _ORIGINAL_RANDOMIZE_PART_MATERIAL(mesh_obj)
    blender_obj = base.get_blender_obj(mesh_obj)
    for mat in blender_obj.data.materials:
        _degloss_material(mat)


def configure_background_and_lights_patched(haven_path, lights, focus_point):
    # 降低环境底亮度，避免整体过曝
    bproc.renderer.set_world_background([0.08, 0.08, 0.08], strength=0.18)

    fx = float(focus_point[0])
    fy = float(focus_point[1])
    fz = float(focus_point[2])

    # 三点光改为低能量+大半径柔光
    lights["top"].set_location([fx, fy, fz + 1.0])
    lights["top"].set_energy(random.uniform(6.0, 10.0))
    lights["top"].set_radius(random.uniform(4.0, 6.0))
    lights["top"].set_color([1.0, 0.98, 0.95])

    lights["fill"].set_location([fx + 0.8, fy + 0.8, fz + 0.7])
    lights["fill"].set_energy(random.uniform(5.0, 9.0))
    lights["fill"].set_radius(random.uniform(4.0, 6.0))
    lights["fill"].set_color([1.0, 0.98, 0.95])

    lights["rim"].set_location([fx - 0.8, fy - 0.8, fz + 0.7])
    lights["rim"].set_energy(random.uniform(5.0, 9.0))
    lights["rim"].set_radius(random.uniform(4.0, 6.0))
    lights["rim"].set_color([1.0, 0.98, 0.95])


base.refresh_background_materials = refresh_background_materials_patched
base.randomize_part_material = randomize_part_material_patched
base.configure_background_and_lights = configure_background_and_lights_patched


if __name__ == "__main__":
    base.main()