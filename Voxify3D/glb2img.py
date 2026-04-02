import sys
import os
import importlib.util
import json
import math
import random
import argparse
import shutil

"""
env -u PYTHONPATH -u PYTHONHOME blender -b -P glb2img.py -- \
  --input_dir /project2/yichuanh/Voxify3D/DVGO_Gumbel/data/GLB/lamb \
  --n_views 100 --res 1200

"""

# --- Step 1: Bootstrap environment (server-specific) ---
def _bootstrap_env():
    # Server-specific conda environment path
    conda_lib = "/home_nfs/rody/miniconda3/envs/DVGO/lib/python3.12"
    np_sp = os.path.join(conda_lib, "site-packages")
    system_dynload = "/usr/lib/python3.12/lib-dynload"

    # Attempt to hijack _ctypes to avoid symbol errors
    target_so = None
    if os.path.exists(system_dynload):
        for f in os.listdir(system_dynload):
            if f.startswith("_ctypes") and f.endswith(".so"):
                target_so = os.path.join(system_dynload, f)
                break
    if target_so and "_ctypes" not in sys.modules:
        try:
            spec = importlib.util.spec_from_file_location("_ctypes", target_so)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules["_ctypes"] = module
        except: pass

    # Inject paths
    new_paths = [p for p in sys.path if "blender" in p.lower()]
    new_paths.extend(["/usr/lib/python3.12", np_sp, conda_lib])
    sys.path[:] = list(dict.fromkeys(new_paths))

# Run bootstrap
try:
    import numpy as np
    import bpy
except ImportError:
    _bootstrap_env()
    import numpy as np
    import bpy

import mathutils

# --- Step 2: Utility functions ---

def parse_args(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else: argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default=None)
    p.add_argument("--glb", type=str, default=None)
    p.add_argument("--n_views", type=int, default=100)
    p.add_argument("--res", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--margin", type=float, default=1.2)
    return p.parse_args(argv)

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Use EEVEE (closer to real-time unlit rendering)
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # Disable color management to preserve original colors
    view = scene.view_settings
    view.view_transform = 'Standard'  # No Filmic
    view.look = 'None'
    view.exposure = 0.0
    view.gamma = 1.0

    return scene


def import_glb(glb_path):
    bpy.ops.import_scene.gltf(filepath=glb_path)
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]

def center_objects(mesh_objs):
    min_v = mathutils.Vector((float("inf"),)*3)
    max_v = mathutils.Vector((float("-inf"),)*3)
    for obj in mesh_objs:
        for corner in obj.bound_box:
            v = obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_v[i] = min(min_v[i], v[i]); max_v[i] = max(max_v[i], v[i])
    center = (min_v + max_v) * 0.5
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "EMPTY", "ARMATURE"}:
            obj.location -= center
    return (max_v - min_v).length * 0.5

def setup_lighting_noop():
    scene = bpy.context.scene

    # Remove all lights
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Set world background to black
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    nodes.clear()
    bg = nodes.new(type="ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    bg.inputs["Strength"].default_value = 0.0
    out = nodes.new(type="ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], out.inputs["Surface"])



def force_unlit_emission_for_all_materials():
    import bpy
    for mat in bpy.data.materials:
        if mat is None:
            continue
        mat.use_nodes = True
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links

        # Find Principled BSDF
        principled = None
        for n in nodes:
            if n.type == "BSDF_PRINCIPLED":
                principled = n
                break
        if principled is None:
            continue

        # Find Material Output
        out = None
        for n in nodes:
            if n.type == "OUTPUT_MATERIAL":
                out = n
                break
        if out is None:
            out = nodes.new(type="ShaderNodeOutputMaterial")

        # Create Emission node
        emis = nodes.new(type="ShaderNodeEmission")
        emis.inputs["Strength"].default_value = 1.0

        # Connect Principled Base Color source to Emission Color
        base_in = principled.inputs.get("Base Color", None)
        if base_in is not None and base_in.is_linked:
            src = base_in.links[0].from_socket
            links.new(src, emis.inputs["Color"])
        else:
            # No texture — use base color value directly
            if base_in is not None:
                emis.inputs["Color"].default_value = base_in.default_value

        # Redirect output to Emission
        for l in list(out.inputs["Surface"].links):
            links.remove(l)
        links.new(emis.outputs["Emission"], out.inputs["Surface"])



def setup_lighting(radius_hint):
    scene = bpy.context.scene

    # Ensure world exists
    if scene.world is None:
        new_world = bpy.data.worlds.new("NewWorld")
        scene.world = new_world

    scene.world.use_nodes = True
    nodes = scene.world.node_tree.nodes
    nodes.clear()

    # Background node for soft ambient light
    node_background = nodes.new(type='ShaderNodeBackground')
    node_background.inputs['Color'].default_value = (1.0, 1.0, 1.0, 1.0)
    node_background.inputs['Strength'].default_value = 0.5

    node_output = nodes.new(type='ShaderNodeOutputWorld')
    scene.world.node_tree.links.new(node_background.outputs['Background'], node_output.inputs['Surface'])

    # Remove existing lights
    for obj in list(scene.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    def add_area(name, loc, rot_euler, energy, size):
        light_data = bpy.data.lights.new(name=name, type="AREA")
        light_data.energy = energy
        light_data.size = size
        light_obj = bpy.data.objects.new(name=name, object_data=light_data)
        scene.collection.objects.link(light_obj)
        light_obj.location = loc
        light_obj.rotation_euler = rot_euler
        return light_obj

    r = max(radius_hint, 1e-3)
    size = 3.0 * r

    # Tuned energy values to avoid overexposure
    add_area("Key", (2.5*r, -2.0*r, 2.0*r), (math.radians(55), 0, math.radians(35)), 300, size)
    add_area("Fill", (-2.0*r, -2.5*r, 1.2*r), (math.radians(65), 0, math.radians(-35)), 150, size)
    add_area("Rim", (0.0, 3.0*r, 2.2*r), (math.radians(120), 0, math.radians(180)), 200, size)

def setup_camera(res, ortho_scale):
    cam_data = bpy.data.cameras.new("OrthoCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_obj = bpy.data.objects.new("OrthoCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    bpy.context.scene.render.resolution_x = res
    bpy.context.scene.render.resolution_y = res
    return cam_obj

def render_and_save(scene, out_path, cam_obj, matrix):
    cam_obj.matrix_world = mathutils.Matrix(matrix)
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return [[float(c) for c in r] for r in cam_obj.matrix_world]

def normalize_objects(mesh_objs, target_extent=2.0):
    min_v = mathutils.Vector((float("inf"),)*3)
    max_v = mathutils.Vector((float("-inf"),)*3)

    for obj in mesh_objs:
        for corner in obj.bound_box:
            v = obj.matrix_world @ mathutils.Vector(corner)
            min_v = mathutils.Vector(map(min, min_v, v))
            max_v = mathutils.Vector(map(max, max_v, v))

    center = (min_v + max_v) * 0.5
    extent = (max_v - min_v)
    max_len = max(extent)

    scale = target_extent / max_len

    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "EMPTY", "ARMATURE"}:
            obj.location = (obj.location - center) * scale
            obj.scale *= scale

    return target_extent

def force_opaque_materials():
    for mat in bpy.data.materials:
        if not mat:
            continue
        mat.use_nodes = True

        # Force opaque blend mode (important for Eevee)
        mat.blend_method = 'OPAQUE'
        mat.shadow_method = 'OPAQUE'
        mat.use_backface_culling = False  # Some models with inverted normals may show holes

        nt = mat.node_tree
        if not nt:
            continue

        # Find Principled BSDF and force alpha=1
        for n in nt.nodes:
            if n.type == "BSDF_PRINCIPLED":
                if "Alpha" in n.inputs:
                    n.inputs["Alpha"].default_value = 1.0
                break


def main():
    args = parse_args(sys.argv)
    if args.glb: glb_path = args.glb
    else:
        glbs = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".glb")])
        glb_path = os.path.join(args.input_dir, glbs[0])

    input_dir = os.path.dirname(glb_path)
    random.seed(args.seed)
    scene = reset_scene()
    mesh_objs = import_glb(glb_path)
    WORLD_SIZE = 2.0 # Revise here. #2.5 default
    normalize_objects(mesh_objs, WORLD_SIZE)
    ortho_scale = WORLD_SIZE  * 1.5

    setup_lighting_noop()
    force_unlit_emission_for_all_materials()
    force_opaque_materials()
    cam_obj = setup_camera(args.res, ortho_scale)



    # --- Render 6 canonical views ---
    dist = 15.0
    canonical_matrices = [
        [[-1, 0, 0, 0], [0, 0, 1, dist], [0, 1, 0, 0], [0, 0, 0, 1]], # Back
        [[0, 0, 1, dist], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # Right
        [[0, 0, -1, -dist], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], # Left
        [[1, 0, 0, 0], [0, 0, -1, -dist], [0, 1, 0, 0], [0, 0, 0, 1]], # Front
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, dist], [0, 0, 0, 1]], # Top
        [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, -dist], [0, 0, 0, 1]], # Bottom
    ]

    six_view_root = os.path.join(input_dir, "6views")
    os.makedirs(os.path.join(six_view_root, "train"), exist_ok=True)
    six_frames = []
    for i, m in enumerate(canonical_matrices):
        file_rel = f"train/r_{i}"
        m_out = render_and_save(scene, os.path.join(six_view_root, file_rel + ".png"), cam_obj, m)
        six_frames.append({"file_path": file_rel, "transform_matrix": m_out})

    with open(os.path.join(six_view_root, "transforms_train.json"), "w") as f:
        json.dump({"camera_angle_x": 0.691111147403717, "frames": six_frames}, f, indent=4)

    # --- Render random views ---
    ortho_root = os.path.join(input_dir, "ortho")
    os.makedirs(os.path.join(ortho_root, "train"), exist_ok=True)
    random_frames = []
    for i in range(args.n_views):
        az = random.uniform(0, 2 * math.pi)

        # Uniform spherical sampling: u = sin(el) uniformly distributed in [-1, 1]
        u  = random.uniform(-1.0, 1.0)
        el = math.asin(u)  # el in [-90deg, 90deg]

        loc = mathutils.Vector((
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el),
        ))



        m = mathutils.Matrix.LocRotScale(loc, (-loc.normalized()).to_track_quat('-Z', 'Y'), None)
        file_rel = f"train/random_view_{i}"
        m_out = render_and_save(scene, os.path.join(ortho_root, file_rel + ".png"), cam_obj, m)
        random_frames.append({"file_path": file_rel, "transform_matrix": m_out})

    train_json = os.path.join(ortho_root, "transforms_train.json")
    val_json = os.path.join(ortho_root, "transforms_val.json")

    with open(train_json, "w") as f:
        json.dump({"camera_angle_x": 0.691111147403717, "frames": random_frames}, f, indent=4)

    shutil.copyfile(train_json, val_json)


if __name__ == "__main__":
    main()
