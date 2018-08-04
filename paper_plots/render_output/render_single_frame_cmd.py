"""
Takes a set of 2D keypoints as input and renders the 3D skeleton using blender.

Assumes input is centered at (0,0,0) and will scale the skeletons to fit in [-1,1]
Does not work if skeleton length changes after each frame.
"""
import math
import bpy
import numpy as np
import sys
sys.path.append('../../src/')

from data_formats.misc import DatasetMisc


def draw_limb(p1, p2, radius, material):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    loc = (dx/2 + p1[0], dy/2 + p1[1], dz/2 + p1[2])
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=dist, location=loc)
    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    limb = bpy.context.object
    limb.rotation_euler[1] = theta
    limb.rotation_euler[2] = phi
    limb.data.materials.append(material)
    return limb


def move_limb(p1, p2, limb):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)
    loc = (dx/2 + p1[0], dy/2 + p1[1], dz/2 + p1[2])
    limb.location = loc
    phi = math.atan2(dy, dx)
    theta = math.acos(dz/dist)
    limb.rotation_euler[1] = theta
    limb.rotation_euler[2] = phi


def draw_joints(pts, radius, material):
    joints = []
    for pp in range(pts.shape[0]):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=32, size=radius, location=tuple(pts[pp, :]))
        bpy.context.object.data.materials.append(material)
        joints.append(bpy.context.object.name)
    return joints


def draw_initial_skeleton(pts, skeleton, radius_limb, radius_joint, materials, skeleton_color):
    joints = draw_joints(pts, radius_joint, materials[2])
    limbs = []
    for ii, ss in enumerate(skeleton):
        ll = draw_limb(pts[ss[0], :], pts[ss[1], :], radius_limb, materials[skeleton_color[ii]])
        limbs.append(ll.name)
    return joints, limbs


def move_skeleton(pts, skeleton, joints, limbs):
    for ii, jj in enumerate(joints):
        bpy.data.objects[jj].location = pts[ii]
    for ii, ll in enumerate(limbs):
        ss = skeleton[ii]
        move_limb(pts[ss[0], :], pts[ss[1], :], bpy.data.objects[ll])


def delete_all(joints, limbs):
    bpy.ops.object.select_all(action='DESELECT')
    for jj in joints + limbs:
        bpy.data.objects[jj].select = True
        bpy.ops.object.delete()


def set_keyframe(joints, limbs):
    for jj in joints + limbs:
        bpy.data.objects[jj].keyframe_insert(data_path="location", index=-1)
        bpy.data.objects[jj].keyframe_insert(data_path="rotation_euler", index=-1)


def create_materials():
    mat_right = bpy.data.materials.new(name="mat_right")
    mat_right.diffuse_color = (236/255.0, 76/255.0, 60/255.0) #3798DB
    mat_left = bpy.data.materials.new(name="mat_left")
    mat_left.diffuse_color = (55/255.0, 152/255.0, 219/255.0) #EC4C3C
    mat_joint = bpy.data.materials.new(name="mat_joint")
    mat_joint.diffuse_color = (55/255.0, 219/255.0, 52/255.0) #37DB34
    return [mat_right, mat_left, mat_joint]


if __name__ == "__main__":

    # specify 3d pose
    radius_limb = 0.025
    radius_joint = 0.025
    scale_factor_single = 1000

    args = sys.argv[sys.argv.index('--') + 1:]  # there has to be a space after the --
    frame_id = int(args[args.index('frame_id')+1])
    op_file = args[args.index('op_file')+1]
    pts_raw = args[args.index('pts_raw')+1:]  # needs to be at the end
    pts_raw = np.array([float(pp) for pp in pts_raw])

    print('\nRendering frame', frame_id, '\n')

    if pts_raw.shape[0] == 17*3:
        dataset_type = 'human36_17k'
    elif pts_raw.shape[0] == 14*3:
        dataset_type = 'lsp_14k'
    else:
        print('Error pts not the correct shape')

    misc = DatasetMisc(dataset_type)

    pts = pts_raw.reshape(len(misc.KEYPOINTS_3D), 3)
    pts = np.dot(pts, np.array([[1,0,0], [0,0,-1], [0,1,0]]))
    pts /= (scale_factor_single + scale_factor_single*0.2)

    materials = create_materials()
    bpy.data.scenes["Scene"].frame_start = 0
    bpy.data.scenes["Scene"].render.image_settings.file_format = 'JPEG'
    bpy.data.scenes["Scene"].render.filepath = op_file

    bpy.context.scene.frame_set(0)
    joints, limbs = draw_initial_skeleton(pts, misc.SKELETON_3D_IDX,
                                          radius_limb, radius_joint, materials,
                                          misc.SKELETON_3D_COLOR)
    set_keyframe(joints, limbs)

    bpy.ops.render.render(write_still=True)
