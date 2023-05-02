import os
import numpy as np
import nibabel as nib

from . import renderer

main_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# basic parameters
surface = 'inflated_MSMAll'
expand = 0

# load an example dscalar
dscalar_file = f'{main_path}/data/templates/HCP/dscalars/ones.dscalar.nii'
dscalar = nib.load(dscalar_file)

brain_models = [x for x in dscalar.header.get_index_map(1).brain_models]

# load surfaces for visualization
left_surface_file = f'{main_path}/data/templates/HCP/surfaces/S1200.L.{surface}.32k_fs_LR.surf.gii'
left_surface = nib.load(left_surface_file)
right_surface_file = f'{main_path}/data/templates/HCP/surfaces/S1200.R.{surface}.32k_fs_LR.surf.gii'
right_surface = nib.load(right_surface_file)

# extract surface information
lx, ly, lz = left_surface.darrays[0].data.T
lt = left_surface.darrays[1].data
rx, ry, rz = right_surface.darrays[0].data.T
rt = right_surface.darrays[1].data

# combine into a complete brain
lrx = np.concatenate([lx - expand, rx + expand])
lry = np.concatenate([ly, ry])
lrz = np.concatenate([lz, rz])
lrt = np.concatenate([lt, (rt + lx.shape[0])])

lxyz = left_surface.darrays[0].data
rxyz = right_surface.darrays[0].data
lrxyz = np.array([lrx, lry, lrz]).T

# create a mapping between surface and cifti vertices
left_cortical_surface_model, right_cortical_surface_model = brain_models[0], brain_models[1]
cifti_to_surface = {}
surface_to_cifti = {}
for (i, x) in enumerate(left_cortical_surface_model.vertex_indices):
    cifti_to_surface[i] = x
    surface_to_cifti[x] = i
for (i, x) in enumerate(right_cortical_surface_model.vertex_indices):
    cifti_to_surface[i + right_cortical_surface_model.index_offset] = x + rx.shape[0]
    surface_to_cifti[x + rx.shape[0]] = i + right_cortical_surface_model.index_offset

# construct data over surface
surface_mask = list(surface_to_cifti.keys())

arbitrary_colors = np.concatenate([
    ((lrxyz - lrxyz.min(0)) / (lrxyz.max(0) - lrxyz.min(0))),
    np.ones((lrxyz.shape[0], 1)) * 0.9
], axis=1)

centered_lrxyz = lrxyz - ((lrxyz.max(0) + lrxyz.min(0)) / 2)


# initialization
my_renderer = renderer.Renderer_panda3d(camera_pos=(400, 0, 0), camera_target=(0, 0, 0), camera_fov=35, camera_rotation=0)

# surface mesh rendering
my_renderer.add_mesh(centered_lrxyz, lrt, arbitrary_colors)

# sphere instancing
coordinates = np.array(
    [[80 * np.cos(i / 4), 100 * np.sin(i / 4), 0] for i in range(1, 26)],
)
radii = np.array([4 + i / 4 for i in range(1, 26)])[:, np.newaxis].repeat(3, 1)
colors = np.array(
    [[0.8 * np.abs(np.cos(i / 8)), 0.8 * np.abs(np.cos((i + 3 * np.pi) / 8)), 0.8 * np.abs(np.cos((i + 6 * np.pi) / 8)), 0.8] for i in range(1, 26)],
)
my_renderer.add_points(coordinates, radii, colors)

# cylinder instancing
coordinates = np.array(
    [[
        [80 * np.cos(i / 4), 100 * np.sin(i / 4), 0],
        [80 * np.cos((i + 1) / 4), 100 * np.sin((i + 1) / 4), 0]
    ] for i in range(1, 26 - 1)],
)
radii = np.array([0.1 * (4 + i / 4) for i in range(1, 26 - 1)])
colors = np.array(
    [[0.8 * np.abs(np.cos(i / 8)), 0.8 * np.abs(np.cos((i + 3 * np.pi) / 8)), 0.8 * np.abs(np.cos((i + 6 * np.pi) / 8)), 0.8] for i in range(1, 26 - 1)],
)
my_renderer.add_lines(coordinates, radii, colors)


# view window
my_renderer.show()

wait = input("Press Enter to continue.")

# update view
my_renderer.change_view(camera_pos=(-400, 0, 0))

# view window
my_renderer.show()

wait = input("Press Enter to continue.")

# clear everything
my_renderer.clear_all()

# view window
my_renderer.show()

wait = input("Press Enter to continue.")
