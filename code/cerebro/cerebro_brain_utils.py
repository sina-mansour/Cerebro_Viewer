"""
This module contains the utility code to handle neuroimaging data.

The goal is to put relevant functions to open and read various
neuroimaging files in this module.

Here are some capabilities that should be implemented for this module:
- Reading surface gifti files (.gii)
- Reading dscalar data to extract information


Notes
-----
Author: Sina Mansour L.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from skimage import measure
import trimesh as tm
from scipy import spatial
import open3d as o3d

# Utility template files and directories


cerebro_directory = os.path.abspath(os.path.dirname(__file__))
code_directory = os.path.dirname(cerebro_directory)
data_directory = os.path.join(code_directory, 'data')

cifti_template_file = os.path.join(data_directory, 'templates/HCP/dscalars/ones.dscalar.nii')


# Utility lists and dictionaries

volumetric_structure_inclusion_dict = {
    'CIFTI_STRUCTURE_CORTEX_LEFT': [],
    'CIFTI_STRUCTURE_CORTEX_RIGHT': [],
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_BRAIN_STEM': ['all'],
    'CIFTI_STRUCTURE_CAUDATE_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': ['all'],
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': ['all'],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_THALAMUS_LEFT': ['all', 'subcortex'],
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': ['all', 'subcortex'],
}

# coefficients of expansions for seperation of subcortical and cerebellar structures in cifti format
cifti_expansion_coeffs = {
    'CIFTI_STRUCTURE_ACCUMBENS_LEFT': (-0.15, 0.25, -0.5),
    'CIFTI_STRUCTURE_ACCUMBENS_RIGHT': (0.15, 0.25, -0.5),
    'CIFTI_STRUCTURE_AMYGDALA_LEFT': (-0.3, 0.25, -0.6),
    'CIFTI_STRUCTURE_AMYGDALA_RIGHT': (0.3, 0.25, -0.6),
    'CIFTI_STRUCTURE_BRAIN_STEM': (0, 0, -0.99),
    'CIFTI_STRUCTURE_CAUDATE_LEFT': (0, 0.35, 0.05),
    'CIFTI_STRUCTURE_CAUDATE_RIGHT': (0, 0.35, 0.05),
    'CIFTI_STRUCTURE_CEREBELLUM_LEFT': (-0.4, -0.3, -0.7),
    'CIFTI_STRUCTURE_CEREBELLUM_RIGHT': (0.4, -0.3, -0.7),
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT': (-0.1, 0, -0.4),
    'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT': (0.1, 0, -0.4),
    'CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT': (-0.25, 0, -0.55),
    'CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT': (0.25, 0, -0.55),
    'CIFTI_STRUCTURE_PALLIDUM_LEFT': (-0.2, 0.2, -0.25),
    'CIFTI_STRUCTURE_PALLIDUM_RIGHT': (0.2, 0.2, -0.25),
    'CIFTI_STRUCTURE_PUTAMEN_LEFT': (-0.45, 0.15, -0.45),
    'CIFTI_STRUCTURE_PUTAMEN_RIGHT': (0.45, 0.15, -0.45),
    'CIFTI_STRUCTURE_THALAMUS_LEFT': (-0.45, 0.2, 0.),
    'CIFTI_STRUCTURE_THALAMUS_RIGHT': (0.45, 0.2, 0.),
}


# Utility functions


def get_data_file(name):
    return os.path.join(data_directory, name)


def get_left_and_right_GIFTI_template_surface(template_surface):
    return (
        get_data_file(f'templates/HCP/surfaces/S1200.L.{template_surface}_MSMAll.32k_fs_LR.surf.gii'),
        get_data_file(f'templates/HCP/surfaces/S1200.R.{template_surface}_MSMAll.32k_fs_LR.surf.gii'),
    )


def load_GIFTI_surface(surface_file):
    # left ccortical surface
    surface = nib.load(surface_file)
    vertices = surface.darrays[0].data
    triangles = surface.darrays[1].data
    return vertices, triangles


def get_voxels_depth_mask(voxels_ijk, neighbor_rule='normal', peel_threshold=1, peel_neighbor_rule='normal', peel_depth=[0],):
    # store voxel information in proper data structures
    voxels_i, voxels_j, voxels_k = voxels_ijk[:, 0], voxels_ijk[:, 1], voxels_ijk[:, 2]
    voxels = set()
    voxel_indices = {}
    all_neighbors = []
    for idx in range(voxels_ijk.shape[0]):
        i = voxels_i[idx]
        j = voxels_j[idx]
        k = voxels_k[idx]
        voxels.add((i, j, k))
        voxel_indices[(i, j, k)] = idx
        neighbors = set()
        # strict neighbors: 26
        if neighbor_rule == 'strict':
            max_neighbors = 27
            for ni in [-1, 0, 1]:
                for nj in [-1, 0, 1]:
                    for nk in [-1, 0, 1]:
                        neighbors.add((i + ni, j + nj, k + nk))
        # normal neighbors: 6
        elif neighbor_rule == 'normal':
            max_neighbors = 7
            for ni in [-1, 0, 1]:
                neighbors.add((i + ni, j, k))
            for nj in [-1, 0, 1]:
                neighbors.add((i, j + nj, k))
            for nk in [-1, 0, 1]:
                neighbors.add((i, j, k + nk))
        all_neighbors.append(neighbors)
    # now compute depth O(n^3/2)
    depths = np.zeros(voxels_ijk.shape[0])
    current_depth = 0
    while len(voxels) > 0:
        removed_voxels = set()
        for voxel in voxels:
            idx = voxel_indices[voxel]
            # if not all_neighbors[idx].issubset(voxels):
            if len(all_neighbors[idx].intersection(voxels)) < (peel_threshold * max_neighbors):
                depths[idx] = current_depth
                i = voxels_i[idx]
                j = voxels_j[idx]
                k = voxels_k[idx]
                removed_voxels.add((i, j, k))
        voxels = voxels.difference(removed_voxels)
        current_depth += 1

    return np.isin(depths, peel_depth)


def generate_surface_marching_cube(voxels_ijk, transformation_matrix, smoothing=200, simplify=False):
    # approximate a surface representation with the marching cube algorithm
    I, J, K = np.meshgrid(
        *[range(x + 3) for x in voxels_ijk.max(0)],
        indexing='ij'
    )
    D = I * 0
    for i in range(voxels_ijk.shape[0]):
        D[voxels_ijk[i, 0] + 1, voxels_ijk[i, 1] + 1, voxels_ijk[i, 2] + 1] = 1
    verts_ijk, faces, normals, values = measure.marching_cubes(D, 0, allow_degenerate=False, gradient_direction='descent')
    verts_xyz = nib.affines.apply_affine(transformation_matrix, (verts_ijk - 1))
    tmesh = tm.Trimesh(vertices=verts_xyz, faces=faces)

    # smooth and remesh the generated marching cube surface
    if smoothing is not None:
        # tm.smoothing.filter_taubin(tmesh, iterations=smoothing,)
        new_vertices, new_faces = tm.remesh.subdivide(vertices=tmesh.vertices, faces=tmesh.faces)
        tmesh = tm.Trimesh(vertices=new_vertices, faces=new_faces)
        tm.smoothing.filter_taubin(tmesh, iterations=smoothing,)

    # reduce number of faces if needed
    max_face_count = (1 * faces.shape[0])
    if (simplify and (tmesh.faces.shape[0] > max_face_count)):
        tmesh = tmesh.simplify_quadratic_decimation(face_count=max_face_count)

    return tmesh.vertices, tmesh.faces


def get_nearest_neighbors(reference_coordinates, query_coordinates):
    # find nearest neighbors of every vertex
    kdtree = spatial.cKDTree(reference_coordinates)
    nearest_distances, nearest_indices = kdtree.query(query_coordinates)

    return (nearest_distances, nearest_indices)
