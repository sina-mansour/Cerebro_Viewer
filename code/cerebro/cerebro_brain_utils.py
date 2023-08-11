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

from __future__ import annotations

from itertools import product
import os
import numpy as np
from numpy.typing import NDArray
import nibabel as nib
from skimage import measure
import trimesh as tm
from scipy import spatial

# Type hint imports
from typing import Callable
from .cerebro_types import Voxel

# suppress trivial nibabel warnings, see https://github.com/nipy/nibabel/issues/771
nib.imageglobals.logger.setLevel(40)

# Utility template files and directories


cerebro_directory = os.path.abspath(os.path.dirname(__file__))
code_directory = os.path.dirname(cerebro_directory)
DATA_DIRECTORY = os.path.join(code_directory, "data")

cifti_template_file = os.path.join(
    DATA_DIRECTORY, "templates/HCP/dscalars/ones.dscalar.nii"
)


# Utility lists and dictionaries

volumetric_structure_inclusion_dict = {
    "CIFTI_STRUCTURE_CORTEX_LEFT": [],
    "CIFTI_STRUCTURE_CORTEX_RIGHT": [],
    "CIFTI_STRUCTURE_ACCUMBENS_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_AMYGDALA_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_AMYGDALA_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_BRAIN_STEM": ["all"],
    "CIFTI_STRUCTURE_CAUDATE_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_CAUDATE_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_CEREBELLUM_LEFT": ["all"],
    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": ["all"],
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_PALLIDUM_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_PALLIDUM_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_PUTAMEN_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_PUTAMEN_RIGHT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_THALAMUS_LEFT": ["all", "subcortex"],
    "CIFTI_STRUCTURE_THALAMUS_RIGHT": ["all", "subcortex"],
}

# coefficients of expansions for seperation of subcortical and cerebellar structures in
# cifti format
cifti_expansion_coeffs = {
    "CIFTI_STRUCTURE_ACCUMBENS_LEFT": (-0.15, 0.25, -0.5),
    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": (0.15, 0.25, -0.5),
    "CIFTI_STRUCTURE_AMYGDALA_LEFT": (-0.3, 0.25, -0.6),
    "CIFTI_STRUCTURE_AMYGDALA_RIGHT": (0.3, 0.25, -0.6),
    "CIFTI_STRUCTURE_BRAIN_STEM": (0, 0, -0.99),
    "CIFTI_STRUCTURE_CAUDATE_LEFT": (0, 0.35, 0.05),
    "CIFTI_STRUCTURE_CAUDATE_RIGHT": (0, 0.35, 0.05),
    "CIFTI_STRUCTURE_CEREBELLUM_LEFT": (-0.4, -0.3, -0.7),
    "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": (0.4, -0.3, -0.7),
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT": (-0.1, 0, -0.4),
    "CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT": (0.1, 0, -0.4),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT": (-0.25, 0, -0.55),
    "CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT": (0.25, 0, -0.55),
    "CIFTI_STRUCTURE_PALLIDUM_LEFT": (-0.2, 0.2, -0.25),
    "CIFTI_STRUCTURE_PALLIDUM_RIGHT": (0.2, 0.2, -0.25),
    "CIFTI_STRUCTURE_PUTAMEN_LEFT": (-0.45, 0.15, -0.45),
    "CIFTI_STRUCTURE_PUTAMEN_RIGHT": (0.45, 0.15, -0.45),
    "CIFTI_STRUCTURE_THALAMUS_LEFT": (-0.45, 0.2, 0.0),
    "CIFTI_STRUCTURE_THALAMUS_RIGHT": (0.45, 0.2, 0.0),
}


# Utility classes for neuroimaging data

class File_handler:
    """File handler

    This class contains logical units used to handle file I/O operations. The
    file handler is intentionally made as a Singleton for efficient caching
    and avoiding duplicates.
    """
    _instance = None

    def __new__(cls):
        # Only create a new instance if its the first time
        if cls._instance is None:
            cls._instance = super(File_handler, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # A data structure to keep loaded files for caching
        self.loaded_files = {}

    def load_file(
        self,
        file_name: str,
        load_func: Callable[[str], object],
        use_cache: bool = True
    ):
        """Load a file using a specified loading function.

        This function loads a file using the provided loading function. It checks if the file has already been loaded and
        returns the cached version if 'use_cache' is set to True. Otherwise, it loads the file using the loading function
        and caches it for future use.

        Args:
            file_name (str): The name or path of the file to be loaded.
            load_func (function): The loading function to be used for loading the file.
            use_cache (bool, optional): Whether to use the cached version of the file if available. Defaults to True.

        Returns:
            Any: The loaded file data returned by the loading function.

        Example:
            data = my_brain_viewer._load_file(file_to_load, my_loading_function)
        """
        # Convert path to absolute
        file_name = os.path.abspath(file_name)

        # Create a unique identifier/key
        file_key = (file_name, load_func.__module__, load_func.__name__)

        # Check for cached file
        if use_cache and (file_key in self.loaded_files):
            return self.loaded_files[file_key]
        # Otherwise load and cache the file
        else:
            loaded_file = load_func(file_name)
            self.loaded_files[file_key] = loaded_file
            return loaded_file

class Volumetric_data:
    """Volumetric data

    This class contains the necessary I/O handlers and logical units to load
    volumetric brain imaging data.

    Parameters
    ----------
    data
        The input file or loaded image.

    Attributes
    ----------
    affine
        The affine transform to convert voxels indices to coordinates.
    data
        An array containing the image data.
    ndim
        The number of dimensions of the image (3 for 3-dimensional data)
    """
    def __init__(self, data: str | nib.Nifti1Image | Volumetric_data):
        # Check if data requires loading
        if (type(data) == str):
            # Loading a NIfTI file
            if ((data[-4:] == ".nii") or (data[-7:] == ".nii.gz")):
                self.loaded_obj = File_handler().load_file(data, nib.load)
            else:
                raise ValueError(f"File type for '{data}' is not supported")
        else:
            # Store the pre-loaded data
            self.loaded_obj = data

        # Now that the data is loaded, create the required objects
        if (type(self.loaded_obj) == nib.Nifti1Image):
            # Convert to RAS orientation
            self.loaded_obj = nib.as_closest_canonical(self.loaded_obj)

            # Store affine, data, and ndim
            self.affine = self.loaded_obj.affine
            self.data = self.loaded_obj.get_fdata()
            self.ndim = self.loaded_obj.ndim
        # Create a copy of another Volumetric_data object
        elif (type(self.loaded_obj) == Volumetric_data):
            # Copy affine, data, and ndim
            self.affine = self.loaded_obj.affine
            self.data = self.loaded_obj.data
            self.ndim = self.loaded_obj.ndim

        # Destroy link to the loaded object
        self.loaded_obj = None

    def mask(self, threshold: float):
        """Convert the data to a binary mask."""
        self.data = self.data > threshold
        return self


# Utility functions


def get_data_file(name: str) -> str:
    """Construct the path to a data file from Cerebro's internal data directory."""
    return os.path.join(DATA_DIRECTORY, name)


def get_left_and_right_GIFTI_template_surface(template_surface: str) -> tuple[str, str]:
    """Return the paths to the left and right GIFTI template surfaces."""
    return (
        get_data_file(
            "templates/HCP/surfaces/"
            f"S1200.L.{template_surface}_MSMAll.32k_fs_LR.surf.gii"
        ),
        get_data_file(
            "templates/HCP/surfaces/"
            f"S1200.R.{template_surface}_MSMAll.32k_fs_LR.surf.gii"
        ),
    )


def load_GIFTI_surface(surface_file: str) -> tuple[NDArray, NDArray]:
    """Read the vertices and triangles representing a GIfTI surface."""
    # left cortical surface
    surface = nib.load(surface_file)
    vertices = surface.darrays[0].data
    triangles = surface.darrays[1].data
    return vertices, triangles


def get_neighbors_normal(voxel: Voxel) -> set[Voxel]:
    """Return a set containing a voxel's 6 "normal" neighbors plus the voxel itself."""
    i, j, k = voxel[0], voxel[1], voxel[2]
    neighbors = set()
    for offset in [-1, 0, 1]:
        neighbors.add((i + offset, j, k))
        neighbors.add((i, j + offset, k))
        neighbors.add((i, j, k + offset))
    return neighbors


def get_neighbors_strict(voxel: Voxel) -> set[Voxel]:
    """Return a set containing a voxel's 26 "strict" neighbors plus the voxel itself."""
    i, j, k = voxel[0], voxel[1], voxel[2]
    neighbors = {
        (i + offset_i, j + offset_j, k + offset_k)
        for offset_i, offset_j, offset_k in product([-1, 0, 1], repeat=3)
    }
    return neighbors


def get_voxels_depth_mask(
    voxels_ijk: NDArray,
    neighbor_rule: str = "normal",
    peel_threshold: float = 1,
    peel_depth: list[int] = [0],
):
    """Peel a volumetric structure to reveal voxels at a given depth.

    Given the voxels corresponding to a volumetric structure, return only those that
    are at the given peel depth(s).

    Parameters
    ----------
    voxels_ijk
        n * 3 array representing the voxels that compose the volumetric structure.
    neighbor_rule
        Either "strict" or "normal", describing what's considered a neighbor.
    peel_threshold
        The proportion of the total possible neighbours that need to be in the structure
        for a voxel to be considered below the outside layer. Should be 1 or less.
    peel_depth
        The depths (layer indices) to keep in the output mask.
    """
    # store voxel information in proper data structures
    voxels_i, voxels_j, voxels_k = voxels_ijk[:, 0], voxels_ijk[:, 1], voxels_ijk[:, 2]
    voxels: set[Voxel] = set()
    voxel_indices: dict[Voxel, int] = {}
    all_neighbors: list[set[Voxel]] = []

    # Generate a list of every neighbour of every voxel in voxels_ijk
    for idx, voxel in enumerate(zip(voxels_i, voxels_j, voxels_k)):
        voxels.add(voxel)
        voxel_indices[voxel] = idx
        # strict neighbors: 26
        if neighbor_rule == "strict":
            max_neighbors = 27
            all_neighbors.append(get_neighbors_strict(voxel))
        # normal neighbors: 6
        elif neighbor_rule == "normal":
            max_neighbors = 7
            all_neighbors.append(get_neighbors_normal(voxel))

    # now compute depth O(n^3/2)
    depths = np.zeros(voxels_ijk.shape[0])
    current_depth = 0
    while voxels:
        removed_voxels = set()
        for voxel in voxels:
            idx = voxel_indices[voxel]

            # If few enough of this voxel's neighbors are within the structure, we
            # assume that it's on the current outside layer
            if len(all_neighbors[idx].intersection(voxels)) < (
                peel_threshold * max_neighbors
            ):
                depths[idx] = current_depth
                removed_voxels.add(voxel)
        # Peel off the voxels we just identified as being on the current outside layer
        voxels = voxels.difference(removed_voxels)
        current_depth += 1

    return np.isin(depths, peel_depth)


def generate_surface_marching_cube(
    voxels_ijk: NDArray,
    transformation_matrix: NDArray,
    smoothing: int | None = 200,
    smoothing_filter: str = "taubin",
    subdivide: bool = True,
    simplify: bool = False,
    simplification_max_face_count: int = None,
    gradient_direction = "descent"
):
    """Approximate a surface mesh representation of a volumetric structure.

    This uses the marching cube algorithm.

    Parameters
    ----------
    voxels_ijk
        Voxels composing the volumetric structure.
    transformation_matrix
        Matrix representing an affine transformation to apply to the generated vertices.
    smoothing
        Iterations of the smoothing algorithm to run, or None to skip smoothing.
    smoothing_filter
        Choice of smoothing algorithm ("taubin", "laplacian").
    subdivide
        Whether the mesh should be subdivided. This increases the quality of low-resolution
        masks, but is better left off in higher resolution files.
    simplify
        If true, simplify the generated mesh with quadratic decimation.
    simplification_max_face_count
        The maximum number of faces used in the simplification.
    gradient_direction
        Determines the definition of outside boundaries for the marching cube. This can
        be either "ascent" or "descent", may need manual adjustment.
    """
    I, J, K = np.meshgrid(*[range(x + 3) for x in voxels_ijk.max(0)], indexing="ij")
    D = I * 0
    for i in range(voxels_ijk.shape[0]):
        D[voxels_ijk[i, 0] + 1, voxels_ijk[i, 1] + 1, voxels_ijk[i, 2] + 1] = 1
    verts_ijk, faces, normals, values = measure.marching_cubes(
        D, 0, allow_degenerate=False, gradient_direction=gradient_direction
    )
    verts_xyz = nib.affines.apply_affine(transformation_matrix, (verts_ijk - 1))
    tmesh = tm.Trimesh(vertices=verts_xyz, faces=faces)

    # smooth and remesh the generated marching cube surface
    if subdivide:
        new_vertices, new_faces = tm.remesh.subdivide(
            vertices=tmesh.vertices, faces=tmesh.faces
        )
        tmesh = tm.Trimesh(vertices=new_vertices, faces=new_faces)
    if smoothing:
        if smoothing_filter == "taubin":
            tm.smoothing.filter_taubin(
                tmesh,
                iterations=smoothing,
            )
        if smoothing_filter == "laplacian":
            tm.smoothing.filter_laplacian(
                tmesh,
                iterations=smoothing,
            )
        if smoothing_filter == "humphrey":
            tm.smoothing.filter_humphrey(
                tmesh,
                iterations=smoothing,
            )

    # reduce number of faces if needed
    if simplification_max_face_count is None:
        simplification_max_face_count = 1 * faces.shape[0]
    if simplify and (tmesh.faces.shape[0] > simplification_max_face_count):
        tmesh = tmesh.simplify_quadratic_decimation(face_count=simplification_max_face_count)

    return tmesh.vertices, tmesh.faces


def get_nearest_neighbors(
    reference_coordinates: NDArray, query_coordinates: NDArray
) -> tuple[NDArray, NDArray]:
    """Find the nearest neighbors of every vertex."""
    kdtree = spatial.cKDTree(reference_coordinates)
    nearest_distances, nearest_indices = kdtree.query(query_coordinates)

    return (nearest_distances, nearest_indices)
