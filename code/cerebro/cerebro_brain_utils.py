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


# Utility template files and directories


cerebro_directory = os.path.abspath(os.path.dirname(__file__))
code_directory = os.path.dirname(cerebro_directory)
data_directory = os.path.join(code_directory, 'data')

cifti_template_file = os.path.join(data_directory, 'templates/HCP/dscalars/ones.dscalar.nii')


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
