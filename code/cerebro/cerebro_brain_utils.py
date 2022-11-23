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

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib


def load_GIFTI_surface(surface_file):
    # left ccortical surface
    surface = nib.load(surface_file)
    vertices = surface.darrays[0].data
    triangles = surface.darrays[1].data
    return vertices, triangles
