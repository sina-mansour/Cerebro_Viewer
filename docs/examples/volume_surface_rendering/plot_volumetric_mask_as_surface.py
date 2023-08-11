"""
================
Rendering volumetric masks as surfaces
================

This is a quick demo showing how to plot a surface mesh generated from a volumetric mask.

Here, we use the template MNI152 T1-weighted brain to create an arbitrary mask. 


First, let's load the basic required libraries
"""
import matplotlib.pyplot as plt
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv

#%%
# Plotting a surface from volumetric masks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Define a viewer
cerebro_viewer = cbv.Cerebro_brain_viewer(offscreen=True, background_color=(1,1,1,1))

# Load an example volumetric file
nii_file = cbu.get_data_file(f"templates/MNI152/MNI152_T1_2mm_brain.nii.gz")

# Create a 3D surface of the mask
volumetric_mask = cerebro_viewer.visualize_mask_surface(
    nii_file,  # The volumetric file
    6000,  # An arbitrary threshold to mask the data
    color=(0.9, 0.4, 0.5, 1),  # Let's make the brain look pink!
    gradient_direction="ascent",  # This defines the surface normal directions
    smoothing=400  # Iteratively smooth the surface to look better
)


fig, ax = plt.subplots(figsize=(8,7))
plt.subplots_adjust(wspace=0, hspace=0)
ax.axis('off')
cerebro_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
cerebro_viewer.viewer.window.destroy()
plt.show()
