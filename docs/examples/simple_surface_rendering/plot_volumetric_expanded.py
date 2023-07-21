"""
================
How to move volumetric structures around
================
Function visualize_cifti_space has optional parameters to (1) expand all structures outwards, (2) separate left and right hemispheres, and (3) add an offset to all volumetric structures relative to cortical surface.
Null_color alpha value was set to 1 to make brain surface opaque.
"""
import matplotlib.pyplot as plt
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv

my_brain_viewer = cbv.Cerebro_brain_viewer(offscreen=True,null_color=(0.7,0.7,0.7,1))
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models('pial')

cifti_space = my_brain_viewer.visualize_cifti_space(
    volumetric_structures="all", 
    cifti_expansion_scale=40,
    cifti_left_right_seperation=20,
    volumetric_structure_offset=(0, 10, -80),
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
my_brain_viewer.viewer.window.destroy()
plt.show()