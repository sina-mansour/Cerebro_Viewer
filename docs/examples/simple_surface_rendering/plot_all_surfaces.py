"""
================
Cortex, subcortex, cerebellum, and brainstem surface rendering
================

This is a quick demo for how to plot a surface mesh rendering for the cortex, subcortex, cerebellum, and brainstem.

Here, we use the template GIFTI data, but you can also use your own GIFTI data. 

You can plot just the cortex (first image below), just the cortex and subcortex (second image below), or the cortex, subcortex, cerebellum, and brainstem (third image below).
"""
import matplotlib.pyplot as plt
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv

#%%
# Plotting cortex surface only
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Define a viewer
cortex_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface_model = cortex_viewer.load_template_GIFTI_cortical_surface_models(template_surface="pial")

cifti_space = cortex_viewer.visualize_cifti_space(
    volumetric_structures="none" # Change to "none" for just cortex, or "subcortex" for just cortex and subcortex
)

fig, ax = plt.subplots(figsize=(8,7))
plt.subplots_adjust(wspace=0, hspace=0)
ax.axis('off')
cortex_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
cortex_viewer.viewer.window.destroy()
plt.show()

#%%
# Cortex and subcortex
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You also have the option to plot the cortex and subcortex together.

# Define a viewer
cortex_subcortex_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface_model = cortex_subcortex_viewer.load_template_GIFTI_cortical_surface_models(template_surface="pial")

cifti_space = cortex_subcortex_viewer.visualize_cifti_space(
    volumetric_structures="subcortex" # Change to "none" for just cortex, or "all" for cortex, subcortex, cerebellum, and brainstem
)

fig, ax = plt.subplots(figsize=(8,7))
plt.subplots_adjust(wspace=0, hspace=0)
ax.axis('off')
cortex_subcortex_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
cortex_subcortex_viewer.viewer.window.destroy()
plt.show()

#%%
# Cortex, subcortex, brainstem, and cerebellum
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Lastly, you can show all of the brain structures together -- cortex, subcortex, brainstem, and cerebellum.

# Define a viewer
all_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface_model = all_viewer.load_template_GIFTI_cortical_surface_models(template_surface="pial")

cifti_space = all_viewer.visualize_cifti_space(
    volumetric_structures="all" # Change to "none" for just cortex, or "subcortex" for just cortex and subcortex
)

fig, ax = plt.subplots(figsize=(8,7))
plt.subplots_adjust(wspace=0, hspace=0)
ax.axis('off')
all_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
all_viewer.viewer.window.destroy()
plt.show()

#%%
# Using your own surface data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If you have your own surface data in GIFTI format, here's how you can plot it -- with this example showing a cortical flatmap:

# Define a viewer
your_surface_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))

# Define your left and right surface GIFTI files
from pathlib import Path
try:
    main_path = Path(__file__).parents[3]
except NameError:
    import sys
    main_path = Path(sys.argv[0]).parents[3]
left_surface_file = f"{main_path}/code/data/templates/HCP/surfaces/S1200.L.inflated_MSMAll.32k_fs_LR.surf.gii"
right_surface_file = f"{main_path}/code/data/templates/HCP/surfaces/S1200.R.inflated_MSMAll.32k_fs_LR.surf.gii"

surface_model = your_surface_viewer.load_GIFTI_cortical_surface_models(
    left_surface_file=left_surface_file,
    right_surface_file=right_surface_file
)

cifti_space = your_surface_viewer.visualize_cifti_space(
    volumetric_structures="subcortex"
)

fig, ax = plt.subplots(figsize=(8,7))
plt.subplots_adjust(wspace=0, hspace=0)
ax.axis('off')
your_surface_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
your_surface_viewer.viewer.window.destroy()
plt.show()