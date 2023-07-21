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

"""
Cortex only
================
Plotting just the cortex.
"""

# Cortex only
cortex_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface = "pial"
surface_model = cortex_viewer.load_template_GIFTI_cortical_surface_models(surface)

cifti_space = cortex_viewer.visualize_cifti_space(
    volumetric_structures="none" # Change to "none" for just cortex, or "subcortex" for just cortex and subcortex
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
cortex_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
cortex_viewer.viewer.window.destroy()
plt.show()

#%%

"""
Cortex and subcortex
================
Plotting just the cortex and subcortex.
"""

cortex_subcortex_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface = "pial"
surface_model = cortex_subcortex_viewer.load_template_GIFTI_cortical_surface_models(surface)

cifti_space = cortex_subcortex_viewer.visualize_cifti_space(
    volumetric_structures="subcortex" # Change to "none" for just cortex, or "all" for cortex, subcortex, cerebellum, and brainstem
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
cortex_subcortex_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
cortex_subcortex_viewer.viewer.window.destroy()
plt.show()

#%%

"""
Cortex, subcortex, brainstem, and cerebellum
================
Plotting just the cortex.
"""

all_viewer = cbv.Cerebro_brain_viewer(offscreen=True,background_color=(255,255,255,1))
surface = "pial"
surface_model = all_viewer.load_template_GIFTI_cortical_surface_models(surface)

cifti_space = all_viewer.visualize_cifti_space(
    volumetric_structures="all" # Change to "none" for just cortex, or "subcortex" for just cortex and subcortex
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
all_viewer.offscreen_draw_to_matplotlib_axes(ax)

# Clear this viewer
all_viewer.viewer.window.destroy()
plt.show()