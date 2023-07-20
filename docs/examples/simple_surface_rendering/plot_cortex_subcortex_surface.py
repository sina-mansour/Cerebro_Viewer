"""
================
Plot a simple static surface mesh rendering
================

This is a quick demo for how to plot a surface mesh rendering for just the cortex.

Here, we use the template GIFTI data, but you can also use your own GIFTI data. 
"""

my_brain_viewer = cbv.Cerebro_brain_viewer(offscreen=True)
surface = "pial"
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)

# cifti_space = my_brain_viewer.visualize_cifti_space()
cifti_space = my_brain_viewer.visualize_cifti_space(
    volumetric_structures="all", # Change to "none" for just cortex, or "subcortex" for just subcortex
    cifti_expansion_scale=20,
    cifti_left_right_seperation=10,
    volumetric_structure_offset=(0, 5, -25),
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)
