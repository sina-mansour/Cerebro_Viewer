# Surface Rendering

To render a surface

```{python}
import matplotlib.pyplot as plt

from cerebro import (
    cerebro_brain_utils as cbu,
    cerebro_brain_viewer as cbv,
)

my_brain_viewer = cbv.Cerebro_brain_viewer(
    offscreen=True,
    background_color=(1,1,1,0),
    null_color=(0.9, 0.9, 0.9, 0.5),
    no_color=(.6, .6, .6, 0.1)
)

# render a surface
surface = 'pial'
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)

cifti_space = my_brain_viewer.visualize_cifti_space()

# render data over surface
dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/hcp.gradients.dscalar.nii')
dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(
    dscalar_file=dscalar_file, colormap=plt.cm.Greys
)

fig, ax = plt.subplots(figsize=(10,10))
ax.axis('off')
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)
```
