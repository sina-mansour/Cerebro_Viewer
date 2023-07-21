"""
================
Plot in interactive window and add scalar layer
================
This demo showcases 2 features:
1) Plot in interactive window instead of on matplotlib axis, so that you can zoom/pan/rotate using the mouse.
2) Add a dscalar.nii surface map, as a colored layer
"""
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv
import matplotlib.pyplot as plt

my_brain_viewer = cbv.Cerebro_brain_viewer(offscreen=False)
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models('pial')
cifti_space = my_brain_viewer.visualize_cifti_space()

dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii')
my_brain_viewer.add_cifti_dscalar_layer(dscalar_file=dscalar_file, colormap=plt.cm.inferno, opacity=1)

my_brain_viewer.show()