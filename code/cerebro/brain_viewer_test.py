import matplotlib.pyplot as plt

from . import cerebro_brain_utils as cbu
from . import cerebro_brain_viewer as cbv

my_brain_viewer = cbv.Cerebro_brain_viewer()

surface = 'pial'
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)

# cifti_space = my_brain_viewer.visualize_cifti_space()
cifti_space = my_brain_viewer.visualize_cifti_space(volumetric_structures='all', cifti_expansion_scale=20, cifti_left_right_seperation=10, volumetric_structure_offset=(0, 5, -25))
# cifti_space = my_brain_viewer.visualize_cifti_space(volumetric_structures='subcortex', volume_rendering='spheres_peeled')

# wait = input("Press Enter to continue.")

stat = 'curvature'
dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/S1200.{stat}_MSMAll.32k_fs_LR.dscalar.nii')
dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(dscalar_file=dscalar_file, colormap=plt.cm.Greys_r, opacity=1)

# wait = input("Press Enter to continue.")

dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/hcp.gradients.dscalar.nii')
dscalar_layer = my_brain_viewer.modify_cifti_dscalar_layer(dscalar_layer, dscalar_file=dscalar_file, colormap=plt.cm.Spectral, opacity=0.6)


my_brain_viewer.show()

wait = input("Press Enter to continue.")
