import matplotlib.pyplot as plt

from . import cerebro_brain_utils as cbu
from . import cerebro_brain_viewer as cbv

my_brain_viewer = cbv.Cerebro_brain_viewer(null_color=(0.7, 0.7, 0.7, 0.9))

### Test 1: surface cifti files

# surface = "pial"
# surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)

# # cifti_space = my_brain_viewer.visualize_cifti_space()
# cifti_space = my_brain_viewer.visualize_cifti_space(
#     volumetric_structures="all",
#     cifti_expansion_scale=20,
#     cifti_left_right_seperation=10,
#     volumetric_structure_offset=(0, 5, -25),
# )
# # cifti_space = my_brain_viewer.visualize_cifti_space(volumetric_structures='subcortex', volume_rendering='spheres_peeled')

# # wait = input("Press Enter to continue.")

# stat = "curvature"
# dscalar_file = cbu.get_data_file(
#     f"templates/HCP/dscalars/S1200.{stat}_MSMAll.32k_fs_LR.dscalar.nii"
# )
# dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(
#     dscalar_file=dscalar_file, colormap=plt.cm.Greys_r, opacity=1
# )

# # wait = input("Press Enter to continue.")

# dscalar_file = cbu.get_data_file(f"templates/HCP/dscalars/hcp.gradients.dscalar.nii")
# dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(
#     dscalar_file=dscalar_file, colormap=plt.cm.viridis, opacity=0.6
# )
# # dscalar_layer = my_brain_viewer.modify_cifti_dscalar_layer(dscalar_layer, dscalar_file=dscalar_file, colormap=plt.cm.Spectral, opacity=0.6)

### Test 2: surface from volumetric mask

nii_file = cbu.get_data_file(f"templates/MNI152/MNI152_T1_2mm_brain.nii.gz")
# nii_file = "/mnt/local_storage/Research/Datasets/MIITRA/MIITRA-T1w-05mm.nii.gz"

volumetric_mask = my_brain_viewer.visualize_mask_surface(nii_file, 6000, color=(0.9, 0.4, 0.5, 1), gradient_direction="ascent", smoothing=400)
# volumetric_mask = my_brain_viewer.visualize_mask_surface(
#     nii_file, 200, color=(0.7, 0.6, 0.4, 1), gradient_direction="ascent",
#     subdivide = False,
#     # smoothing=20, smoothing_filter='humphrey',
#     smoothing=20, smoothing_filter='laplacian',
#     # smoothing=200, smoothing_filter='taubin',
#     simplify=True, simplification_max_face_count=5e5,
# )

my_brain_viewer.show()

wait = input("Press Enter to continue.")
