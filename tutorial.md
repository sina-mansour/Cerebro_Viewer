# Tutorial

```
import matplotlib.pyplot as plt
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv
```

Initialise brain viewer object. From this point onwards, use my_brain_viewer.show() to visualise the object.
```
my_brain_viewer = cbv.Cerebro_brain_viewer()
```

Alternatively, specify background colours and view angle
```
my_brain_viewer = cbv.Cerebro_brain_viewer(background_color=(0.1, 0.1, 0.1, 0.0),view='R', null_color=(0.7, 0.7, 0.7, 0.3), no_color=(0., 0., 0., 0.))
```

Add a surface mesh specified as a .gii filepath. 'leftsurface.gii' is an example here.
```
surface_model = my_brain_viewer.load_GIFTI_cortical_surface_models('leftsurface.gii','rightsurface.gii')
```

Alternatively, use one of the template surfaces. Options flat, pial, midthickness, inflated, very_inflated, sphere
```
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models('pial') 
```

Add cifti space. By default, no volumetric structures are included
```
cifti_space = my_brain_viewer.visualize_cifti_space()
```

Optionally, visualise volumetric structures. 
volumetric_structures options are 'none' (default), 'subcortex' or 'all'. volume_rendering options are 'surface' (default), 'spheres','spheres_peeled'
```
cifti_space = my_brain_viewer.visualize_cifti_space(volumetric_structures='subcortex',volume_rendering='spheres')
```

Add a scalar surface map specified as a dscalar.nii filepath. 'file.nii' is an example.
```
dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(dscalar_file='file.nii', colormap=plt.cm.Greys_r, opacity=1)
```

Alternatively, use one of the example dscalars actually provided
```
dscalar_file = cbu.get_data_file(f'templates/HCP/dscalars/S1200.curvature_MSMAll.32k_fs_LR.dscalar.nii')
dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(dscalar_file=dscalar_file, colormap=plt.cm.Greys_r, opacity=1)
```

You can superimpose more layers with further calls of add_cifti_dscalar_layer

Change the view
```
my_brain_viewer.change_view('L')
```

Add some spherical ROIs
```
node_coordinates = [[0,0,0],[25,0,50],[25,25,25]]
my_brain_viewer.visualize_spheres(node_coordinates, radii=3, color=[0,1,0,0.5])
```

Add a ball and stick network with same ROIs as above
```
node_coordinates = [[0,0,0],[25,0,50],[25,25,25]]
adjacency = [[0,1,1],[1,0,1],[1,1,0]]
my_brain_viewer.visualize_network(adjacency,node_coordinates,node_radii=5,edge_radii=1,node_color=[1,0,0,0.5],edge_color=[0,0,1,0.5])
```


```
my_brain_viewer.show()
```

