"""
This module contains the code to visualize brains and connectomes.

The goal is to have a class that can read in different neuroimaging file
formats and present them in a 3-dimensional space.

Here are some capabilities that should be implemented for this module:
- Rendering a 3D surface (such as .gii formats)
- Rendering a surface dscalar file
- Rendering a surface dscalar file with subcortical information
- Render connectomes (given a parcellation and connectivity matrix)
- Render high-resolution connectomes

Notes
-----
Author: Sina Mansour L.
"""

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from . import renderer
from . import cerebro_utils as utils
from . import cerebro_brain_utils as cbu


class Cerebro_brain_viewer():
    """
    Cerebero brain viewer engine

    This class contains the necessary logical units and input/output handlers
    required to visualize various brain imaging formats in the same viewer
    window.
    """

    def __init__(self,
                 background_color=(0.1, 0.1, 0.1, 0.0),
                 view='R', null_color=(0.7, 0.7, 0.7, 0.3), no_color=(0., 0., 0., 0.),
                 offscreen=False):
        # store intializations
        self.background_color = background_color
        self.view = view
        self.null_color = null_color
        self.no_color = no_color
        self.default_colormap = plt.cm.plasma
        self.offscreen = offscreen

        # initialize object boundaries
        self.min_coordinate = np.array([np.inf, np.inf, np.inf])
        self.max_coordinate = np.array([-np.inf, -np.inf, -np.inf])
        self.center_coordinate = np.array([0, 0, 0])

        # initialize render window
        self.renderer_type = 'panda3d'
        self.camera_config = self.view_to_camera_config(self.view)

        self.viewer = renderer.Renderer_panda3d(background_color=background_color, offscreen=offscreen, **self.camera_config)

        # Create a dictionary for created objects
        self.created_objects = {}

        # Create a dictionary for created layers
        self.created_layers = {}

        # Create a dictionary for loaded files
        self.loaded_files = {}

        # Create a dictionary for loaded default objects
        self.default_objects = {}

    # Camera view configuration
    def view_to_camera_config(self, view):
        camera_target = (0, 0, 0)
        camera_fov = 35
        camera_rotation = 0

        if ((view == 'R') or (view == 'Right')):
            camera_pos = (400, 0, 0)
        elif ((view == 'L') or (view == 'Left')):
            camera_pos = (-400, 0, 0)
        elif ((view == 'A') or (view == 'Anterior')):
            camera_pos = (0, 400, 0)
        elif ((view == 'P') or (view == 'Posterior')):
            camera_pos = (0, 400, 0)
        elif ((view == 'S') or (view == 'Superior')):
            camera_pos = (0, 0, 400)
            camera_rotation = -90
        elif ((view == 'I') or (view == 'Inferior')):
            camera_pos = (0, 0, -400)
            camera_rotation = 90
        else:
            # Alternatively the user could provide an arbitrary camera config instead of the view
            camera_pos = view[0]
            camera_target = view[1]
            camera_fov = view[2]
            camera_rotation = view[3]

        return {
            'camera_pos': camera_pos,
            'camera_target': camera_target,
            'camera_fov': camera_fov,
            'camera_rotation': camera_rotation,
        }

    def change_view(self, view):
        self.view = view
        self.camera_config = self.view_to_camera_config(self.view)
        self.viewer.change_view(**self.camera_config)
        # force recenter
        if isinstance(view, str):
            self.viewer.change_view(**self.view_to_camera_config((None, self.center_coordinate, None, None)))

    def load_GIFTI_cortical_surface_models(self, left_surface_file, right_surface_file):
        # Get a unique ID
        object_type = 'cortical_surface_model'
        object_id = f'{object_type}#{utils.generate_unique_id()}'
        # left ccortical surface
        left_vertices, left_triangles = self.load_file(left_surface_file, cbu.load_GIFTI_surface)
        # right cortical surface
        right_vertices, right_triangles = self.load_file(right_surface_file, cbu.load_GIFTI_surface)
        created_object = {
            'object_id': object_id,
            'object_type': object_type,
            'left_vertices': left_vertices,
            'left_triangles': left_triangles,
            'right_vertices': right_vertices,
            'right_triangles': right_triangles,
        }
        self.created_objects[object_id] = created_object

        # set as default surface model
        self.default_objects[object_type] = object_id

        # return object to user
        return created_object

    def load_template_GIFTI_cortical_surface_models(self, template_surface='inflated'):
        left_surface_file, right_surface_file = cbu.get_left_and_right_GIFTI_template_surface(template_surface)
        return self.load_GIFTI_cortical_surface_models(left_surface_file, right_surface_file)

    def load_file(self, file_name, load_func, use_cache=True):
        if use_cache and (file_name in self.loaded_files):
            return self.loaded_files[file_name]
        else:
            loaded_file = load_func(file_name)
            self.loaded_files[file_name] = loaded_file
            return loaded_file

    def create_surface_mesh_object(self, object_id, vertices, triangles, **kwargs):
        return {
            **{
                'object_id': object_id,
                'object_type': 'surface_mesh',
                'vertices': vertices,
                'triangles': triangles,
                'layers': {},
                'visibility': True,
                'render_update_required': True,
                'rendered': False,
            },
            **kwargs
        }

    def create_spheres_object(self, object_id, coordinates, radii, **kwargs):
        return {
            **{
                'object_id': object_id,
                'object_type': 'spheres',
                'coordinates': coordinates,
                'radii': radii,
                'layers': {},
                'visibility': True,
                'render_update_required': True,
                'rendered': False,
            },
            **kwargs
        }

    def visualize_cifti_space(self, cortical_surface_model_id=None, cifti_template_file=None, volumetric_structures='none', volume_rendering='surface', **kwargs):
        # initialization
        if cortical_surface_model_id is None:
            cortical_surface_model_id = self.default_objects['cortical_surface_model']
        if cifti_template_file is None:
            # use default cifti template
            cifti_template_file = cbu.cifti_template_file

        # load the template cifti
        cifti_template = self.load_file(cifti_template_file, nib.load)
        brain_models = [x for x in cifti_template.header.get_index_map(1).brain_models]
        brain_structures = [x.brain_structure for x in brain_models]

        # get appropriate IDs
        model_id = self.created_objects[cortical_surface_model_id]['object_id']
        unique_id = f'{utils.generate_unique_id()}'
        object_collection_id = f'cifti_space#{unique_id}'

        # store all visualized objects
        contained_object_ids = []

        # add the left cortical surface model
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_LEFT'
        brain_model = brain_models[brain_structures.index(brain_structure)]
        object_id = f'{brain_structure}#{unique_id}'
        contained_object_ids.append(object_id)
        self.created_objects[object_id] = self.create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]['left_vertices'],
            triangles=self.created_objects[cortical_surface_model_id]['left_triangles'],
            surface_model_id=model_id,
            data_vertex_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
        )

        # add the right cortical surface model
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        brain_model = brain_models[brain_structures.index(brain_structure)]
        object_id = f'{brain_structure}#{unique_id}'
        contained_object_ids.append(object_id)
        self.created_objects[object_id] = self.create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]['right_vertices'],
            triangles=self.created_objects[cortical_surface_model_id]['right_triangles'],
            surface_model_id=model_id,
            data_vertex_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
        )

        # add the subcortical structures
        transformation_matrix = cifti_template.header.get_index_map(1).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        for brain_structure in brain_structures:
            if volumetric_structures in cbu.volumetric_structure_inclusion_dict[brain_structure]:
                brain_model = brain_models[brain_structures.index(brain_structure)]
                object_id = f'{brain_structure}#{unique_id}'
                contained_object_ids.append(object_id)
                voxels_ijk = np.array(brain_model.voxel_indices_ijk)
                coordinates = nib.affines.apply_affine(transformation_matrix, voxels_ijk)
                voxel_size = nib.affines.voxel_sizes(transformation_matrix)
                radii = voxel_size[np.newaxis, :].repeat(coordinates.shape[0], 0) / 2
                if volume_rendering == 'spheres':
                    self.created_objects[object_id] = self.create_spheres_object(
                        object_id=object_id,
                        coordinates=coordinates,
                        radii=radii,
                        data_index_offset=brain_model.index_offset,
                        data_index_count=brain_model.index_count,
                        object_collection_id=object_collection_id,
                    )
                elif volume_rendering == 'spheres_peeled':
                    # apply peeling to get a thin layer from subcortical structures
                    selection_mask = cbu.get_voxels_depth_mask(voxels_ijk, **kwargs)
                    self.created_objects[object_id] = self.create_spheres_object(
                        object_id=object_id,
                        coordinates=coordinates[selection_mask],
                        radii=radii[selection_mask],
                        data_map=np.where(selection_mask),
                        data_index_offset=brain_model.index_offset,
                        data_index_count=brain_model.index_count,
                        object_collection_id=object_collection_id,
                    )
                elif volume_rendering == 'surface':
                    # use a marching cube algorithm with smoothing to generate a surface model
                    surface_vertices, surface_triangles = cbu.generate_surface_marching_cube(voxels_ijk, transformation_matrix, **kwargs)
                    nearest_distances, nearest_indices = cbu.get_nearest_neighbors(coordinates, surface_vertices)
                    self.created_objects[object_id] = self.create_surface_mesh_object(
                        object_id=object_id,
                        vertices=surface_vertices,
                        triangles=surface_triangles,
                        data_index_offset=brain_model.index_offset,
                        data_index_count=brain_model.index_count,
                        data_vertex_map=nearest_indices,
                        object_collection_id=object_collection_id,
                    )

        # create the cifti collection space
        collection_object = {
            'object_id': object_collection_id,
            'object_type': 'object_collection',
            'collection_type': 'cifti_space',
            'cifti_template': cifti_template,
            'contained_object_ids': contained_object_ids,
            'layers': {},
            'surface_model_id': model_id,
        }
        self.created_objects[object_collection_id] = collection_object

        # set as default cifti space
        self.default_objects['cifti_space'] = object_collection_id

        # draw to update visualization
        self.draw()

        # return object to user
        return collection_object

    def data_to_colors(self, data, colormap=None, clims=None, vlims=None, invert=False, opacity=1, exclusion_color=None, scale=None, dscalar_index=0):
        # initialization
        if colormap is None:
            colormap = self.default_colormap
        if exclusion_color is None:
            exclusion_color = self.no_color

        # create exclusion mask
        exclude = np.isinf(data) | np.isnan(data)
        # update exclusion criteria by vlims
        if vlims is not None:
            vlims_exclude = (data > vlims[1]) | (data < vlims[0])
            if invert:
                vlims_exclude = ~vlims_exclude
            exclude |= vlims_exclude

        # normalize data to range 0-1 (or use clims if provided)
        if clims is not None:
            cmin, cmax = clims
            normalized_data = (data - cmin)
            if cmin != cmax:
                normalized_data = normalized_data / (cmax - cmin)
        else:
            normalized_data = data - data[~exclude].min()
            if data[~exclude].min() != data[~exclude].max():
                normalized_data = normalized_data / (data[~exclude].max() - data[~exclude].min())

        # apply log-scale normalization if requested
        if scale == 'log':
            normalized_data = np.log2(1 + normalized_data)

        # exclude any invalid values created
        invalid_data = np.isinf(normalized_data) | np.isnan(normalized_data)
        # normalized_data[invalid_data] = normalized_data[~invalid_data].min()
        exclude |= invalid_data

        # load default null colors
        colors = np.array(exclusion_color)[np.newaxis, :].repeat(data.shape[0], 0)

        # produce colors
        colors[~exclude] = colormap(normalized_data[~exclude])

        # override opacity of valid colors
        colors[~exclude, 3] = opacity

        return colors

    def compute_overlay_colors(self, bottom_colors, top_colors):
        # convert to 0-1 range
        overlay_colors = bottom_colors * 0

        # rename for simplicity
        bottom_alpha = bottom_colors[:, 3].reshape(-1, 1)
        bottom_color = bottom_colors[:, :3]
        top_alpha = top_colors[:, 3].reshape(-1, 1)
        top_color = top_colors[:, :3]

        # compute opacity
        overlay_colors[:, 3] = (top_alpha + np.multiply(bottom_alpha, (1 - top_alpha))).reshape(-1)

        # compute color
        overlay_colors[:, :3] = np.multiply(top_color, top_alpha) + np.multiply(bottom_color, (1 - top_alpha))

        return overlay_colors

    def add_cifti_dscalar_layer(self, cifti_space_id=None, dscalar_file=None, loaded_dscalar=None, dscalar_data=None, dscalar_index=0, **kwargs):
        if cifti_space_id is None:
            # use default loaded cifti space
            cifti_space_id = self.default_objects['cifti_space']

        # initialization
        unique_id = f'{utils.generate_unique_id()}'
        layer_type = 'cifti_dscalar_layer'
        layer_id = f'{layer_type}#{unique_id}'

        # load the cifti dscalar file
        if dscalar_file is not None:
            dscalar = self.load_file(dscalar_file, nib.load)
            dscalar_data = dscalar.get_fdata()[dscalar_index]
        elif loaded_dscalar is not None:
            dscalar_data = loaded_dscalar.get_fdata()[dscalar_index]
        elif dscalar_data is None:
            raise Exception(f'No dscalar was provided for add_CIFTI_dscalar_layer.')

        # convert data to colors
        dscalar_colors = self.data_to_colors(dscalar_data, **kwargs)

        # load the cifti_template
        cifti_space = self.created_objects[cifti_space_id]
        layer_order = len(cifti_space['layers'])

        # store layer
        created_layer = {
            'layer_id': layer_id,
            'layer_type': layer_type,
            'layer_order': layer_order,
            'visibility': True,
            'dscalar_data': dscalar_data,
            'layer_colors': dscalar_colors,
            'cifti_space_id': cifti_space_id,
            'layer_update_required': True,
        }
        self.created_layers[layer_id] = created_layer
        cifti_space['layers'][layer_order] = layer_id

        # draw to update visualization
        self.draw()

        return created_layer

    def update_cifti_dscalar_layer(self, layer_id):
        cifti_space_id = self.created_layers[layer_id]['cifti_space_id']
        cifti_space = self.created_objects[cifti_space_id]
        layer_idx = len(cifti_space['layers'])
        cifti_space['layers'][layer_idx] = layer_id
        for object_id in cifti_space['contained_object_ids']:
            self.created_objects[object_id]['render_update_required'] = True
            self.created_objects[object_id]['layers'] = cifti_space['layers']

        # update flag
        self.created_layers[layer_id]['layer_update_required'] = False

    def update_layer(self, layer_id):
        if self.created_layers[layer_id]['layer_type'] == 'cifti_dscalar_layer':
            self.update_cifti_dscalar_layer(layer_id)

    def update_layers(self):
        for layer_id in self.created_layers:
            if self.created_layers[layer_id]['layer_update_required']:
                self.update_layer(layer_id)

    def render_surface_mesh(self, object_id):
        # load vertices and triangles
        surface_mesh_object = self.created_objects[object_id]
        surface_vertices = surface_mesh_object['vertices']
        surface_triangles = surface_mesh_object['triangles']

        # initial colors
        surface_colors = np.array(self.null_color)[np.newaxis, :].repeat(surface_vertices.shape[0], 0)

        # add layers one by one
        for layer_idx in range(len(surface_mesh_object['layers'])):
            layer_id = surface_mesh_object['layers'][layer_idx]
            layer_object = self.created_layers[layer_id]

            # extract colors from layer
            index_offset = surface_mesh_object['data_index_offset']
            index_count = surface_mesh_object['data_index_count']
            extracted_colors = layer_object['layer_colors'][index_offset: (index_offset + index_count)]

            # check colors have expected shape
            if (extracted_colors.shape[0] == index_count):
                if surface_mesh_object.get('data_vertex_map', None) is not None:
                    extracted_colors = extracted_colors[surface_mesh_object['data_vertex_map']]

                # the extracted colors need to be mapped to the surface
                layer_surface_colors = np.array(self.no_color)[np.newaxis, :].repeat(surface_vertices.shape[0], 0)
                if surface_mesh_object.get('data_vertex_indices', None) is not None:
                    layer_surface_colors[surface_mesh_object['data_vertex_indices']] = extracted_colors
                else:
                    layer_surface_colors = extracted_colors

                # compute the layer overlay
                surface_colors = self.compute_overlay_colors(surface_colors, layer_surface_colors)

        # clear existing render
        if surface_mesh_object['rendered']:
            self.viewer.clear_object(surface_mesh_object['rendered_mesh']['node_name'])
            surface_mesh_object.pop('rendered_mesh')
            surface_mesh_object['rendered'] = False

        # render the object
        surface_mesh_object['vertex_colors'] = surface_colors
        rendered_mesh = self.viewer.add_mesh(surface_vertices, surface_triangles, surface_colors)
        surface_mesh_object['rendered_mesh'] = rendered_mesh
        surface_mesh_object['rendered'] = True

        # update object boundaries
        self.min_coordinate = np.min([self.min_coordinate, surface_vertices.min(0)], 0)
        self.max_coordinate = np.max([self.max_coordinate, surface_vertices.max(0)], 0)

        # signal that render was updated
        self.created_objects[object_id]['render_update_required'] = False

    def render_spheres(self, object_id):
        # load vertices and triangles
        spheres_object = self.created_objects[object_id]
        coordinates = spheres_object['coordinates']
        radii = spheres_object['radii']

        # initial colors
        colors = np.array(self.null_color)[np.newaxis, :].repeat(coordinates.shape[0], 0)

        # add layers one by one
        for layer_idx in range(len(spheres_object['layers'])):
            layer_id = spheres_object['layers'][layer_idx]
            layer_object = self.created_layers[layer_id]

            # extract colors from layer
            index_offset = spheres_object['data_index_offset']
            index_count = spheres_object['data_index_count']
            extracted_colors = layer_object['layer_colors'][index_offset: (index_offset + index_count)]

            # check colors have expected shape
            if (extracted_colors.shape[0] == index_count):
                if spheres_object.get('data_map', None) is not None:
                    extracted_colors = extracted_colors[spheres_object['data_map']]

                # the extracted colors need to be mapped to the spheres
                layer_colors = np.array(self.no_color)[np.newaxis, :].repeat(coordinates.shape[0], 0)
                if spheres_object.get('data_indices', None) is not None:
                    layer_colors[spheres_object['data_indices']] = extracted_colors
                else:
                    layer_colors = extracted_colors

                # compute the layer overlay
                colors = self.compute_overlay_colors(colors, layer_colors)

        # clear existing render

        # render the object
        spheres_object['colors'] = colors
        rendered_spheres = self.viewer.add_points(coordinates, radii, colors)
        spheres_object['rendered_spheres'] = rendered_spheres
        spheres_object['rendered'] = True

        # update object boundaries
        self.min_coordinate = np.min([self.min_coordinate, coordinates.min(0)], 0)
        self.max_coordinate = np.max([self.max_coordinate, coordinates.max(0)], 0)

        # signal that render was updated
        self.created_objects[object_id]['render_update_required'] = False

    def render_object(self, object_id):
        if self.created_objects[object_id]['object_type'] == 'surface_mesh':
            self.render_surface_mesh(object_id)
        elif self.created_objects[object_id]['object_type'] == 'spheres':
            self.render_spheres(object_id)

    def center_camera(self):
        new_center_coordinate = (self.min_coordinate + self.max_coordinate) / 2
        if (self.center_coordinate != new_center_coordinate).any():
            self.center_coordinate = new_center_coordinate
            self.change_view((None, self.center_coordinate, None, None))

    def render_update(self):
        for object_id in self.created_objects:
            if self.created_objects[object_id].get('render_update_required', False):
                self.render_object(object_id)
        self.center_camera()
        utils.garbage_collect()

    def draw(self):
        # update any required renders
        self.update_layers()
        self.render_update()

        # run the viewer window
        self.viewer.draw()

    def show(self):
        # update any required renders
        self.update_layers()
        self.render_update()

        # run the viewer window
        self.viewer.show()

    def offscreen_draw_to_matplotlib_axes(self, ax):
        """Draw an offscreen-rendered view to a matplotlib axes.

        Note: this functionality is experimental and might not fully work depending on
        viewer configuration.

        Args:
            ax (matplotlib.Axes): the axes into which the view will be drawn.
        """
        self.viewer.window.offscreen_draw_to_matplotlib_axes(ax)
