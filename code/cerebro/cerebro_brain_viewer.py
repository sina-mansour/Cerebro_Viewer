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
                 offscreen=False,):
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

    def __del__(self):
        del self.viewer

    # Camera view configuration
    def view_to_camera_config(self, view):
        if isinstance(view, str):
            self.camera_target = self.center_coordinate
            self.camera_fov = 25
            self.camera_rotation = 0

        if ((view == 'R') or (view == 'Right')):
            self.camera_pos = (400, 0, 0)
        elif ((view == 'L') or (view == 'Left')):
            self.camera_pos = (-400, 0, 0)
        elif ((view == 'A') or (view == 'Anterior')):
            self.camera_pos = (0, 400, 0)
        elif ((view == 'P') or (view == 'Posterior')):
            self.camera_pos = (0, 400, 0)
        elif ((view == 'S') or (view == 'Superior')):
            self.camera_pos = (0, 0, 400)
            self.camera_rotation = -90
        elif ((view == 'I') or (view == 'Inferior')):
            self.camera_pos = (0, 0, -400)
            self.camera_rotation = 90
        else:
            # Alternatively the user could provide an arbitrary camera config instead of the view
            if view[0] is not None:
                self.camera_pos = view[0]
            if view[1] is not None:
                self.camera_target = view[1]
            if view[2] is not None:
                self.camera_fov = view[2]
            if view[3] is not None:
                self.camera_rotation = view[3]

        return {
            'camera_pos': self.camera_pos,
            'camera_target': self.camera_target,
            'camera_fov': self.camera_fov,
            'camera_rotation': self.camera_rotation,
        }

    def change_view(self, view, fit=False):
        self.view = view
        self.camera_config = self.view_to_camera_config(self.view)
        if fit:
            self.camera_config = self.zoom_camera_to_content(self.camera_config)
        self.viewer.change_view(**self.camera_config)

    def zoom_camera_to_content(self, camera_config):
        coverage_radius = (self.max_coordinate - self.min_coordinate) / 2
        if np.isnan(coverage_radius).any():
            return camera_config
        coverage_radius = np.linalg.norm(coverage_radius)
        appropriate_distance = 0.75 * coverage_radius / np.sin(np.deg2rad(camera_config['camera_fov'] / 2))
        current_distance = np.linalg.norm(camera_config['camera_pos'])
        zoom_factor = appropriate_distance / current_distance
        camera_config['camera_pos'] = tuple([x * zoom_factor for x in camera_config['camera_pos']])
        return camera_config

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

    def prepare_color(self, color):
        # prepare the color to the right format
        # set a base color if not specified
        if color is None:
            color = self.null_color
        # make the colors into a numpy array
        color = np.array(color)
        return color

    def create_surface_mesh_object(self, object_id, vertices, triangles, color=None, **kwargs):
        # reformat color
        color = self.prepare_color(color)

        return {
            **{
                'object_id': object_id,
                'object_type': 'surface_mesh',
                'vertices': vertices,
                'triangles': triangles,
                'base_color': color,
                'layers': {},
                'visibility': True,
                'render_update_required': True,
                'rendered': False,
            },
            **kwargs
        }

    def create_spheres_object(self, object_id, coordinates, radii, color=None, **kwargs):
        # reformat color
        color = self.prepare_color(color)

        # reshape radii to expected shape
        radii = np.array(radii)
        if radii.shape == ():
            # assume equal radii along all directions
            radii = radii[np.newaxis].repeat(3)
        if radii.shape == (coordinates.shape[0],):
            # assume equal radii along all directions for all spheres
            radii = radii[:, np.newaxis].repeat(3, 1)
        if radii.shape == (3,):
            # assume equal radii for all spheres
            radii = radii[np.newaxis, :].repeat(coordinates.shape[0], 0)

        return {
            **{
                'object_id': object_id,
                'object_type': 'spheres',
                'coordinates': coordinates,
                'radii': radii,
                'base_color': color,
                'layers': {},
                'visibility': True,
                'render_update_required': True,
                'rendered': False,
            },
            **kwargs
        }

    def create_cylinders_object(self, object_id, coordinates, radii, color=None, **kwargs):
        # reformat color
        color = self.prepare_color(color)

        # reshape radii to expected shape
        radii = np.array(radii)
        if radii.shape == ():
            # assume equal radii along all directions
            radii = radii[np.newaxis].repeat(coordinates.shape[0])

        return {
            **{
                'object_id': object_id,
                'object_type': 'cylinders',
                'coordinates': coordinates,
                'radii': radii,
                'base_color': color,
                'layers': {},
                'visibility': True,
                'render_update_required': True,
                'rendered': False,
            },
            **kwargs
        }

    def visualize_spheres(self, coordinates, radii, coordinate_offset=0, color=None, **kwargs):
        """
        This function can be used to add arbitrary spheres to the view.
        """
        # generate a unique id for the object
        unique_id = f'{utils.generate_unique_id()}'
        object_id = f'spheres#{unique_id}'
        self.created_objects[object_id] = self.create_spheres_object(
            object_id=object_id,
            coordinates=coordinates,
            radii=radii,
            color=color,
            object_offset_coordinate=coordinate_offset,
            **kwargs,
        )

        # draw to update visualization
        self.draw()

        return self.created_objects[object_id]

    def visualize_cylinders(self, coordinates, radii, coordinate_offset=0, color=None, **kwargs):
        """
        This function can be used to add arbitrary cylinders to the view to
        represent lines connecting pairs of coordinates.
        """
        # generate a unique id for the object
        unique_id = f'{utils.generate_unique_id()}'
        object_id = f'cylinders#{unique_id}'
        self.created_objects[object_id] = self.create_cylinders_object(
            object_id=object_id,
            coordinates=coordinates,
            radii=radii,
            color=color,
            object_offset_coordinate=coordinate_offset,
            **kwargs,
        )

        # draw to update visualization
        self.draw()

        return self.created_objects[object_id]

    def visualize_cifti_space(self, cortical_surface_model_id=None, cifti_template_file=None, volumetric_structures='none',
                              volume_rendering='surface', cifti_expansion_scale=0, cifti_expansion_coeffs=cbu.cifti_expansion_coeffs,
                              cifti_left_right_seperation=0, volumetric_structure_offset=(0, 0, 0), **kwargs):
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
        coordinate_offset = np.array([(-cifti_left_right_seperation / 2), 0, 0])
        self.created_objects[object_id] = self.create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]['left_vertices'],
            triangles=self.created_objects[cortical_surface_model_id]['left_triangles'],
            surface_model_id=model_id,
            data_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
            object_offset_coordinate=coordinate_offset,
        )

        # add the right cortical surface model
        brain_structure = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
        brain_model = brain_models[brain_structures.index(brain_structure)]
        object_id = f'{brain_structure}#{unique_id}'
        contained_object_ids.append(object_id)
        coordinate_offset = np.array([(cifti_left_right_seperation / 2), 0, 0])
        self.created_objects[object_id] = self.create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]['right_vertices'],
            triangles=self.created_objects[cortical_surface_model_id]['right_triangles'],
            surface_model_id=model_id,
            data_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
            object_offset_coordinate=coordinate_offset,
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
                coordinate_offset = (cifti_expansion_scale * np.array(cifti_expansion_coeffs[brain_structure])) + np.array(volumetric_structure_offset)
                if volume_rendering == 'spheres':
                    self.created_objects[object_id] = self.create_spheres_object(
                        object_id=object_id,
                        coordinates=coordinates,
                        radii=radii,
                        data_index_offset=brain_model.index_offset,
                        data_index_count=brain_model.index_count,
                        object_collection_id=object_collection_id,
                        object_offset_coordinate=coordinate_offset,
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
                        object_offset_coordinate=coordinate_offset,
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
                        data_map=nearest_indices,
                        object_collection_id=object_collection_id,
                        object_offset_coordinate=coordinate_offset,
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

    # TODO: modify function is not working correctly...
    # def modify_cifti_dscalar_layer(self, created_layer, dscalar_file=None, loaded_dscalar=None, dscalar_data=None, dscalar_index=0, **kwargs):
    #     # load the cifti dscalar file
    #     if dscalar_file is not None:
    #         dscalar = self.load_file(dscalar_file, nib.load)
    #         dscalar_data = dscalar.get_fdata()[dscalar_index]
    #     elif loaded_dscalar is not None:
    #         dscalar_data = loaded_dscalar.get_fdata()[dscalar_index]
    #     elif dscalar_data is None:
    #         raise Exception(f'No dscalar was provided for add_CIFTI_dscalar_layer.')

    #     # convert data to colors
    #     dscalar_colors = self.data_to_colors(dscalar_data, **kwargs)

    #     # modify layer
    #     created_layer['dscalar_data'] = dscalar_data
    #     created_layer['dscalar_colors'] = dscalar_colors
    #     created_layer['layer_update_required'] = True

    #     # save modified layer
    #     layer_id = created_layer['layer_id']
    #     self.created_layers[layer_id] = created_layer

    #     # draw to update visualization
    #     self.draw()

    #     return created_layer

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

    def get_object_base_colors_for_render(self, object_id, size):
        # load the object
        colored_object = self.created_objects[object_id]

        # load base colors and reshape if required
        base_color = colored_object.get('base_color', self.null_color)
        if base_color.shape == (3,):
            # add alpha channel
            base_color = np.append(base_color, 1)
        if base_color.shape == (4,):
            # generate fixed color for all spheres
            base_color = np.array(base_color)[np.newaxis, :].repeat(size, 0)

        # enusure that the base colors have the correct shape
        assertion_error_message = (
            f"The provided colors for {colored_object['object_type']} cannot be unpacked appropriately: "
            f"{colored_object['base_color'].shape}"
        )
        assert base_color.shape == (size, 4), assertion_error_message

        return base_color

    def apply_layer_colors_for_render(self, object_id, size, colors):
        # load the object
        colored_object = self.created_objects[object_id]

        # add layers one by one
        for layer_idx in range(len(colored_object['layers'])):
            layer_id = colored_object['layers'][layer_idx]
            layer_object = self.created_layers[layer_id]

            # extract colors from layer
            index_offset = colored_object['data_index_offset']
            index_count = colored_object['data_index_count']
            extracted_colors = layer_object['layer_colors'][index_offset: (index_offset + index_count)]

            # check colors have expected shape
            if (extracted_colors.shape[0] == index_count):
                # the extracted colors may need to be resampled by a map
                if colored_object.get('data_map', None) is not None:
                    extracted_colors = extracted_colors[colored_object['data_map']]

                # the extracted colors need to be assigned according to indices
                layer_colors = np.array(self.no_color)[np.newaxis, :].repeat(size, 0)
                if colored_object.get('data_indices', None) is not None:
                    layer_colors[colored_object['data_indices']] = extracted_colors
                else:
                    layer_colors = extracted_colors

                # compute the layer overlay
                colors = self.compute_overlay_colors(colors, layer_colors)

        return colors

    def get_object_render_colors(self, object_id, size):
        colors = self.get_object_base_colors_for_render(object_id, size)
        colors = self.apply_layer_colors_for_render(object_id, size, colors)
        return colors

    def render_surface_mesh(self, object_id):
        # load vertices and triangles
        surface_mesh_object = self.created_objects[object_id]
        surface_vertices = surface_mesh_object['vertices']
        surface_triangles = surface_mesh_object['triangles']

        # apply necessary changes in coordinates by the offset
        surface_vertices += surface_mesh_object.get('object_offset_coordinate', 0)

        # load appropriate render colors
        surface_colors = self.get_object_render_colors(object_id, surface_vertices.shape[0])

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

        # apply necessary changes in coordinates by the offset
        coordinates += spheres_object.get('object_offset_coordinate', 0)

        # load appropriate render colors
        colors = self.get_object_render_colors(object_id, coordinates.shape[0])

        # clear existing render

        # render the object
        spheres_object['colors'] = colors
        rendered_spheres = self.viewer.add_points(coordinates, radii, colors)
        spheres_object['rendered_spheres'] = rendered_spheres
        spheres_object['rendered'] = True

        # update object boundaries
        self.min_coordinate = np.min([self.min_coordinate, (coordinates - radii).min(0)], 0)
        self.max_coordinate = np.max([self.max_coordinate, (coordinates + radii).max(0)], 0)

        # signal that render was updated
        self.created_objects[object_id]['render_update_required'] = False

    def render_cylinders(self, object_id):
        # load vertices and triangles
        cylinders_object = self.created_objects[object_id]
        coordinates = cylinders_object['coordinates']
        radii = cylinders_object['radii']

        # apply necessary changes in coordinates by the offset
        coordinates += cylinders_object.get('object_offset_coordinate', 0)

        # load appropriate render colors
        colors = self.get_object_render_colors(object_id, coordinates.shape[0])

        # clear existing render

        # render the object
        cylinders_object['colors'] = colors
        rendered_cylinders = self.viewer.add_lines(coordinates, radii, colors)
        cylinders_object['rendered_cylinders'] = rendered_cylinders
        cylinders_object['rendered'] = True

        # update object boundaries
        self.min_coordinate = np.min([self.min_coordinate, (coordinates.min(0).min(0) - radii.min())], 0)
        self.max_coordinate = np.max([self.max_coordinate, (coordinates.max(0).max(0) + radii.max())], 0)

        # signal that render was updated
        self.created_objects[object_id]['render_update_required'] = False

    def render_object(self, object_id):
        if self.created_objects[object_id]['object_type'] == 'surface_mesh':
            self.render_surface_mesh(object_id)
        elif self.created_objects[object_id]['object_type'] == 'spheres':
            self.render_spheres(object_id)
        elif self.created_objects[object_id]['object_type'] == 'cylinders':
            self.render_cylinders(object_id)

    def center_camera(self, fit=True):
        new_center_coordinate = (self.min_coordinate + self.max_coordinate) / 2
        if (self.center_coordinate != new_center_coordinate).any():
            self.center_coordinate = new_center_coordinate
            self.change_view((None, self.center_coordinate, None, None), fit=fit)

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
