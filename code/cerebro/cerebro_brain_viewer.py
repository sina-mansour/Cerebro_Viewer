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

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import scipy.sparse as sparse

from . import renderer
from . import cerebro_utils as utils
from . import cerebro_brain_utils as cbu

# Type hint imports
from typing import Dict, Any

# suppress trivial nibabel warnings, see https://github.com/nipy/nibabel/issues/771
nib.imageglobals.logger.setLevel(40)


class Cerebro_brain_viewer:
    """Cerebero brain viewer engine

    This class contains the necessary logical units and input/output handlers
    required to visualize various brain imaging formats in the same viewer
    window.

    Parameters
    ----------
    background_color
        RGBA spec for the background of the rendered brain image.
    view
        Description of the rendered viewing angle.
    null_color
        RGBA spec for the color of objects without any overlay.
    no_color
        RGBA spec for the color of objects that have been masked out.
    offscreen
        True if the viewer should be run "headless", i.e. without the live GUI.

    Attributes
    ----------
    min_coordinate
        The minimum coordinate to be rendered.
    max_coordinate
        The maximum coordinate to be rendered.
    center_coordinate
        The center of the rendered region.
    renderer_type
        Unused(?) string describing the renderer.
    camera_config
        Dictionary describing the camera configuration based on the view.
    viewer
        The actual renderer to be used.
    created_objects
        Dictionary storing created objects.
    created_layers
        Dictionary storing created layers.
    loaded_files
        Cache for loaded files
    default_objects
        Dictionary storing the default object of each type.
    """

    def __init__(
        self,
        background_color: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 0.0),
        view: str
        | tuple[
            tuple[float, float, float], tuple[float, float, float], float, float
        ] = "R",
        null_color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 0.3),
        no_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        offscreen: bool = False,
    ):
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
        self.renderer_type = "panda3d"
        self.camera_config = self._view_to_camera_config(self.view)

        self.viewer = renderer.Renderer_panda3d(
            background_color=background_color, offscreen=offscreen, **self.camera_config
        )

        # Create a dictionary for created objects
        self.created_objects = {}

        # Create a dictionary for created layers
        self.created_layers = {}

        # Create a dictionary for loaded default objects
        self.default_objects = {}

        # Create a file handler
        self.file_handler = cbu.File_handler()

    def __del__(self):
        del self.viewer

    # Camera view configuration
    def _view_to_camera_config(self, view):
        if isinstance(view, str):
            self.camera_target = self.center_coordinate
            self.camera_fov = 25
            self.camera_rotation = 0

        if (view == "R") or (view == "Right"):
            self.camera_pos = (400, 0, 0)
        elif (view == "L") or (view == "Left"):
            self.camera_pos = (-400, 0, 0)
        elif (view == "A") or (view == "Anterior"):
            self.camera_pos = (0, 400, 0)
        elif (view == "P") or (view == "Posterior"):
            self.camera_pos = (0, 400, 0)
        elif (view == "S") or (view == "Superior"):
            self.camera_pos = (0, 0, 400)
            self.camera_rotation = -90
        elif (view == "I") or (view == "Inferior"):
            self.camera_pos = (0, 0, -400)
            self.camera_rotation = 90
        else:
            # Alternatively the user could provide an arbitrary camera config instead
            # of the view
            if view[0] is not None:
                self.camera_pos = view[0]
            if view[1] is not None:
                self.camera_target = view[1]
            if view[2] is not None:
                self.camera_fov = view[2]
            if view[3] is not None:
                self.camera_rotation = view[3]

        return {
            "camera_pos": self.camera_pos,
            "camera_target": self.camera_target,
            "camera_fov": self.camera_fov,
            "camera_rotation": self.camera_rotation,
        }

    def _zoom_camera_to_content(self, camera_config):
        coverage_radius = (self.max_coordinate - self.min_coordinate) / 2
        if np.isnan(coverage_radius).any():
            return camera_config
        coverage_radius = np.linalg.norm(coverage_radius)
        appropriate_distance = (
            0.75 * coverage_radius / np.sin(np.deg2rad(camera_config["camera_fov"] / 2))
        )
        current_distance = np.linalg.norm(camera_config["camera_pos"])
        zoom_factor = appropriate_distance / current_distance
        camera_config["camera_pos"] = tuple(
            [x * zoom_factor for x in camera_config["camera_pos"]]
        )
        return camera_config

    def change_view(self, view, fit=False):
        """Specify the viewing angle of the brain.
        This method can be used to change the viewing angle of the brain using
        pre-configured angle options or custom viewing options.

        Parameters
        ----------
        self
        The Cerebro_brain_viewer object.

        view
        Description of the rendered viewing angle. Pre-configured options include:
            "R" or "Right" for right hemisphere lateral view
            "L" or "Left" for left hemisphere lateral view
            "A" or "Anterior" for anterior view
            "P" or "Posterior" for posterior view
            "S" or "Superior" for superior view
            "I" or "Inferior" for inferior view
        Alternatively, you may provide a tuple of the form (camera_pos, camera_target,
        camera_fov, camera_rotation) to specify the camera configuration directly.

        fit
        If True, the camera will be zoomed to fit the content of the scene.
        """

        self.view = view
        self.camera_config = self._view_to_camera_config(self.view)
        if fit:
            self.camera_config = self._zoom_camera_to_content(self.camera_config)
        self.viewer.change_view(**self.camera_config)

    def center_camera(self, fit=True):
        """Center the camera on the brain.
        This method can be used to center the camera on the brain.

        Parameters
        ----------
        self
        The Cerebro_brain_viewer object.

        fit
        If True, the camera will be zoomed to fit the brain.
        """
        new_center_coordinate = (self.min_coordinate + self.max_coordinate) / 2
        if (self.center_coordinate != new_center_coordinate).any():
            self.center_coordinate = new_center_coordinate
            self.change_view((None, self.center_coordinate, None, None), fit=fit)

    def load_GIFTI_cortical_surface_models(self, left_surface_file, right_surface_file):
        """Load a GIFTI cortical surface model.

        This function loads a GIFTI cortical surface model from two separate GIFTI surface files
        for the left and right hemispheres. The loaded cortical surface model is
        stored in the object's internal data structure.

        Args:
            left_surface_file (str): File path to the GIFTI surface file containing the left hemisphere data.
            right_surface_file (str): File path to the GIFTI surface file containing the right hemisphere data.

        Returns:
            dict: A dictionary representing the loaded cortical surface model, with the following keys:

                - 'object_id' (str): A unique identifier for the cortical surface model.
                - 'object_type' (str): The type of the object, which is 'cortical_surface_model'.
                - 'left_vertices' (list): A list of 3D coordinates representing the vertices of the left hemisphere surface.
                - 'left_triangles' (list): A list of triplets representing the triangles of the left hemisphere surface.
                - 'right_vertices' (list): A list of 3D coordinates representing the vertices of the right hemisphere surface.
                - 'right_triangles' (list): A list of triplets representing the triangles of the right hemisphere surface.

        Raises:
            FileNotFoundError: If either 'left_surface_file' or 'right_surface_file' is not found or cannot be accessed.
            ValueError: If the loaded GIFTI surface files have an incompatible format or structure.

        Example:
            left_file_path = '/path/to/left_surface.gii'
            right_file_path = '/path/to/right_surface.gii'
            surface_model = my_brain_viewer.load_GIFTI_cortical_surface_models(left_file_path, right_file_path)

        """
        # get a unique ID
        object_type = "cortical_surface_model"
        object_id = f"{object_type}#{utils.generate_unique_id()}"
        # left cortical surface
        left_vertices, left_triangles = self.file_handler.load_file(
            left_surface_file, cbu.load_GIFTI_surface
        )
        # right cortical surface
        right_vertices, right_triangles = self.file_handler.load_file(
            right_surface_file, cbu.load_GIFTI_surface
        )

        # create a dictionary to store the loaded surface model data
        created_object = {
            "object_id": object_id,
            "object_type": object_type,
            "left_vertices": left_vertices,
            "left_triangles": left_triangles,
            "right_vertices": right_vertices,
            "right_triangles": right_triangles,
        }
        self.created_objects[object_id] = created_object

        # set as default surface model
        self.default_objects[object_type] = object_id

        # return object to user
        return created_object

    def load_template_GIFTI_cortical_surface_models(self, template_surface="inflated"):
        """Load a GIFTI cortical surface model using a template surface.

        This function loads a GIFTI cortical surface model by using a template surface name.
        It retrieves the file paths of the left and right hemispheres from the template surface.
        The loaded cortical surface model is then stored in the object's internal data structure.

        Args:
            template_surface (str, optional): The name of the template surface to use.
            Defaults to 'inflated'.

        Returns:
            dict: see load_GIFTI_cortical_surface_models for keys.

        Raises:
            ValueError: If the provided 'template_surface' is not recognized or not available.

        Example:
            surface = 'pial'
            surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models(surface)
        """
        # get file paths from template surface name
        (
            left_surface_file,
            right_surface_file,
        ) = cbu.get_left_and_right_GIFTI_template_surface(template_surface)

        # load the surface model
        cortical_surface_model = self.load_GIFTI_cortical_surface_models(
            left_surface_file, right_surface_file
        )

        # return object to user
        return cortical_surface_model

    def _prepare_color(self, color):
        """Prepare the color value to the appropriate format.

        This function takes a color value, checks if it is specified (not None), and sets a base color if it is not provided.
        It then converts the color value into a NumPy array.

        Args:
            color (tuple or None): The color value to be prepared. If None, a base color will be used.

        Returns:
            numpy.ndarray: The color value as a NumPy array.

        Example:
            color = (1.0, 0.5, 0.0)  # RGB values
            prepared_color = my_brain_viewer._prepare_color(color)
        """
        # set a base color if not specified
        if color is None:
            color = self.null_color
        # make the colors into a numpy array
        color = np.array(color)
        return color

    def _create_surface_mesh_object(
        self, object_id, vertices, triangles, color=None, **kwargs
    ):
        """Create a surface mesh object.

        This function creates a surface mesh object with the given 'object_id', 'vertices', and 'triangles'.
        The object can be customized with additional 'kwargs' for specific use cases.

        Args:
            object_id (str): A unique identifier for the surface mesh object.
            vertices (numpy.ndarray): The vertices of the surface mesh as a 2D NumPy array.
            triangles (numpy.ndarray): The triangles (faces) of the surface mesh as a 2D NumPy array.
            color (tuple or None, optional): The base color for the surface mesh. If None, a default color will be used.
            **kwargs: Additional keyword arguments to customize the surface mesh object.

        Returns:
            dict: A dictionary representing the surface mesh object with the following keys:
                - 'object_id' (str): The unique identifier of the surface mesh object.
                - 'object_type' (str): The type of the object, set as 'surface_mesh'.
                - 'vertices' (numpy.ndarray): The vertices of the surface mesh.
                - 'triangles' (numpy.ndarray): The triangles (faces) of the surface mesh.
                - 'base_color' (numpy.ndarray): The base color of the surface mesh as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            object_id = "surface_mesh_1"
            vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            triangles = np.array([[0, 1, 2]])
            surface_mesh = my_brain_viewer.create_surface_mesh_object(
                object_id,
                vertices,
                triangles,
                color=(1.0, 0.5, 0.0),
                visibility=True
            )
        """
        # reformat color
        color = self._prepare_color(color)

        return {
            **{
                "object_id": object_id,
                "object_type": "surface_mesh",
                "vertices": vertices,
                "triangles": triangles,
                "base_color": color,
                "layers": {},
                "visibility": True,
                "render_update_required": True,
                "rendered": False,
            },
            **kwargs,
        }

    def create_spheres_object(
        self, object_id, coordinates, radii, color=None, **kwargs
    ):
        """Create a spheres object.

        This function creates a spheres object with the given 'object_id', 'coordinates', and 'radii'.
        The object can be customized with additional 'kwargs' for specific use cases.

        Args:
            object_id (str): A unique identifier for the spheres object.
            coordinates (numpy.ndarray): The coordinates of the spheres as a 2D NumPy array (shape: Nx3).
            radii (float, numpy.ndarray): The radii of the spheres. Can be a single value or a 1D NumPy array (shape: N).
            color (tuple or None, optional): The base color for the spheres. If None, a default color will be used.
            **kwargs: Additional keyword arguments to customize the spheres object.

        Returns:
            dict: A dictionary representing the spheres object with the following keys:
                - 'object_id' (str): The unique identifier of the spheres object.
                - 'object_type' (str): The type of the object, set as 'spheres'.
                - 'coordinates' (numpy.ndarray): The coordinates of the spheres.
                - 'radii' (numpy.ndarray): The radii of the spheres.
                - 'base_color' (numpy.ndarray): The base color of the spheres as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
            radii = 0.5
            object_id = "spheres_1"
            spheres = my_brain_viewer.create_spheres_object(
                object_id=object_id,
                coordinates,
                radii,
                color=(0.0, 1.0, 0.0),
                visibility=True
            )
        """
        # reformat color
        color = self._prepare_color(color)

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
                "object_id": object_id,
                "object_type": "spheres",
                "coordinates": coordinates,
                "radii": radii,
                "base_color": color,
                "layers": {},
                "visibility": True,
                "render_update_required": True,
                "rendered": False,
            },
            **kwargs,
        }

    def create_cylinders_object(
        self, object_id, coordinates, radii, color=None, **kwargs
    ):
        """Create a cylinders object.

        This function creates a cylinders object with the given 'object_id', 'coordinates', and 'radii'.
        The object can be customized with additional 'kwargs' for specific use cases.

        Args:
            object_id (str): A unique identifier for the cylinders object.
            coordinates (numpy.ndarray): The coordinates of the cylinders as a 2D NumPy array (shape: Nx3).
            radii (float, numpy.ndarray): The radii of the cylinders. Can be a single value or a 1D NumPy array (shape: N).
            color (tuple or None, optional): The base color for the cylinders. If None, a default color will be used.
            **kwargs: Additional keyword arguments to customize the cylinders object.

        Returns:
            dict: A dictionary representing the cylinders object with the following keys:
                - 'object_id' (str): The unique identifier of the cylinders object.
                - 'object_type' (str): The type of the object, set as 'cylinders'.
                - 'coordinates' (numpy.ndarray): The coordinates of the cylinders.
                - 'radii' (numpy.ndarray): The radii of the cylinders.
                - 'base_color' (numpy.ndarray): The base color of the cylinders as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
            radii = 0.2
            object_id = "cylinders_1"
            cylinders = my_brain_viewer.create_cylinders_object(
                object_id=object_id,
                coordinates,
                radii,
                color=(0.0, 0.0, 1.0),
                visibility=True
            )
        """
        # reformat color
        color = self._prepare_color(color)

        # reshape radii to expected shape
        radii = np.array(radii)
        if radii.shape == ():
            # assume equal radii along all directions
            radii = radii[np.newaxis].repeat(coordinates.shape[0])

        return {
            **{
                "object_id": object_id,
                "object_type": "cylinders",
                "coordinates": coordinates,
                "radii": radii,
                "base_color": color,
                "layers": {},
                "visibility": True,
                "render_update_required": True,
                "rendered": False,
            },
            **kwargs,
        }

    def visualize_spheres(
        self, coordinates, radii=1, coordinate_offset=0, color=None, **kwargs
    ):
        """Visualize arbitrary spheres in the viewer.

        This function allows you to add arbitrary spheres to the viewer's visualization.
        It creates a new spheres object with the specified 'coordinates', 'radii', 'color',
        and other custom properties using additional keyword arguments (kwargs).

        Args:
            coordinates (numpy.ndarray): The coordinates of the spheres as a 2D NumPy array (shape: Nx3).
            radii (float, numpy.ndarray, optional): The radii of the spheres. Can be a single value or a 1D NumPy array (shape: N).
                Default value is 1.
            coordinate_offset (float, optional): An offset to apply to the 'coordinates'. Default value is 0.
                Note: the offset can be a list/vector of length 3 denoting a 3-dimensional offset (x, y, z)
            color (tuple or None, optional): The base color for the spheres. If None, a default color will be used.
                Default value is None.
            **kwargs: Additional keyword arguments to customize the spheres object.

        Returns:
            dict: A dictionary representing the created spheres object with the following keys:
                - 'object_id' (str): The unique identifier of the spheres object.
                - 'object_type' (str): The type of the object, set as 'spheres'.
                - 'coordinates' (numpy.ndarray): The coordinates of the spheres.
                - 'radii' (numpy.ndarray): The radii of the spheres.
                - 'base_color' (numpy.ndarray): The base color of the spheres as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
            radii = 0.2
            spheres = my_brain_viewer.visualize_spheres(
                coordinates,
                radii,
                color=(1.0, 0.0, 0.0),
                coordinate_offset=10.0,
                visibility=True
            )
        """
        if type(coordinates)==list:
            coordinates = np.array(coordinates)

        # generate a unique id for the object
        unique_id = f"{utils.generate_unique_id()}"
        object_id = f"spheres#{unique_id}"
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

    def visualize_cylinders(
        self, coordinates, radii=1, coordinate_offset=0, color=None, **kwargs
    ):
        """Visualize arbitrary cylinders in the viewer to represent lines connecting pairs of coordinates.

        This function allows you to add arbitrary cylinders to the viewer's visualization. The cylinders
        are used to represent lines connecting pairs of 'coordinates'.

        Args:
            coordinates (numpy.ndarray): The coordinates of the cylinders as a 2D NumPy array (shape: Nx2x3).
            radii (float, numpy.ndarray, optional): The radii of the cylinders. Can be a single value or a 1D NumPy array (shape: N).
                Default value is 1.
            coordinate_offset (float, optional): An offset to apply to the 'coordinates'. Default value is 0.
                Note: the offset can be a list/vector of length 3 denoting a 3-dimensional offset (x, y, z)
            color (tuple or None, optional): The base color for the cylinders. If None, a default color will be used.
                Default value is None.
            **kwargs: Additional keyword arguments to customize the cylinders object.

        Returns:
            dict: A dictionary representing the created cylinders object with the following keys:
                - 'object_id' (str): The unique identifier of the cylinders object.
                - 'object_type' (str): The type of the object, set as 'cylinders'.
                - 'coordinates' (numpy.ndarray): The coordinates of the cylinders.
                - 'radii' (numpy.ndarray): The radii of the cylinders.
                - 'base_color' (numpy.ndarray): The base color of the cylinders as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            coordinates = np.array([[[0, 0, 0], [1, 1, 1]], [[2, 2, 2], [3, 3, 3]]])
            radii = 0.1
            cylinders = my_brain_viewer.visualize_cylinders(
                coordinates,
                radii,
                color=(0.0, 0.0, 1.0),
                coordinate_offset=5.0,
                visibility=True
            )
        """
        if type(coordinates)==list:
            coordinates = np.array(coordinates)
        # generate a unique id for the object
        unique_id = f"{utils.generate_unique_id()}"
        object_id = f"cylinders#{unique_id}"
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

    def visualize_network(
        self,
        adjacency,
        node_coordinates,
        node_radii=5,
        edge_radii=1,
        node_color=None,
        edge_color=None,
        node_kwargs={},
        edge_kwargs={},
    ):
        """Visualize a 3D network with a ball and stick model.

        This function allows you to visualize a 3D network using a ball and stick model. Nodes in the network are represented
        as spheres, and edges connecting the nodes are represented as cylinders.

        Args:
            adjacency (numpy.ndarray or scipy.sparse.spmatrix): The adjacency matrix representing the network connections.
                Should be a square matrix where each entry (i, j) indicates the weight or presence of an edge between nodes i and j.
            node_coordinates (numpy.ndarray): The 3D coordinates of the nodes in the network as a 2D NumPy array (shape: Nx3).
            node_radii (float or numpy.ndarray, optional): The radii of the spheres representing the nodes. Can be a single value or a 1D NumPy array (shape: N).
                Default value is 5.
            edge_radii (float or numpy.ndarray, optional): The radii of the cylinders representing the edges. Can be a single value or a 1D NumPy array (shape: M).
                Default value is 1.
            node_color (tuple or None, optional): The base color for the nodes. If None, a default color will be used.
                Default value is None.
            edge_color (tuple or None, optional): The base color for the edges. If None, a default color will be used.
                Default value is None.
            node_kwargs (dict, optional): Additional keyword arguments to customize the nodes. These arguments will be passed to the `visualize_spheres` function.
                Default value is an empty dictionary ({}).
            edge_kwargs (dict, optional): Additional keyword arguments to customize the edges. These arguments will be passed to the `visualize_cylinders` function.
                Default value is an empty dictionary ({}).

        Returns:
            dict: A dictionary representing the created network collection object with the following keys:
                - 'object_id' (str): The unique identifier of the network collection object.
                - 'object_type' (str): The type of the object, set as 'object_collection'.
                - 'collection_type' (str): The type of collection, set as 'network'.
                - 'contained_object_ids' (list): A list of unique identifiers of the objects contained in the collection (nodes and edges).
                - 'layers' (dict): A dictionary to store additional layers associated with the network collection.

        Example:
            adjacency = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
            node_coordinates = np.array([[10, 10, 10], [20, 20, 20], [30, 30, 30]])
            node_radii = 4
            edge_radii = 0.5
            network_collection = my_brain_viewer.visualize_network(
                adjacency,
                node_coordinates,
                node_radii,
                edge_radii,
                node_color=(1.0, 0.0, 0.0),
                edge_color=(0.0, 1.0, 0.0),
                node_kwargs={"visibility": True},
                edge_kwargs={"visibility": True}
            )
        """

        #if needed, convert list of lists to numpy array
        if type(adjacency)==list:
            adjacency = np.array(adjacency)
        if type(node_coordinates)==list:
            node_coordinates = np.array(node_coordinates)

        # Create edge list from adjacency
        adjacency = sparse.coo_matrix(adjacency)
        edge_list = np.array([adjacency.row, adjacency.col]).T

        # create nodes and edges
        nodes = self.visualize_spheres(
            node_coordinates, radii=node_radii, color=node_color, **node_kwargs
        )
        edges = self.visualize_cylinders(
            node_coordinates[edge_list],
            radii=edge_radii,
            color=edge_color,
            **edge_kwargs,
        )

        # generate a unique id for the object
        unique_id = f"{utils.generate_unique_id()}"
        object_id = f"network#{unique_id}"

        # store all visualized objects
        contained_object_ids = [nodes["object_id"], edges["object_id"]]

        # create the network collection object
        collection_object = {
            "object_id": object_id,
            "object_type": "object_collection",
            "collection_type": "network",
            "contained_object_ids": contained_object_ids,
            "layers": {},
        }
        self.created_objects[object_id] = collection_object

        # draw to update visualization
        self.draw()

        # return object to user
        return collection_object

    def visualize_cifti_space(
        self,
        cortical_surface_model_id=None,
        cifti_template_file=None,
        volumetric_structures="none",
        volume_rendering="surface",
        cifti_expansion_scale=0,
        cifti_expansion_coeffs=cbu.cifti_expansion_coeffs,
        cifti_left_right_seperation=0,
        volumetric_structure_offset=(0, 0, 0),
        **kwargs,
    ):
        """Visualize a CIFTI space combining cortical surface models and subcortical structures.

        This function allows you to visualize a CIFTI space by combining cortical surface models and subcortical structures.
        The cortical surface models are rendered as surface meshes, while the subcortical structures can be rendered either
        as spheres or as a surface generated using the marching cube algorithm with optional smoothing.

        Args:
            cortical_surface_model_id (str, optional): The unique identifier of the cortical surface model object to be visualized.
                If not provided, the default cortical surface model will be used.
            cifti_template_file (str or None, optional): The file path of the CIFTI template file to be used for visualization.
                If None, the default CIFTI template file will be used.
            volumetric_structures (str or None, optional): A string specifying which volumetric structures to visualize.
                It can take the following values: "none" (no volumetric structures), "all" (all available volumetric structures),
                or a space-separated string with specific volumetric structure names (e.g., "CIFTI_STRUCTURE_ACCUMBENS_LEFT CIFTI_STRUCTURE_AMYGDALA_LEFT").
                Default value is "none".
            volume_rendering (str, optional): The rendering method for subcortical structures. It can take one of the following values:
                "surface" (use the marching cube algorithm with optional smoothing), "spheres" (render as spheres), or "spheres_peeled"
                (apply peeling to get a thin layer from subcortical structures). Default value is "surface".
            cifti_expansion_scale (float, optional): The scale factor for expanding the volumetric structures along their normal vectors.
                This value is applied to all structures. Default value is 0.
            cifti_expansion_coeffs (dict, optional): A dictionary containing expansion coefficients for each volumetric structure.
                The keys should be CIFTI structure names (e.g., "CIFTI_STRUCTURE_ACCUMBENS_LEFT") and the values should be 3D arrays representing
                the expansion coefficients along the X, Y, and Z axes. Default value is cbu.cifti_expansion_coeffs.
            cifti_left_right_seperation (float, optional): The distance between the left and right cortical surface models.
                Default value is 0.
            volumetric_structure_offset (tuple, optional): A 3D tuple specifying the offset for the volumetric structures.
                This value is applied to all structures. Default value is (0, 0, 0).
            **kwargs: Additional keyword arguments that can be passed to the visualization methods (e.g., smoothing parameters).

        Returns:
            dict: A dictionary representing the created CIFTI space collection object with the following keys:
                - 'object_id' (str): The unique identifier of the CIFTI space collection object.
                - 'object_type' (str): The type of the object, set as 'object_collection'.
                - 'collection_type' (str): The type of collection, set as 'cifti_space'.
                - 'cifti_template' (nibabel.cifti2.Cifti2Image): The loaded CIFTI template as a Cifti2Image object.
                - 'contained_object_ids' (list): A list of unique identifiers of the objects contained in the collection (surface models and subcortical structures).
                - 'layers' (dict): A dictionary to store additional layers associated with the CIFTI space collection.
                - 'surface_model_id' (str): The unique identifier of the cortical surface model object used for visualization.

        Example:
            cifti_template_file = "path/to/my/cifti/template.dscalar.nii"
            cortical_surface_model_id = "cortical_surface_model#abc123"
            cifti_space_collection = my_brain_viewer.visualize_cifti_space(
                cortical_surface_model_id,
                cifti_template_file,
                volumetric_structures="all",
                volume_rendering="surface",
                cifti_expansion_scale=0.5,
                cifti_expansion_coeffs={
                    "CIFTI_STRUCTURE_ACCUMBENS_LEFT": [0.1, 0.2, 0.3],
                    "CIFTI_STRUCTURE_ACCUMBENS_RIGHT": [0.4, 0.3, 0.2],
                },
                cifti_left_right_seperation=10.0,
                volumetric_structure_offset=(0, 0, 5),
                smoothing_iterations=10,
                smoothing_lambda=0.5
            )
        """
        # initialization
        if cortical_surface_model_id is None:
            cortical_surface_model_id = self.default_objects["cortical_surface_model"]
        if cifti_template_file is None:
            # use default cifti template
            cifti_template_file = cbu.cifti_template_file

        # load the template cifti
        cifti_template = self.file_handler.load_file(cifti_template_file, nib.load)
        brain_models = [x for x in cifti_template.header.get_index_map(1).brain_models]
        brain_structures = [x.brain_structure for x in brain_models]

        # get appropriate IDs
        model_id = self.created_objects[cortical_surface_model_id]["object_id"]
        unique_id = f"{utils.generate_unique_id()}"
        object_collection_id = f"cifti_space#{unique_id}"

        # store all visualized objects
        contained_object_ids = []

        # add the left cortical surface model
        brain_structure = "CIFTI_STRUCTURE_CORTEX_LEFT"
        brain_model = brain_models[brain_structures.index(brain_structure)]
        object_id = f"{brain_structure}#{unique_id}"
        contained_object_ids.append(object_id)
        coordinate_offset = np.array([(-cifti_left_right_seperation / 2), 0, 0])
        self.created_objects[object_id] = self._create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]["left_vertices"],
            triangles=self.created_objects[cortical_surface_model_id]["left_triangles"],
            surface_model_id=model_id,
            data_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
            object_offset_coordinate=coordinate_offset,
        )

        # add the right cortical surface model
        brain_structure = "CIFTI_STRUCTURE_CORTEX_RIGHT"
        brain_model = brain_models[brain_structures.index(brain_structure)]
        object_id = f"{brain_structure}#{unique_id}"
        contained_object_ids.append(object_id)
        coordinate_offset = np.array([(cifti_left_right_seperation / 2), 0, 0])
        self.created_objects[object_id] = self._create_surface_mesh_object(
            object_id=object_id,
            vertices=self.created_objects[cortical_surface_model_id]["right_vertices"],
            triangles=self.created_objects[cortical_surface_model_id][
                "right_triangles"
            ],
            surface_model_id=model_id,
            data_indices=brain_model.vertex_indices,
            data_index_offset=brain_model.index_offset,
            data_index_count=brain_model.index_count,
            object_collection_id=object_collection_id,
            object_offset_coordinate=coordinate_offset,
        )

        # add the subcortical structures
        transformation_matrix = cifti_template.header.get_index_map(
            1
        ).volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        for brain_structure in brain_structures:
            if (
                volumetric_structures
                in cbu.volumetric_structure_inclusion_dict[brain_structure]
            ):
                brain_model = brain_models[brain_structures.index(brain_structure)]
                object_id = f"{brain_structure}#{unique_id}"
                contained_object_ids.append(object_id)
                voxels_ijk = np.array(brain_model.voxel_indices_ijk)
                coordinates = nib.affines.apply_affine(
                    transformation_matrix, voxels_ijk
                )
                voxel_size = nib.affines.voxel_sizes(transformation_matrix)
                radii = voxel_size[np.newaxis, :].repeat(coordinates.shape[0], 0) / 2
                coordinate_offset = (
                    cifti_expansion_scale
                    * np.array(cifti_expansion_coeffs[brain_structure])
                ) + np.array(volumetric_structure_offset)
                if volume_rendering == "spheres":
                    self.created_objects[object_id] = self.create_spheres_object(
                        object_id=object_id,
                        coordinates=coordinates,
                        radii=radii,
                        data_index_offset=brain_model.index_offset,
                        data_index_count=brain_model.index_count,
                        object_collection_id=object_collection_id,
                        object_offset_coordinate=coordinate_offset,
                    )
                elif volume_rendering == "spheres_peeled":
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
                elif volume_rendering == "surface":
                    # use a marching cube algorithm with smoothing to generate a surface model
                    (
                        surface_vertices,
                        surface_triangles,
                    ) = cbu.generate_surface_marching_cube(
                        voxels_ijk, transformation_matrix, **kwargs
                    )
                    nearest_distances, nearest_indices = cbu.get_nearest_neighbors(
                        coordinates, surface_vertices
                    )
                    self.created_objects[object_id] = self._create_surface_mesh_object(
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
            "object_id": object_collection_id,
            "object_type": "object_collection",
            "collection_type": "cifti_space",
            "cifti_template": cifti_template,
            "contained_object_ids": contained_object_ids,
            "layers": {},
            "surface_model_id": model_id,
        }
        self.created_objects[object_collection_id] = collection_object

        # set as default cifti space
        self.default_objects["cifti_space"] = object_collection_id

        # draw to update visualization
        self.draw()

        # return object to user
        return collection_object

    def visualize_mask_surface(
        self,
        volumetric_mask: str | nib.Nifti1Image | Volumetric_data,
        threshold: float = 0.5,
        coordinate_offset: [float, float, float] | float =0,
        color: [float, float, float, float]= None,
        **kwargs: Dict[str, Any]
    ):
        """Visualize a CIFTI space combining cortical surface models and subcortical structures.

        This function allows you to visualize a CIFTI space by combining cortical surface models and subcortical structures.
        The cortical surface models are rendered as surface meshes, while the subcortical structures can be rendered either
        as spheres or as a surface generated using the marching cube algorithm with optional smoothing.

        Args:
            volumetric_mask (str | object): The volumetric mask to be converted to a surface mesh.
                You can provide either the file path, or a loaded mask.
            threshold (float, optional): The threshold to create a binary mask if a nonbinary mask is provided.
            coordinate_offset (float, optional): An offset to apply to the 'coordinates'. Default value is 0.
                Note: the offset can be a list/vector of length 3 denoting a 3-dimensional offset (x, y, z)
            color (tuple or None, optional): The base color for the spheres. If None, a default color will be used.
                Default value is None.
            **kwargs: Additional keyword arguments that can be passed to the visualization methods (e.g., smoothing parameters).

        Returns:
            dict: A dictionary representing the created surface mesh with the following keys:
                - 'object_id' (str): The unique identifier of the surface mesh object.
                - 'object_type' (str): The type of the object, set as 'surface_mesh'.
                - 'vertices' (numpy.ndarray): The vertices of the surface mesh.
                - 'triangles' (numpy.ndarray): The triangles (faces) of the surface mesh.
                - 'base_color' (numpy.ndarray): The base color of the surface mesh as a NumPy array.
                - 'layers' (dict): A dictionary to store additional layers associated with the object.
                - 'visibility' (bool): A flag indicating whether the object is visible.
                - 'render_update_required' (bool): A flag indicating if the object requires a render update.
                - 'rendered' (bool): A flag indicating if the object has been rendered.

        Example:
            volumetric_mask = cbu.get_data_file(f"templates/standard/MNI152/MNI152_T1_2mm_brain.nii.gz")
            mask_surface = my_brain_viewer.visualize_mask_surface(
                volumetric_mask,
                threshold = 4000,
            )
        """
        # Load the volumetric mask
        volumetric_mask = cbu.Volumetric_data(volumetric_mask)

        # Ensure this is a mask
        volumetric_mask = volumetric_mask.mask(threshold)

        # Get the voxels inside mask
        selected_voxels = np.array(np.where(volumetric_mask.data)).T

        # use a marching cube algorithm with smoothing to generate a surface model
        (
            surface_vertices,
            surface_triangles,
        ) = cbu.generate_surface_marching_cube(
            selected_voxels, volumetric_mask.affine, **kwargs
        )

        # generate a unique id for the object
        unique_id = f"{utils.generate_unique_id()}"
        object_id = f"surface_mesh#{unique_id}"
        self.created_objects[object_id] = self._create_surface_mesh_object(
            object_id=object_id,
            vertices=surface_vertices,
            triangles=surface_triangles,
            color=self._prepare_color(color),
            object_offset_coordinate=coordinate_offset,
        )

        # draw to update visualization
        self.draw()

        # return object to user
        return self.created_objects[object_id]

    def data_to_colors(
        self,
        data,
        colormap=None,
        clims=None,
        vlims=None,
        invert=False,
        opacity=1,
        exclusion_color=None,
        scale=None,
        dscalar_index=0,
    ):
        """Convert data values to RGBA colors based on the provided colormap and normalization options.

        Args:
            data (ndarray): The data values to convert to colors.
            colormap (str or Colormap, optional): The name of the colormap to use for color mapping.
                If not provided, the default colormap will be used.
            clims (tuple, optional): Custom color limits for data normalization.
                If not provided, the minimum and maximum non-excluded data values will be used.
            vlims (tuple, optional): Exclusion limits for data values.
                Values outside this range will be excluded from the color mapping.
                If invert is True, values inside this range will be excluded.
            invert (bool, optional): If True, the exclusion criteria will be inverted.
            opacity (float, optional): Opacity value for the generated colors (0 to 1).
            exclusion_color (tuple, optional): RGBA color for excluded data points.
            scale (str, optional): Scale option for data normalization. Supported values are 'log' or None.
                If 'log', data will be log-scaled (log2(1 + data)) before normalization.
            dscalar_index (int, optional): If the data represents a dscalar file, the index of the dscalar map.

        Returns:
            ndarray: An array of RGBA colors representing the input data.

        Example:
            data = np.array([0.5, 0.8, 0.2, 1.0, np.nan, 0.3, -0.1, np.inf, -np.inf])
            colors = my_brain_viewer.data_to_colors(
                data,
                colormap="viridis",
                clims=(-1.0, 1.0),
                vlims=(0.1, 0.9),
                invert=False,
                opacity=0.8,
                exclusion_color=(0.5, 0.5, 0.5, 1.0),
            )
        """
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
            normalized_data = data - cmin
            if cmin != cmax:
                normalized_data = normalized_data / (cmax - cmin)
        else:
            normalized_data = data - data[~exclude].min()
            if data[~exclude].min() != data[~exclude].max():
                normalized_data = normalized_data / (
                    data[~exclude].max() - data[~exclude].min()
                )

        # apply log-scale normalization if requested
        if scale == "log":
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
        """Compute overlay colors by combining two sets of colors.

        The function takes two sets of colors with alpha transparency and computes the
        resulting overlay colors by blending the top colors over the bottom colors.

        Args:
            bottom_colors (ndarray): An array of colors with alpha transparency, shape (N, 4),
                                    representing the bottom layer.
            top_colors (ndarray): An array of colors with alpha transparency, shape (N, 4),
                                representing the top layer.

        Returns:
            ndarray: An array of blended overlay colors with alpha transparency, shape (N, 4).

        Example:
            bottom_colors = np.array([[0.7, 0.3, 0.2, 0.8], [0.5, 0.2, 0.9, 0.7]])
            top_colors = np.array([[0.9, 0.1, 0.3, 0.6], [0.3, 0.6, 0.1, 0.4]])
            overlay_colors = my_brain_viewer.compute_overlay_colors(bottom_colors, top_colors)
        """
        # convert to 0-1 range
        overlay_colors = bottom_colors * 0

        # rename for simplicity
        bottom_alpha = bottom_colors[:, 3].reshape(-1, 1)
        bottom_color = bottom_colors[:, :3]
        top_alpha = top_colors[:, 3].reshape(-1, 1)
        top_color = top_colors[:, :3]

        # compute opacity
        overlay_colors[:, 3] = (
            top_alpha + np.multiply(bottom_alpha, (1 - top_alpha))
        ).reshape(-1)

        # compute color
        overlay_colors[:, :3] = np.multiply(top_color, top_alpha) + np.multiply(
            bottom_color, (1 - top_alpha)
        )

        return overlay_colors

    def add_cifti_dscalar_layer(
        self,
        cifti_space_id=None,
        dscalar_file=None,
        loaded_dscalar=None,
        dscalar_data=None,
        dscalar_index=0,
        **kwargs,
    ):
        """Add a CIFTI dscalar layer to the specified CIFTI space.

        This function allows you to add a new dscalar layer to an existing CIFTI space.
        You can provide the dscalar data directly, or load it from a file (either in NIfTI
        format or using a loaded nibabel CIFTI object). The data will be converted to colors
        using the data_to_colors method before adding the layer.

        Args:
            cifti_space_id (str, optional): The ID of the CIFTI space to which the layer
                will be added. If not provided, the default loaded CIFTI space will be used.
            dscalar_file (str, optional): The path to the dscalar file from which the data
                will be loaded. Required if 'loaded_dscalar' and 'dscalar_data' are not provided.
            loaded_dscalar (nibabel.Cifti2Image, optional): A loaded nibabel CIFTI object
                containing the dscalar data. Required if 'dscalar_file' and 'dscalar_data'
                are not provided.
            dscalar_data (ndarray, optional): The data array for the dscalar layer. Required
                if 'dscalar_file' and 'loaded_dscalar' are not provided.
            dscalar_index (int, optional): The index of the dscalar data if the file or object
                contains multiple datasets. Default is 0.
            **kwargs: Additional keyword arguments that will be passed to the data_to_colors
                method for converting the dscalar data to colors.

        Returns:
            dict: A dictionary containing information about the created CIFTI dscalar layer.

        Raises:
            Exception: If no dscalar data is provided.

        Example:
            dscalar_data = np.random.rand(100)
            cifti_space_id = "cifti_space#unique_id"
            dscalar_layer = my_brain_viewer.add_cifti_dscalar_layer(
                cifti_space_id,
                dscalar_data=dscalar_data,
                colormap="coolwarm",
                clims=(0, 1),
                opacity=0.7,
            )
        """
        if cifti_space_id is None:
            # use default loaded cifti space
            cifti_space_id = self.default_objects["cifti_space"]

        # initialization
        unique_id = f"{utils.generate_unique_id()}"
        layer_type = "cifti_dscalar_layer"
        layer_id = f"{layer_type}#{unique_id}"

        # load the cifti dscalar file
        if dscalar_file is not None:
            dscalar = self.file_handler.load_file(dscalar_file, nib.load)
            dscalar_data = dscalar.get_fdata()[dscalar_index]
        elif loaded_dscalar is not None:
            dscalar_data = loaded_dscalar.get_fdata()[dscalar_index]
        elif dscalar_data is None:
            raise Exception(f"No dscalar was provided for add_CIFTI_dscalar_layer.")

        # convert data to colors
        dscalar_colors = self.data_to_colors(dscalar_data, **kwargs)

        # load the cifti_template
        cifti_space = self.created_objects[cifti_space_id]
        layer_order = len(cifti_space["layers"])

        # store layer
        created_layer = {
            "layer_id": layer_id,
            "layer_type": layer_type,
            "layer_order": layer_order,
            "visibility": True,
            "dscalar_data": dscalar_data,
            "layer_colors": dscalar_colors,
            "cifti_space_id": cifti_space_id,
            "layer_update_required": True,
        }
        self.created_layers[layer_id] = created_layer
        cifti_space["layers"][layer_order] = layer_id

        # draw to update visualization
        self.draw()

        return created_layer

    # TODO: modify function is not working correctly...
    # def modify_cifti_dscalar_layer(self, created_layer, dscalar_file=None, loaded_dscalar=None, dscalar_data=None, dscalar_index=0, **kwargs):
    #     # load the cifti dscalar file
    #     if dscalar_file is not None:
    #         dscalar = self.file_handler.load_file(dscalar_file, nib.load)
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
        """Update a CIFTI dscalar layer in the brain visualization.

        This function updates a CIFTI dscalar layer by adding it to the corresponding CIFTI space object.
        It sets the necessary flags to trigger the render update for the objects associated with the layer.

        Args:
            layer_id (str): The unique identifier of the CIFTI dscalar layer to be updated.

        Returns:
            None.
        """
        cifti_space_id = self.created_layers[layer_id]["cifti_space_id"]
        cifti_space = self.created_objects[cifti_space_id]
        layer_idx = len(cifti_space["layers"])
        cifti_space["layers"][layer_idx] = layer_id
        for object_id in cifti_space["contained_object_ids"]:
            self.created_objects[object_id]["render_update_required"] = True
            self.created_objects[object_id]["layers"] = cifti_space["layers"]

        # update flag
        self.created_layers[layer_id]["layer_update_required"] = False

    def update_layer(self, layer_id):
        """Update a layer of the brain visualization.

        This function checks the type of the specified layer and updates it accordingly. Currently, it supports
        updating "cifti_dscalar_layer" type layers. It retrieves the necessary data and calls the appropriate
        function to perform the update.

        Args:
            layer_id (str): The unique identifier of the layer to be updated.

        Returns:
            None.
        """
        if self.created_layers[layer_id]["layer_type"] == "cifti_dscalar_layer":
            self.update_cifti_dscalar_layer(layer_id)

    def update_layers(self):
        """Update all layers in the brain visualization.

        This function iterates through all the layers in the brain visualization and checks if an update is required for each layer.
        If a layer requires an update, it calls the 'update_layer' function to perform the update for that specific layer.

        Returns:
            None
        """
        for layer_id in self.created_layers:
            if self.created_layers[layer_id]["layer_update_required"]:
                self.update_layer(layer_id)

    def get_object_base_colors_for_render(self, object_id, size):
        """Get the base colors for rendering an object in the brain visualization.

        This function retrieves the base colors for a specific object identified by 'object_id' in the brain visualization.
        It loads the object's color information from the internal data structure.
        The base colors are reshaped if necessary to ensure they have the appropriate shape for rendering.
        If the object has a single base color, it is expanded to a fixed color for all elements with the specified 'size'.
        The final base colors are returned as an array.

        Args:
            object_id (str): The unique identifier of the object for which to obtain base colors.
            size (int): The size of the object, used for rendering.

        Returns:
            ndarray: An array containing the base colors for rendering the object.

        Raises:
            AssertionError: If the provided base colors cannot be unpacked appropriately.
        """
        # load the object
        colored_object = self.created_objects[object_id]

        # load base colors and reshape if required
        base_color = colored_object.get("base_color", self.null_color)
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
        """Apply layer colors to the rendering colors of an object in the brain visualization.

        This function applies layer colors to the rendering colors of a specific object identified by 'object_id' in the brain visualization.
        It loads the object's color and layer information from the internal data structure.
        For each layer associated with the object, it extracts the colors from the layer and assigns them to the corresponding indices in the object's rendering colors.
        The function also considers the data mapping and indices specified in the object's data, if applicable.
        Finally, it computes the overlay of layer colors and returns the updated rendering colors.

        Args:
            object_id (str): The unique identifier of the object for which to apply layer colors.
            size (int): The size of the object, used for rendering.
            colors (ndarray): The base rendering colors of the object.

        Returns:
            ndarray: An array containing the updated rendering colors with applied layer colors.
        """
        # load the object
        colored_object = self.created_objects[object_id]

        # add layers one by one
        for layer_idx in range(len(colored_object["layers"])):
            layer_id = colored_object["layers"][layer_idx]
            layer_object = self.created_layers[layer_id]

            # extract colors from layer
            index_offset = colored_object["data_index_offset"]
            index_count = colored_object["data_index_count"]
            extracted_colors = layer_object["layer_colors"][
                index_offset : (index_offset + index_count)
            ]

            # check colors have expected shape
            if extracted_colors.shape[0] == index_count:
                # the extracted colors may need to be resampled by a map
                if colored_object.get("data_map", None) is not None:
                    extracted_colors = extracted_colors[colored_object["data_map"]]

                # the extracted colors need to be assigned according to indices
                layer_colors = np.array(self.no_color)[np.newaxis, :].repeat(size, 0)
                if colored_object.get("data_indices", None) is not None:
                    layer_colors[colored_object["data_indices"]] = extracted_colors
                else:
                    layer_colors = extracted_colors

                # compute the layer overlay
                colors = self.compute_overlay_colors(colors, layer_colors)

        return colors

    def get_object_render_colors(self, object_id, size):
        """Get the rendering colors for an object in the brain visualization.

        This function retrieves the rendering colors for a specific object identified by 'object_id' in the brain visualization.
        It first obtains the base colors for the object using 'get_object_base_colors_for_render' function.
        Then, it applies layer colors to the base colors using 'apply_layer_colors_for_render' function, if applicable.
        The resulting colors are returned as an array.

        Args:
            object_id (str): The unique identifier of the object for which to obtain rendering colors.
            size (int): The size of the object, used for rendering.

        Returns:
            ndarray: An array containing the rendering colors for the object.
        """
        colors = self.get_object_base_colors_for_render(object_id, size)
        colors = self.apply_layer_colors_for_render(object_id, size, colors)
        return colors

    def render_surface_mesh(self, object_id):
        """Render a surface mesh object in the brain visualization.

        This function renders a surface mesh object identified by 'object_id' in the brain visualization.
        It loads the vertices and triangles data of the surface mesh from the internal data structure.
        The function applies any necessary changes in coordinates by considering the object's offset.
        The appropriate render colors for the mesh are obtained using the 'get_object_render_colors' function.
        The existing render, if any, is cleared, and the object is rendered using the calculated vertex colors.
        The object's rendered state and boundaries are updated accordingly.

        Args:
            object_id (str): The unique identifier of the surface mesh object to be rendered.

        Returns:
            None.
        """
        # load vertices and triangles
        surface_mesh_object = self.created_objects[object_id]
        surface_vertices = surface_mesh_object["vertices"]
        surface_triangles = surface_mesh_object["triangles"]

        # apply necessary changes in coordinates by the offset
        surface_vertices = surface_vertices + surface_mesh_object.get(
            "object_offset_coordinate", 0
        )

        # load appropriate render colors
        surface_colors = self.get_object_render_colors(
            object_id, surface_vertices.shape[0]
        )

        # clear existing render
        if surface_mesh_object["rendered"]:
            self.viewer.clear_object(surface_mesh_object["rendered_mesh"]["node_name"])
            surface_mesh_object.pop("rendered_mesh")
            surface_mesh_object["rendered"] = False

        # render the object
        surface_mesh_object["vertex_colors"] = surface_colors
        rendered_mesh = self.viewer.add_mesh(
            surface_vertices, surface_triangles, surface_colors
        )
        surface_mesh_object["rendered_mesh"] = rendered_mesh
        surface_mesh_object["rendered"] = True

        # update object boundaries
        self.min_coordinate = np.min([self.min_coordinate, surface_vertices.min(0)], 0)
        self.max_coordinate = np.max([self.max_coordinate, surface_vertices.max(0)], 0)

        # signal that render was updated
        self.created_objects[object_id]["render_update_required"] = False

    def render_spheres(self, object_id):
        """Render spheres in the brain visualization.

        This function renders spheres identified by 'object_id' in the brain visualization.
        It loads the coordinates and radii data of the spheres from the internal data structure.
        The function applies any necessary changes in coordinates by considering the object's offset.
        The appropriate render colors for the spheres are obtained using the 'get_object_render_colors' function.
        The existing render, if any, is cleared, and the spheres are rendered using the calculated colors.
        The object's rendered state and boundaries are updated accordingly.

        Args:
            object_id (str): The unique identifier of the spheres object to be rendered.

        Returns:
            None.
        """
        # load vertices and triangles
        spheres_object = self.created_objects[object_id]
        coordinates = spheres_object["coordinates"]
        radii = spheres_object["radii"]

        # apply necessary changes in coordinates by the offset
        coordinates = coordinates + spheres_object.get("object_offset_coordinate", 0)

        # load appropriate render colors
        colors = self.get_object_render_colors(object_id, coordinates.shape[0])

        # clear existing render

        # render the object
        spheres_object["colors"] = colors
        rendered_spheres = self.viewer.add_points(coordinates, radii, colors)
        spheres_object["rendered_spheres"] = rendered_spheres
        spheres_object["rendered"] = True

        # update object boundaries
        self.min_coordinate = np.min(
            [self.min_coordinate, (coordinates - radii).min(0)], 0
        )
        self.max_coordinate = np.max(
            [self.max_coordinate, (coordinates + radii).max(0)], 0
        )

        # signal that render was updated
        self.created_objects[object_id]["render_update_required"] = False

    def render_cylinders(self, object_id):
        """Render cylinders in the brain visualization.

        This function renders cylinders identified by 'object_id' in the brain visualization.
        It loads the coordinates and radii data of the cylinders from the internal data structure.
        The function applies any necessary changes in coordinates by considering the object's offset.
        The appropriate render colors for the cylinders are obtained using the 'get_object_render_colors' function.
        The existing render, if any, is cleared, and the cylinders are rendered using the calculated colors.
        The object's rendered state and boundaries are updated accordingly.

        Args:
            object_id (str): The unique identifier of the cylinders object to be rendered.

        Returns:
            None.
        """
        # load vertices and triangles
        cylinders_object = self.created_objects[object_id]
        coordinates = cylinders_object["coordinates"]
        radii = cylinders_object["radii"]

        # apply necessary changes in coordinates by the offset
        coordinates = coordinates + cylinders_object.get("object_offset_coordinate", 0)

        # load appropriate render colors
        colors = self.get_object_render_colors(object_id, coordinates.shape[0])

        # clear existing render

        # render the object
        cylinders_object["colors"] = colors
        rendered_cylinders = self.viewer.add_lines(coordinates, radii, colors)
        cylinders_object["rendered_cylinders"] = rendered_cylinders
        cylinders_object["rendered"] = True

        # update object boundaries
        self.min_coordinate = np.min(
            [self.min_coordinate, (coordinates.min(0).min(0) - radii.min())], 0
        )
        self.max_coordinate = np.max(
            [self.max_coordinate, (coordinates.max(0).max(0) + radii.max())], 0
        )

        # signal that render was updated
        self.created_objects[object_id]["render_update_required"] = False

    def render_object(self, object_id):
        """Render a specific object in the brain visualization.

        This function renders the object specified by 'object_id' in the brain visualization.
        The type of the object (surface mesh, spheres, or cylinders) is determined from the object's metadata.
        Depending on the object type, the corresponding 'render_*' function is called to perform the rendering.

        Args:
            object_id (str): The unique identifier of the object to be rendered.

        Returns:
            None.
        """
        if self.created_objects[object_id]["object_type"] == "surface_mesh":
            self.render_surface_mesh(object_id)
        elif self.created_objects[object_id]["object_type"] == "spheres":
            self.render_spheres(object_id)
        elif self.created_objects[object_id]["object_type"] == "cylinders":
            self.render_cylinders(object_id)

    def render_update(self):
        """Update the rendered objects in the brain visualization.

        This function iterates through all created objects and checks if their rendering needs to be updated
        based on the 'render_update_required' flag in their metadata. If an object requires rendering update,
        the corresponding 'render_object' function is called to re-render the object with any changes.
        After updating all objects, the camera is centered to ensure they are visible within the view,
        and a garbage collection is performed to release any unused resources.

        Args:
            None.

        Returns:
            None.
        """
        for object_id in self.created_objects:
            if self.created_objects[object_id].get("render_update_required", False):
                self.render_object(object_id)
        self.center_camera()
        utils.garbage_collect()

    def draw(self):
        """Draw the brain visualization.

        This function is responsible for updating any required renders by calling 'update_layers' and 'render_update'
        functions. The 'update_layers' function checks and updates individual layers of the visualization, and the
        'render_update' function updates the rendered objects based on the 'render_update_required' flag in their metadata.
        After updating the layers and rendered objects, the function calls the 'draw' method of the viewer window to display
        the visualization interactively.

        Args:
            None.

        Returns:
            None.
        """
        # update any required renders
        self.update_layers()
        self.render_update()

        # run the viewer window
        self.viewer.draw()

    def show(self):
        """Show the brain visualization in an interactive window.

        This function updates any required renders by calling 'update_layers' and 'render_update' functions,
        which ensures that the brain visualization is up to date with any changes made to its objects. After updating
        the layers and rendered objects, the function calls the 'show' method of the viewer window, allowing the
        brain visualization to be displayed interactively.

        Args:
            None.

        Returns:
            None.
        """
        # update any required renders
        self.update_layers()
        self.render_update()

        # run the viewer window
        self.viewer.show()

    def offscreen_draw_to_matplotlib_axes(self, ax):
        """Draw an offscreen-rendered view to a matplotlib axes.

        This function allows the offscreen-rendered view from the viewer window to be drawn into a matplotlib
        axes object. Note that this functionality is experimental and might not fully work depending on
        viewer configuration.

        Args:
            ax (matplotlib.Axes): The matplotlib axes into which the offscreen-rendered view will be drawn.

        Returns:
            None.
        """
        self.viewer.window.offscreen_draw_to_matplotlib_axes(ax)
