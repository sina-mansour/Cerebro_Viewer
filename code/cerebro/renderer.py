"""
This module contains the code to handle rendering requests.

The aim is to add multiple render engines to choose from, this provides a renderer
independent approach which could be later upgraded to work with other rendering
software.

Here are some default rendering capabilities that should be implemented for
every engine added:
- 3D Surface mesh
- 3D Point
- 3D Line
- 2D Text
- 3D Network

The default (read most extensively developed) renderer for cerebro viewer uses
th Panda3d game engine. Nevertheless, other render engines can be added in
future development.

Notes
-----
Author: Sina Mansour L.
"""

from abc import ABC, abstractmethod
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


class Renderer(ABC):
    """
    Abstract base class for all renderers.

    An abstract base class for all renderer implementations to ensure that the
    basic methods are included for all renderers. Essentially, any future renderer
    class needs to provide the functions listed here.
    """

    @abstractmethod
    def add_mesh(self, vertices, triangles, colors):
        """
        A method to add a surface mesh to the rendered objects.

        Meshes can be colored according to a list of RGBA values.
        """
        pass

    @abstractmethod
    def add_points(self, coordinates, radii, colors):
        """
        A method to add points (spheres) to the rendered objects.
        """
        pass

    @abstractmethod
    def add_lines(self, coordinates, radii, colors):
        """
        A method to add a single point to the rendered objects.
        """
        pass

    @abstractmethod
    def show(self):
        """
        A method to display all rendered objects.
        """
        pass

    @abstractmethod
    def clear_all(self):
        """
        A method to clear all rendered objects.
        """
        pass


class Renderer_panda3d(Renderer):
    """
    Panda3D renderer engine

    This is a render engine that uses panda3d for visualizations. This renderer is
    not integrated with a jupyter notebook/lab environment. It instead creates
    a separate visualization dialog (X11) when executed locally.
    """

    def __init__(self, **kwargs):
        # import ipygany engine only when the renderer class is instantiated
        from .cerebro_window import Cerebro_window

        # view angle can be provided to renderer
        self.window = Cerebro_window(**kwargs)

    def __del__(self):
        self.window.destroy()

    def add_mesh(self, vertices, triangles, colors):
        """
        A method to add a surface mesh to the rendered objects.

        Meshes can be colored according to a list of RGBA values.
        """
        return self.window.create_surface_mesh(
            vertices,
            triangles,
            vertex_colors=colors,
        )

    def add_points(self, coordinates, radii, colors):
        """
        A method to add points (spheres) to the rendered objects.
        """
        return self.window.create_multiple_sphere_instances(
            coordinates,
            radii,
            colors
        )

    def add_lines(self, coordinates, radii, colors):
        """
        A method to add a single point to the rendered objects.
        """
        return self.window.create_multiple_cylinder_instances(
            coordinates,
            radii,
            colors
        )

    def show(self):
        """
        A method to display all rendered objects.
        """
        self.window.run()

    def draw(self):
        """
        A method to update the display without starting the show loop.
        """
        self.window.draw()

    def clear_all(self):
        """
        A method to clear all rendered objects.
        """
        self.window.clear_all_created_objects()

    def clear_object(self, object_id):
        """
        A method to clear all rendered objects.
        """
        self.window.clear_created_object(object_id)

    def change_view(self, **kwargs):
        """
        A method to update camera configuration.
        """
        self.window.update_camera(**kwargs)

    def get_view(self, **kwargs):
        """
        A method to provide camera configuration.
        """
        return self.window.get_camera_view()
