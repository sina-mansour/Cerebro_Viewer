"""
This module contains the code to create a viewer window in panda3d.

The viewer window will contain the rendered scene, it may potentially contain
multiple viewpoints/sub-windows.

Notes
-----
Author: Sina Mansour L.
"""

import numpy as np
import array
import trimesh

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import WindowProperties, ModifierButtons, ClockObject
from panda3d.core import GeomVertexFormat, GeomVertexArrayFormat, Geom
from panda3d.core import GeomVertexData, GeomTriangles, GeomEnums, GeomNode
from panda3d.core import RenderState, TransparencyAttrib, NodePath
from panda3d.core import load_prc_file_data

from . import cerebro_utils as utils


# configurations
load_prc_file_data('', 'win-size 1280 720')
# load_prc_file_data('', 'window-title Cerebro Viewer')
# load_prc_file_data('', 'icon-filename Cerebro_Viewer.ico')
# load_prc_file_data("","framebuffer-multisample True")
# load_prc_file_data("","multisamples 16")
# load_prc_file_data('', 'background-color 0.1 0.1 0.1 0.0')
# load_prc_file_data('', 'show-scene-graph-analyzer True')


class Cerebro_window(ShowBase):

    def __init__(self, background_color=(0.1, 0.1, 0.1, 0.0), camera_pos=(400, 0, 0), camera_target=(0, 0, 0), camera_fov=35, camera_rotation=0, rotation_speed=100, movement_speed=100, zoom_speed=0.2, window_size=(1280, 720)):
        super().__init__(self)

        # Initial configurations
        self.set_background_color(*list(background_color))
        # self.set_scene_graph_analyzer_meter(True)
        # self.set_frame_rate_meter(True)
        # Window properties
        # Ref: https://docs.panda3d.org/1.11/cpp/reference/panda3d.core.WindowProperties
        window_properties = WindowProperties()
        window_properties.set_size(window_size[0], window_size[1])
        window_properties.set_title('Cerebro Viewer')
        window_properties.set_icon_filename('Cerebro_Viewer.ico')
        self.win.requestProperties(window_properties)

        # Keyboard and mouse setup
        self.setup_keyboard_and_mouse()

        # Camera positioning
        # position is relative to target
        self.cam_init_x, self.cam_init_y, self.cam_init_z = list(camera_pos)
        self.cam_target_x, self.cam_target_y, self.cam_target_z = list(camera_target)
        self.camera_fov = camera_fov
        self.camera_rotation = camera_rotation
        self.setup_camera()

        # Update speed
        self.rotation_speed = rotation_speed
        self.movement_speed = movement_speed
        self.zoom_speed = zoom_speed

        # Add the update task to the task manager.
        self.taskMgr.add(self.update_task, "update_task")

        # Add the repeated checking task to the task manager
        self.taskMgr.doMethodLater(0.05, self.repeated_checks_task, "repeated_checks_task")

        # Create a dictionary for rendered objects
        self.created_objects = {}

        # A flag to check if re-rendering by reordering of faces is required
        self.face_reordering_required = False

        self.create_object_templates()

    # Keyboard and mouse setup procedure
    def setup_keyboard_and_mouse(self):
        self.key_map = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'shift': False,
            'control': False,
            'o': False,
            'i': False,
            'left-click': False,
            'right-click': False,
            'middle-click': False,
            'wheel': 0,
        }

        # Disable the default camera trackball controls.
        self.disableMouse()

        # Initialize mouse positions
        self.mouse_x = 0.5
        self.mouse_y = 0.5
        self.mouse_x_previous = self.mouse_x
        self.mouse_y_previous = self.mouse_y

        # Treat modifier keys as normal keys
        self.mouseWatcherNode.set_modifier_buttons(ModifierButtons())
        self.buttonThrowers[0].node().set_modifier_buttons(ModifierButtons())

        # Handle key press
        self.accept('arrow_up', self.update_key_map, ['up', True])
        self.accept('arrow_down', self.update_key_map, ['down', True])
        self.accept('arrow_right', self.update_key_map, ['right', True])
        self.accept('arrow_left', self.update_key_map, ['left', True])
        self.accept('shift', self.update_key_map, ['shift', True])
        self.accept('control', self.update_key_map, ['control', True])
        self.accept('o', self.update_key_map, ['o', True])
        self.accept('i', self.update_key_map, ['i', True])
        self.accept('mouse1', self.update_key_map, ['left-click', True])
        self.accept('mouse2', self.update_key_map, ['middle-click', True])
        self.accept('mouse3', self.update_key_map, ['right-click', True])
        self.accept('wheel_up', self.update_key_map, ['wheel', (self.key_map['wheel'] + 1)])
        self.accept('wheel_down', self.update_key_map, ['wheel', (self.key_map['wheel'] - 1)])

        # Handle key release
        self.accept('arrow_up-up', self.update_key_map, ['up', False])
        self.accept('arrow_down-up', self.update_key_map, ['down', False])
        self.accept('arrow_right-up', self.update_key_map, ['right', False])
        self.accept('arrow_left-up', self.update_key_map, ['left', False])
        self.accept('shift-up', self.update_key_map, ['shift', False])
        self.accept('control-up', self.update_key_map, ['control', False])
        self.accept('o-up', self.update_key_map, ['o', False])
        self.accept('i-up', self.update_key_map, ['i', False])
        self.accept('mouse1-up', self.update_key_map, ['left-click', False])
        self.accept('mouse2-up', self.update_key_map, ['middle-click', False])
        self.accept('mouse3-up', self.update_key_map, ['right-click', False])

    # Define a function to update the keymap
    def update_key_map(self, key, state):
        self.key_map[key] = state

    # Camera setup procedure
    def setup_camera(self):
        # Create a camera pivot node at the center + two other pivot nodes to do rotation along other axes
        # Place the camera looking at the center
        dist = np.sqrt((self.cam_init_x ** 2) + (self.cam_init_y ** 2) + (self.cam_init_z ** 2))

        # Center pivot direction
        self.camera_pivot_node = self.render.attachNewNode("camera-pivot")
        self.camera_pivot_node.setPos(0, 0, 0)
        self.camera_pivot_node.lookAt(dist, 0, 0)

        # rotation pivots
        self.camera_pivot_node_rl = self.render.attachNewNode("camera-pivot-rl-rotation")
        self.camera_pivot_node_rl.setPos(0, 0, dist)
        self.camera_pivot_node_rl.lookAt(0, 0, 0)
        self.camera_pivot_node_rl.wrtReparentTo(self.camera_pivot_node)
        self.camera_pivot_node_ud = self.render.attachNewNode("camera-pivot-ud-rotation")
        self.camera_pivot_node_ud.setPos(0, dist, 0)
        self.camera_pivot_node_ud.lookAt(0, 0, 0)
        self.camera_pivot_node_ud.wrtReparentTo(self.camera_pivot_node)

        # Camera direction
        self.camera.setPos(dist, 0, 0)
        self.camera.lookAt(0, 0, 0)
        self.camera.wrtReparentTo(self.camera_pivot_node)

        # Camera position to look at target (via pivot node reference)
        self.camera_pivot_node.setPos(self.cam_target_x, self.cam_target_y, self.cam_target_z)
        self.camera_pivot_node.lookAt(self.cam_init_x, self.cam_init_y, self.cam_init_z)
        self.camera_pivot_node.setR(self.camera_rotation)
        self.cam.node().getLens().setFov(self.camera_fov)
        # self.camera_direction = np.array(self.camera.get_quat())[1:]
        self.camera_direction = np.array(self.camera.get_pos(self.render))

    # Update camera direction
    def update_camera(self, camera_target=None, camera_pos=None, camera_rotation=None, camera_fov=None):
        if camera_pos is not None:
            (self.cam_init_x, self.cam_init_y, self.cam_init_z) = camera_pos
        if camera_target is not None:
            (self.cam_target_x, self.cam_target_y, self.cam_target_z) = camera_target
        if camera_rotation is not None:
            self.camera_rotation = camera_rotation
        if camera_fov is not None:
            self.camera_fov = camera_fov
        self.camera_pivot_node.setPos(
            self.cam_target_x,
            self.cam_target_y,
            self.cam_target_z
        )
        self.camera_pivot_node.lookAt(self.cam_init_x, self.cam_init_y, self.cam_init_z)
        self.camera_pivot_node.setR(self.camera_rotation)
        self.cam.node().getLens().setFov(self.camera_fov)
        self.camera_direction = np.array(self.camera.get_pos(self.render))
        self.reorder_faces_of_all_objects()

    # Define a procedure to update the camera direction every frame.
    def update_task(self, task):
        # Check that mouse is in window
        if self.mouseWatcherNode.hasMouse():
            # get mouse postion
            self.mouse_x = self.mouseWatcherNode.getMouseX()
            self.mouse_y = self.mouseWatcherNode.getMouseY()

        # A frame-based time multiplier
        dt = ClockObject.getGlobalClock().getDt()

        ud_rotation = 0
        rl_rotation = 0
        ud_position = 0
        rl_position = 0
        zoom_factor = 0

        # if shift is pressed
        if self.key_map['shift']:
            if self.key_map['up']:
                ud_position += dt * self.movement_speed
            if self.key_map['down']:
                ud_position -= dt * self.movement_speed
            if self.key_map['right']:
                rl_position += dt * self.movement_speed
            if self.key_map['left']:
                rl_position -= dt * self.movement_speed
            if self.key_map['left-click']:
                rl_position += (self.mouse_x - self.mouse_x_previous) * self.movement_speed
                ud_position += (self.mouse_y - self.mouse_y_previous) * self.movement_speed
        # if control is pressed
        elif self.key_map['control']:
            if self.key_map['up']:
                zoom_factor -= dt * self.zoom_speed
            if self.key_map['down']:
                zoom_factor += dt * self.zoom_speed
            if self.key_map['left-click']:
                zoom_factor += (self.mouse_y - self.mouse_y_previous) * self.zoom_speed
        # otherwise
        else:
            if self.key_map['up']:
                ud_rotation -= dt * self.rotation_speed
            if self.key_map['down']:
                ud_rotation += dt * self.rotation_speed
            if self.key_map['right']:
                rl_rotation += dt * self.rotation_speed
            if self.key_map['left']:
                rl_rotation -= dt * self.rotation_speed
            if self.key_map['left-click']:
                rl_rotation += (self.mouse_x - self.mouse_x_previous) * self.rotation_speed
                ud_rotation -= (self.mouse_y - self.mouse_y_previous) * self.rotation_speed

        if self.key_map['o']:
            zoom_factor += dt * self.zoom_speed
        if self.key_map['i']:
            zoom_factor -= dt * self.zoom_speed
        if self.key_map['wheel'] != 0:
            zoom_factor += dt * self.key_map['wheel']
            self.key_map['wheel'] = 0

        # Perform right-left rotation and up-down movement
        if (rl_rotation != 0) or (ud_position != 0):
            self.camera_rotation_node_rl = self.render.attachNewNode("camera-pivot-rl-rotation")
            self.camera_rotation_node_rl.setPos(self.camera_pivot_node_rl.getPos(self.render))
            self.camera_rotation_node_rl.setHpr(self.camera_pivot_node_rl.getHpr(self.render))
            self.camera_pivot_node.wrtReparentTo(self.camera_rotation_node_rl)
            current_rl_rotation = self.camera_rotation_node_rl.getR()
            self.camera_rotation_node_rl.setR(current_rl_rotation + rl_rotation)
            self.camera_pivot_node.setY(self.camera_pivot_node.getY() + ud_position)
            self.camera_pivot_node.wrtReparentTo(self.render)
            self.camera_rotation_node_rl.removeNode()

        # Perform up-down rotation and right-left movement
        if (ud_rotation != 0) or (rl_position != 0):
            self.camera_rotation_node_ud = self.render.attachNewNode("camera-pivot-ud-rotation")
            self.camera_rotation_node_ud.setPos(self.camera_pivot_node_ud.getPos(self.render))
            self.camera_rotation_node_ud.setHpr(self.camera_pivot_node_ud.getHpr(self.render))
            self.camera_pivot_node.wrtReparentTo(self.camera_rotation_node_ud)
            current_ud_rotation = self.camera_rotation_node_ud.getR()
            self.camera_rotation_node_ud.setR(current_ud_rotation + ud_rotation)
            self.camera_pivot_node.setY(self.camera_pivot_node.getY() + rl_position)
            self.camera_pivot_node.wrtReparentTo(self.render)
            self.camera_rotation_node_ud.removeNode()

        # Perform zoom
        if (zoom_factor != 0):
            current_pos = self.camera.getPos()
            self.camera.setPos(
                current_pos[0] * np.exp(zoom_factor),
                current_pos[1] * np.exp(zoom_factor),
                current_pos[2] * np.exp(zoom_factor),
            )

        # Update previous mouse positions
        self.mouse_x_previous = self.mouse_x
        self.mouse_y_previous = self.mouse_y

        # Continue
        return Task.cont

    # Define a procedure to get a screenshot of the view.
    def get_screenshot(self, output_file):
        self.win.save_screenshot(output_file)

    # Define a function to depth sort the faces
    def sort_faces(self, direction, triangles, vertices):
        # reorder triangles according to a direction vector
        return triangles[np.argsort(vertices[triangles].mean(1).dot(direction))]

    # Define a function to create a node object for a surface mesh
    def create_surface_mesh_node(self, vertices, triangles, node_name, vertex_colors=None, direction=None):
        # convert information into array format
        coords = array.array("f", vertices.reshape(-1))
        if vertex_colors is not None:
            colors = array.array("B", (255 * vertex_colors.reshape(-1)).astype(np.uint8))
        if direction is None:
            faces_array = array.array("I", triangles.reshape(-1))
        else:
            faces_array = array.array("I", self.sort_faces(direction, triangles, vertices).reshape(-1))

        # specify a generic vertex format
        vertex_format = GeomVertexFormat()
        # add 3d coordinates to the format
        vertex_format.add_array(GeomVertexFormat.get_v3().arrays[0])
        # add color information to the format if requested
        if vertex_colors is not None:
            vertex_format.add_array(GeomVertexArrayFormat('color', 4, Geom.NT_uint8, Geom.C_color))
        # register format
        vertex_format = GeomVertexFormat.register_format(vertex_format)

        # create data using this format
        vertex_data = GeomVertexData("vertex_data", vertex_format, Geom.UH_static)
        # set the number of vertices according to the surface mesh
        vertex_data.unclean_set_num_rows(vertices.shape[0])

        # get memoryview handles to data and fill by the available information
        vertex_array = vertex_data.modify_array(0)
        vertex_memview = memoryview(vertex_array).cast("B").cast("f")
        vertex_memview[:] = coords
        if vertex_colors is not None:
            color_array = vertex_data.modify_array(1)
            color_memview = memoryview(color_array).cast("B")
            color_memview[:] = colors

        # store face indices
        tris_prim = GeomTriangles(GeomEnums.UH_static)
        tris_prim.set_index_type(GeomEnums.NT_uint32)
        tris_array = tris_prim.modify_vertices()
        tris_array.unclean_set_num_rows(len(faces_array))
        tris_memview = memoryview(tris_array).cast('B').cast('I')
        tris_memview[:] = faces_array

        # construct geometry
        surface_geometry = Geom(vertex_data)
        surface_geometry.addPrimitive(tris_prim)

        # construct node to hold the geometry
        surface_node = GeomNode(node_name)
        surface_node.addGeom(surface_geometry, RenderState.makeEmpty())

        # return the created node
        return surface_node

    # Define a function to prepared a two-sided transparent node path
    def apply_transparency_to_node_path(self, node_path, transparent):
        if transparent:
            # Ref: https://docs.panda3d.org/1.11/cpp/reference/panda3d.core.TransparencyAttrib
            node_path.setTwoSided(True)
            node_path.setTransparency(TransparencyAttrib.MDual)
        else:
            node_path.setTransparency(False)

    # Define a function to prepared a two-sided transparent node path
    def prepare_node_path(self, node_path, transparent=True, visualize=True):
        # Add transparency to node path
        self.apply_transparency_to_node_path(node_path, transparent=transparent)

        # Add lighting
        # Default lighting can be used for simplicity

        # visualize if requested
        if visualize:
            self.visualize_node_path(node_path)

    # Define a function to create surfaces
    def create_surface_mesh(self, vertices, triangles, vertex_colors=None, node_name=None, direction='camera_view', visualize=True, transparent=True, reorder_faces=True):
        ################################################################################
        # create the surface mesh and return nodepath
        surface_id = f"{utils.generate_unique_id()}"
        if node_name is None:
            node_name = f'surface#{surface_id}'
        else:
            node_name = f'{node_name}#{surface_id}'
        if direction == 'camera_view':
            direction = self.camera_direction

        # create the node
        surface_node = self.create_surface_mesh_node(
            vertices=vertices,
            triangles=triangles,
            node_name=node_name,
            vertex_colors=vertex_colors,
            direction=direction
        )

        # connect the node to a nodepath
        surface_node_path = NodePath(surface_node)

        # prepare nodepath
        self.prepare_node_path(surface_node_path, transparent=True, visualize=visualize)

        self.created_objects[node_name] = {
            'surface_id': surface_id,
            'node_name': node_name,
            'node_path': surface_node_path,
            'node_type': 'surface_mesh',
            'reorder_faces': reorder_faces,
            'vertices': vertices,
            'triangles': triangles,
            'vertex_colors': vertex_colors,
            'visualize': visualize,
            'transparent': transparent,
        }

        return self.created_objects[node_name]

    # Define a function to reorder the faces of a single object
    def reorder_faces_of_object(self, node_name):
        node_type = self.created_objects[node_name]['node_type']
        if node_type == 'surface_mesh':
            # first remove the rendered node path
            self.created_objects[node_name]['node_path'].removeNode()

            # create a new node and replace the old node path
            surface_node = self.create_surface_mesh_node(
                self.created_objects[node_name]['vertices'],
                self.created_objects[node_name]['triangles'],
                self.created_objects[node_name]['node_name'],
                vertex_colors=self.created_objects[node_name]['vertex_colors'],
                direction=self.camera_direction
            )
            surface_node_path = NodePath(surface_node)
            self.prepare_node_path(
                surface_node_path,
                transparent=self.created_objects[node_name]['transparent'],
                visualize=self.created_objects[node_name]['visualize'],
            )
            self.created_objects[node_name]['node_path'] = surface_node_path
        else:
            raise Exception(f'Reordering is not implemented for "{node_type}" nodes.')

    # Define a function to reorder the faces for appropriate rendering of transparency
    def reorder_faces_of_all_objects(self):
        for node_name in self.created_objects:
            if self.created_objects[node_name]['visualize']:
                if self.created_objects[node_name]['reorder_faces']:
                    self.reorder_faces_of_object(node_name)
        utils.garbage_collect()
        self.face_reordering_required = False

    # Define a function to check for camera direction updates for possible reordering
    def direction_update(self):
        # get the direction that the camera is aimed towards
        # direction = np.array(self.camera.get_quat())[1:]
        direction = np.array(self.camera.get_pos(self.render))

        if (direction != self.camera_direction).any():
            self.camera_direction = direction
            self.face_reordering_required = True

        elif self.face_reordering_required:
            # redraw surface nodes
            self.reorder_faces_of_all_objects()
            # Note: this function needs to be revisited and updated once appropriate OIT is implemented in panda3d

    # Define a procedure to have repeated checks at intervals (not every frame).
    def repeated_checks_task(self, task):
        self.direction_update()
        return Task.again

    # Define a procedure to render a created object
    def visualize_node_path(self, node_path, attach_to=None):
        if attach_to is None:
            attach_to = self.render

        node_path.detach_node()
        node_path.reparent_to(attach_to)

    # Create template objects for instancing
    def create_object_templates(self):
        self.template_sphere_object = self.create_sphere_mesh(node_name='template_sphere', visualize=False, transparent=False, reorder_faces=False)
        self.template_cylinder_object = self.create_cylinder_mesh(node_name='template_sphere', visualize=False, transparent=False, reorder_faces=False)

    # Define a function to create a sphere
    def create_sphere_mesh(self, node_name='sphere', **kwargs):
        sphere = trimesh.creation.icosphere(subdivisions=3)
        return self.create_surface_mesh(
            vertices=np.array(sphere.vertices),
            triangles=np.array(sphere.faces),
            node_name=node_name,
            **kwargs
        )

    # Define a function to create multiple sphere instances
    def create_multiple_sphere_instances(self, coordinates, radii, colors):
        surface_id = f'{utils.generate_unique_id()}'
        node_name = f'sphere_instance#{surface_id}'
        sphere_placehoders = {}
        for i in range(coordinates.shape[0]):
            sphere_placehoders[i] = self.render.attachNewNode(f'sphere-placeholder-{i}#{surface_id}')
            sphere_placehoders[i].setPos(*list(coordinates[i]))
            sphere_placehoders[i].setScale(radii[i], radii[i], radii[i])
            sphere_placehoders[i].setColor(*list(colors[i]))
            sphere_placehoders[i].setTransparency(TransparencyAttrib.MDual)
            self.template_sphere_object['node_path'].instanceTo(sphere_placehoders[i])

        self.created_objects[node_name] = {
            'surface_id': surface_id,
            'node_name': node_name,
            'sphere_placehoders': sphere_placehoders,
            'node_type': 'sphere_instance',
            'reorder_faces': False,
            'coordinates': coordinates,
            'radii': radii,
            'colors': colors,
            'visualize': True,
            'transparent': True,
        }

        return self.created_objects[node_name]

    # Define a function to create a cylinder
    def create_cylinder_mesh(self, node_name='cylinder', **kwargs):
        cylinder = trimesh.creation.cylinder(height=1., radius=1, sections=16)
        return self.create_surface_mesh(
            vertices=np.array(cylinder.vertices),
            triangles=np.array(cylinder.faces),
            node_name=node_name,
            **kwargs
        )

    # Define a function to create multiple cylinder instances to act as lines
    def create_multiple_cylinder_instances(self, coordinates, radii, colors):
        surface_id = f'{utils.generate_unique_id()}'
        node_name = f'cylinder_instance#{surface_id}'
        cylinder_placehoders = {}
        cylinder_placehoder_pivots = {}
        for i in range(coordinates.shape[0]):
            # location information
            start = coordinates[i][0]
            end = coordinates[i][1]
            length = np.linalg.norm(start - end)
            radius = radii[i]

            # resize the placeholder
            cylinder_placehoders[i] = NodePath(f'cylinder-placeholder-{i}#{surface_id}')
            cylinder_placehoders[i].setPos(*list(start + [0, 0, length / 2]))
            cylinder_placehoders[i].setScale(radius, radius, length)
            cylinder_placehoders[i].setColor(*list(colors[i]))
            cylinder_placehoders[i].setTransparency(TransparencyAttrib.MDual)
            self.template_cylinder_object['node_path'].instanceTo(cylinder_placehoders[i])

            cylinder_placehoder_pivots[i] = self.render.attachNewNode(f'cylinder-placeholder-pivot-{i}#{surface_id}')
            cylinder_placehoder_pivots[i].setPos(*list(start))
            cylinder_placehoder_pivots[i].setHpr(0, 90, 0)
            cylinder_placehoders[i].wrtReparentTo(cylinder_placehoder_pivots[i])
            cylinder_placehoder_pivots[i].lookAt(*list(end))

        self.created_objects[node_name] = {
            'surface_id': surface_id,
            'node_name': node_name,
            'cylinder_placehoders': cylinder_placehoders,
            'cylinder_placehoder_pivots': cylinder_placehoder_pivots,
            'node_type': 'cylinder_instance',
            'reorder_faces': False,
            'coordinates': coordinates,
            'radii': radii,
            'colors': colors,
            'visualize': True,
            'transparent': True,
        }

        return self.created_objects[node_name]

    # Define a function to clear created objects (not templates)
    def clear_created_object(self, node_name):
        node_type = self.created_objects[node_name]['node_type']
        if node_type == 'surface_mesh':
            self.created_objects[node_name]['node_path'].removeNode()
        elif node_type == 'sphere_instance':
            for sphere_placehoder in self.created_objects[node_name]['sphere_placehoders']:
                self.created_objects[node_name]['sphere_placehoders'][sphere_placehoder].removeNode()
        elif node_type == 'cylinder_instance':
            for cylinder_placehoder in self.created_objects[node_name]['cylinder_placehoders']:
                self.created_objects[node_name]['cylinder_placehoders'][cylinder_placehoder].removeNode()
            for cylinder_placehoder_pivot in self.created_objects[node_name]['cylinder_placehoder_pivots']:
                self.created_objects[node_name]['cylinder_placehoder_pivots'][cylinder_placehoder_pivot].removeNode()
        else:
            raise Exception(f'Clearing is not implemented for "{node_type}" nodes.')

    # Define a function to clear created objects (not templates)
    def clear_all_created_objects(self):
        for node_name in list(self.created_objects):
            if 'template' not in node_name:
                self.clear_created_object(node_name)
                self.created_objects.pop(node_name)
        utils.garbage_collect()
