"""
This module contains the code to create a viewer window in panda3d.

The viewer window will contain the rendered scene, it may potentially contain
multiple viewpoints/sub-windows.

Notes
-----
Author: Sina Mansour L.
"""

import os
import numpy as np
import array
import trimesh

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d import core

from . import cerebro_utils as utils


# configurations
core.load_prc_file_data('', 'aux-display pandadx9')
core.load_prc_file_data('', 'aux-display pandadx8')
core.load_prc_file_data('', 'aux-display p3tinydisplay')
# suppressing logs, see https://discourse.panda3d.org/t/how-to-hide-all-console-output/13664/2
core.load_prc_file_data('', f'notify-output {os.path.dirname(__file__)}/cerebro_window_logs.log')
core.load_prc_file_data('', 'win-size 1280 720')
# load_prc_file_data('', 'window-title Cerebro Viewer')
# load_prc_file_data('', 'icon-filename Cerebro_Viewer.ico')
# load_prc_file_data("","framebuffer-multisample True")
# load_prc_file_data("","multisamples 16")
# load_prc_file_data('', 'background-color 0.1 0.1 0.1 0.0')
# load_prc_file_data('', 'show-scene-graph-analyzer True')


class Cerebro_window(ShowBase):

    def __init__(self, background_color=(0.1, 0.1, 0.1, 0.0), camera_pos=(400, 0, 0), camera_target=(0, 0, 0), camera_fov=35, camera_rotation=0, rotation_speed=100, movement_speed=100, zoom_speed=0.2, window_size=(1280, 720), offscreen=False):
        # Handle offscreen rendering
        windowType = 'offscreen' if offscreen else None
        super().__init__(self, windowType=windowType)

        # Initial configurations

        # Window information
        self.background_color = list(background_color)
        self.window_size = window_size
        self.offscreen = offscreen

        # Camera positioning
        # position is relative to target
        self.cam_init_x, self.cam_init_y, self.cam_init_z = list(camera_pos)
        self.cam_target_x, self.cam_target_y, self.cam_target_z = list(camera_target)
        self.camera_fov = camera_fov
        self.camera_rotation = camera_rotation

        # Update speed
        self.rotation_speed = rotation_speed
        self.movement_speed = movement_speed
        self.zoom_speed = zoom_speed

        # Setup procedures

        # Material setup
        self.setup_materials()

        # Camera setup
        self.setup_camera()

        # Window setup
        self.setup_window()

        if not offscreen:
            # Keyboard and mouse setup
            self.setup_keyboard_and_mouse()

            # Add the update task to the task manager.
            self.taskMgr.add(self.update_task, "update_task")

            # Add the repeated checking task to the task manager
            self.taskMgr.doMethodLater(0.05, self.repeated_checks_task, "repeated_checks_task")

        # Create a dictionary for rendered objects
        self.created_objects = {}

        # A flag to check if re-rendering by reordering of faces is required
        self.face_reordering_required = False

        self.create_object_templates()

    # Draw without running
    def draw(self):
        self.taskMgr.step()
        self.taskMgr.step()

    # Window setup procedure
    def setup_window(self):
        # Window properties
        # Ref: https://docs.panda3d.org/1.11/cpp/reference/panda3d.core.WindowProperties
        self.window_properties = core.WindowProperties()
        self.window_properties.set_size(self.window_size[0], self.window_size[1])
        self.window_properties.set_title('Cerebro Viewer')
        self.window_properties.set_icon_filename('Cerebro_Viewer.ico')

        if self.offscreen:
            # Configure a window buffer and display region for offscreen rendering.

            # Frame buffer properties
            self.framebuffer_properties = core.FrameBufferProperties()
            self.framebuffer_properties.setRgbColor(True)
            # Only render RGB with 8 bit for each channel, no alpha channel
            self.framebuffer_properties.setRgbaBits(8, 8, 8, 0)
            self.framebuffer_properties.setDepthBits(24)

            # Window buffer
            self.make_offscreen_buffer(name="cameraview")

            # Create display region
            self.display_region = self.window_buffer.makeDisplayRegion()
            self.display_region.setCamera(self.cam)

        else:
            # Configure the default window

            # Background color
            self.set_background_color(*self.background_color)

            # self.set_scene_graph_analyzer_meter(True)
            # self.set_frame_rate_meter(True)

            # Window properties
            self.win.requestProperties(self.window_properties)

    # Change window size of offscreen renderer
    def make_offscreen_buffer(self, name, pipe=None, sort=0, fb_prop=None, win_prop=None, flags=core.GraphicsPipe.BFRefuseWindow):
        if pipe is None:
            pipe = self.pipe
        if fb_prop is None:
            fb_prop = self.framebuffer_properties
        if win_prop is None:
            win_prop = self.window_properties
        # Window buffer
        self.window_buffer = self.graphicsEngine.makeOutput(
            pipe=pipe, name=name, sort=0, fb_prop=fb_prop, win_prop=win_prop,
            flags=flags, gsg=self.win.getGsg(), host=self.win,
        )

        # set the background color for the offscreen buffer
        self.window_buffer.set_clear_color_active(True)
        self.window_buffer.set_clear_color(core.LVecBase4f(*self.background_color))

    # Change window size of offscreen renderer
    def reset_offscreen_size(self, x, y):
        # self.win.removeDisplayRegion(self.display_region)
        self.window_buffer.remove_all_display_regions()
        self.window_properties.set_size(x, y)

        # Window buffer
        self.make_offscreen_buffer(name="cameraview")

        # Create display region
        self.display_region = self.window_buffer.makeDisplayRegion()
        self.camLens.setAspectRatio(float(x) / y)
        self.display_region.setCamera(self.cam)

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
        self.mouseWatcherNode.set_modifier_buttons(core.ModifierButtons())
        self.buttonThrowers[0].node().set_modifier_buttons(core.ModifierButtons())

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

    # Material setup procedure
    def setup_materials(self):
        self.materials = {}
        self.materials[1] = core.Material("StandardMaterial")
        # self.materials[1].setDiffuse((1., 1., 1., 1.)) # Replaces actual color
        # self.materials[1].setAmbient((1., 1., 1., 1.)) # Creates a paler material
        self.materials[1].setEmission((0.1, 0.1, 0.1, 0.1)) # Higher values are brighter
        self.materials[1].setSpecular((0.2, 0.2, 0.2, 0.2)) # Higher values are more glossy
        self.materials[1].setShininess(10)

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

        # Add lighting at the position of the camera
        # Fixed background ambient lighting
        self.alight = core.AmbientLight('alight')
        self.alight.setColor((0.3, 0.3, 0.3, 1))
        self.alnp = self.render.attachNewNode(self.alight)
        self.render.setLight(self.alnp)
        
        # # A dummy sphere for camera and light position
        # self.slight = self.loader.loadModel('models/misc/sphere')
        # self.slight.setMaterial(self.materials[1])
        # self.slight.setPos(0, 200, 0)
        # self.slight.setScale(1, 1, 1)
        # self.slight.reparentTo(self.cam)
        
        # Directional light according to camera
        self.dlight = core.DirectionalLight('dlight')
        self.dlight.setColor((0.7, 0.7, 0.7, 1))
        self.dlnp = core.NodePath(self.dlight)
        self.render.setLight(self.dlnp)
        self.dlnp.reparentTo(self.cam)
        self.dlnp.lookAt(self.cam_target_x, self.cam_target_y, self.cam_target_z)
        
        # # Point light from camera
        # self.plight = core.PointLight('plight')
        # self.plight.setColor((0.9, 0.9, 0.9, 1))
        # self.plight.setAttenuation((1, 0, 1))
        # self.plnp = core.NodePath(self.plight)
        # self.plnp.setPos(0, 100, 0)
        # self.render.setLight(self.plnp)
        # self.plnp.reparentTo(self.cam)

    # Update camera direction
    def update_camera(self, camera_target=None, camera_pos=None, camera_rotation=None, camera_fov=None):
        if camera_pos is not None:
            (self.cam_init_x, self.cam_init_y, self.cam_init_z) = camera_pos
        self.camera_pivot_node.setPos(0, 0, 0)
        self.camera_pivot_node.lookAt(self.cam_init_x, self.cam_init_y, self.cam_init_z)
        dist = np.sqrt((self.cam_init_x ** 2) + (self.cam_init_y ** 2) + (self.cam_init_z ** 2))
        self.camera.setPos(0, dist, 0)
        if camera_rotation is not None:
            self.camera_rotation = camera_rotation
        self.camera_pivot_node.setR(self.camera_rotation)
        if camera_target is not None:
            (self.cam_target_x, self.cam_target_y, self.cam_target_z) = camera_target
        self.camera_pivot_node.setPos(
            self.cam_target_x,
            self.cam_target_y,
            self.cam_target_z
        )
        if camera_fov is not None:
            self.camera_fov = camera_fov
        self.cam.node().getLens().setFov(self.camera_fov)

        # reorder faces
        self.camera_direction = np.array(self.camera.get_pos(self.render))
        self.reorder_faces_of_all_objects()

    # Get camera direction information
    def get_camera_view(self):
        return {
            'camera_target': (self.cam_target_x, self.cam_target_y, self.cam_target_z),
            'camera_pos': (self.cam_init_x, self.cam_init_y, self.cam_init_z),
            'camera_rotation': self.camera_rotation,
            'camera_fov': self.camera_fov,
        }

    # Get camera position
    def get_camera_target_position(self):
        return np.array(self.camera_pivot_node.getPos())

    # Define a procedure to update the camera direction every frame.
    def update_task(self, task):
        # Check that mouse is in window
        if self.mouseWatcherNode.hasMouse():
            # get mouse postion
            self.mouse_x = self.mouseWatcherNode.getMouseX()
            self.mouse_y = self.mouseWatcherNode.getMouseY()

        # A frame-based time multiplier
        dt = core.ClockObject.getGlobalClock().getDt()

        ud_rotation = 0
        rl_rotation = 0
        ud_position = 0
        rl_position = 0
        zoom_factor = 0
        camera_rotation = 0

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
            if self.key_map['right']:
                camera_rotation += dt * self.rotation_speed
            if self.key_map['left']:
                camera_rotation -= dt * self.rotation_speed
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

        # Perform camera rotation
        if (camera_rotation != 0):
            current_R = self.camera.getR()
            self.camera.setR(current_R + camera_rotation)

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

        # 
        # if (ud_rotation != 0) or (rl_position != 0) or (rl_rotation != 0) or (ud_position != 0) or (camera_rotation != 0) or (zoom_factor != 0):
        #     print("-"*80)
        #     print("Camera pos", self.cam.getPos(self.render))
        #     print("Camera dir", self.cam.getHpr(self.render))
        #     print("Light pos", self.dlnp.getPos(self.render))
        #     print("Light dir", self.dlnp.getHpr(self.render))

        # Continue
        return Task.cont

    # Move camera to a particular distance from center
    def move_camera_to_distance(self, distance):
        current_pos = self.camera.getPos()
        current_distance = np.linalg.norm(current_pos)
        zoom_factor = distance / current_distance
        self.camera.setPos(
            current_pos[0] * np.exp(zoom_factor),
            current_pos[1] * np.exp(zoom_factor),
            current_pos[2] * np.exp(zoom_factor),
        )

    # Define a procedure to get a screenshot of the view.
    def get_screenshot(self, output_file):
        self.win.save_screenshot(output_file)

    # Define a function to depth sort the faces
    def sort_faces(self, direction, triangles, vertices):
        # reorder triangles according to a direction vector
        return triangles[np.argsort(vertices[triangles].mean(1).dot(direction))]

    # Define a function to compute surface normals of vertices
    def compute_vertex_normals(self, vertices, triangles):
        # Compute surface normals for each triangle
        tri_normals = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                               vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
        tri_normals = tri_normals / np.linalg.norm(tri_normals, axis=1)[:, None]

        # Initialize vertex normals to zero
        vertex_normals = np.zeros(vertices.shape, dtype=vertices.dtype)

        # Assign triangle normals to their vertices
        vertex_normals[triangles[:, 0]] += tri_normals
        vertex_normals[triangles[:, 1]] += tri_normals
        vertex_normals[triangles[:, 2]] += tri_normals

        # Normalize vertex normals
        vertex_normals = vertex_normals / np.maximum(np.linalg.norm(vertex_normals, axis=1)[:, None], 1e-6)

        return vertex_normals

    # Define a function to create a node object for a surface mesh
    def create_surface_mesh_node(self, vertices, triangles, node_name, vertex_colors=None, direction=None):
        # convert information into array format
        # vertex coordinates
        coords = array.array("f", vertices.ravel())
        # vertex normals
        vertex_normals = array.array("f", self.compute_vertex_normals(vertices, triangles).ravel())
        # vertex colors
        if vertex_colors is not None:
            colors = array.array("B", (255 * vertex_colors.ravel()).astype(np.uint8))
        # faces sorted by direction
        if direction is None:
            faces_array = array.array("I", triangles.ravel())
        else:
            faces_array = array.array("I", self.sort_faces(direction, triangles, vertices).ravel())

        # specify a generic vertex format
        vertex_format = core.GeomVertexFormat()
        # add 3d coordinates to the format
        vertex_format.add_array(core.GeomVertexFormat.get_v3().arrays[0])
        # add surface normal information
        vertex_format.add_array(core.GeomVertexArrayFormat("normal", 3, core.Geom.NT_float32, core.Geom.C_normal))
        # add color information to the format if requested
        if vertex_colors is not None:
            vertex_format.add_array(core.GeomVertexArrayFormat('color', 4, core.Geom.NT_uint8, core.Geom.C_color))
        # register format
        vertex_format = core.GeomVertexFormat.register_format(vertex_format)

        # create data using this format
        vertex_data = core.GeomVertexData("vertex_data", vertex_format, core.Geom.UH_static)
        # set the number of vertices according to the surface mesh
        vertex_data.unclean_set_num_rows(vertices.shape[0])

        # get memoryview handles to data and fill by the available information
        vertex_array = vertex_data.modify_array(0)
        vertex_memview = memoryview(vertex_array).cast("B").cast("f")
        vertex_memview[:] = coords
        normal_array = vertex_data.modify_array(1)
        normal_memview = memoryview(normal_array).cast("B").cast("f")
        normal_memview[:] = vertex_normals
        if vertex_colors is not None:
            color_array = vertex_data.modify_array(2)
            color_memview = memoryview(color_array).cast("B")
            color_memview[:] = colors

        # store face indices
        tris_prim = core.GeomTriangles(core.GeomEnums.UH_static)
        tris_prim.set_index_type(core.GeomEnums.NT_uint32)
        tris_array = tris_prim.modify_vertices()
        tris_array.unclean_set_num_rows(len(faces_array))
        tris_memview = memoryview(tris_array).cast('B').cast('I')
        tris_memview[:] = faces_array

        # construct geometry
        surface_geometry = core.Geom(vertex_data)
        surface_geometry.addPrimitive(tris_prim)

        # construct node to hold the geometry
        surface_node = core.GeomNode(node_name)
        surface_node.addGeom(surface_geometry, core.RenderState.makeEmpty())

        # return the created node
        return surface_node

    # Define a function to prepared a two-sided transparent node path
    def apply_transparency_to_node_path(self, node_path, transparent):
        node_path.setTwoSided(True)
        if transparent:
            # Ref: https://docs.panda3d.org/1.11/cpp/reference/panda3d.core.TransparencyAttrib
            node_path.setTransparency(core.TransparencyAttrib.MDual)
        else:
            node_path.setTransparency(False)

    # Define a function to prepare a two-sided transparent node path
    def prepare_node_path(self, node_path, transparent=True, visualize=True):
        # Add transparency to node path
        self.apply_transparency_to_node_path(node_path, transparent=transparent)

        # Assign a material to the node_path
        node_path.setMaterial(self.materials[1])

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
        surface_node_path = core.NodePath(surface_node)

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
            surface_node_path = core.NodePath(surface_node)
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
        self.template_cylinder_object = self.create_cylinder_mesh(node_name='template_cylinder', visualize=False, transparent=False, reorder_faces=False)

    # Define a function to create a sphere
    def create_sphere_mesh(self, node_name='sphere', **kwargs):
        sphere = trimesh.creation.icosphere(subdivisions=2)
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
            sphere_placehoders[i].setScale(radii[i][0], radii[i][1], radii[i][2])
            sphere_placehoders[i].setColor(*list(colors[i]))
            sphere_placehoders[i].setTransparency(core.TransparencyAttrib.MDual)
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
            cylinder_placehoders[i] = core.NodePath(f'cylinder-placeholder-{i}#{surface_id}')
            cylinder_placehoders[i].setPos(*list(start + [0, 0, length / 2]))
            cylinder_placehoders[i].setScale(radius, radius, length)
            cylinder_placehoders[i].setColor(*list(colors[i]))
            cylinder_placehoders[i].setTransparency(core.TransparencyAttrib.MDual)
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

        self.created_objects.pop(node_name)

    # Define a function to clear created objects (not templates)
    def clear_all_created_objects(self):
        for node_name in list(self.created_objects):
            if 'template' not in node_name:
                self.clear_created_object(node_name)
        utils.garbage_collect()

    # Draw an offscreen-rendered view to a matplotlib axes
    def offscreen_draw_to_matplotlib_axes(self, ax):
        """Draw an offscreen-rendered view to a matplotlib axes.

        Note: this functionality is experimental and might not fully work depending on
        viewer configuration.

        Args:
            ax (matplotlib.Axes): the axes into which the view will be drawn.
        """
        # First, update the aspect ration from axis information
        x, y = self.window_size
        ax_aspect = ax.get_window_extent().width / ax.get_window_extent().height
        if (ax_aspect * y) > x:
            x = int(ax_aspect * y)
        else:
            y = int(x / ax_aspect)
        self.reset_offscreen_size(x, y)

        # Create the texture that contain the image buffer
        bgra_tex = core.Texture()
        self.window_buffer.addRenderTexture(bgra_tex, core.GraphicsOutput.RTMCopyRam, core.GraphicsOutput.RTPColor)

        # Now we can render the frame manually
        self.graphicsEngine.renderFrame()

        # Get the frame data as numpy array
        bgra_img = np.frombuffer(bgra_tex.getRamImage(), dtype=np.uint8)
        bgra_img.shape = (bgra_tex.getYSize(), bgra_tex.getXSize(), bgra_tex.getNumComponents())

        # invert the channels from bgr to rgb
        rgb_img = np.flip(bgra_img[:,:,:3], axis=2)

        # plot the image in a matplotlib axes
        ax.imshow(rgb_img, origin='lower')
