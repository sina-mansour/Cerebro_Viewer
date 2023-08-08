"""
================
Camera usage
================
This demo shows different ways to position the camera
"""
import matplotlib.pyplot as plt
from cerebro import cerebro_brain_utils as cbu
from cerebro import cerebro_brain_viewer as cbv

my_brain_viewer = cbv.Cerebro_brain_viewer(offscreen=True,null_color=(0.7,0.7,0.7,1))
surface_model = my_brain_viewer.load_template_GIFTI_cortical_surface_models('pial')
cifti_space = my_brain_viewer.visualize_cifti_space()

fig, axs = plt.subplots(1,3,figsize=(9,3))
plt.subplots_adjust(wspace=0, hspace=0)

### Method 1 (manual) Camera position and camera target are specified as (x,y,z) coordinates. camera_fov is angle in degrees.
ax = axs[0]
ax.axis('off')
camconf = {'camera_pos': (0, 300, 0), 'camera_target': (0, -15, 15), 'camera_fov': 35, 'camera_rotation': 0}
my_brain_viewer.viewer.change_view(**camconf)
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)

### Method 2. Specify camera position only with coordinates
ax = axs[1]
ax.axis('off')
view = ((-400, 0, 0), None, None, None) #last 3 values are camera_target. When set to None, defaults to centre of mass of all visible objects
camconf = my_brain_viewer._view_to_camera_config(view)
camconf = my_brain_viewer._zoom_camera_to_content(camconf) #Adjust FOV to barely fit all visible objects. To zoom in/out further, you can manually change camconf['camera_pos'].
my_brain_viewer.viewer.change_view(**camconf)
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)

### Method 3. Specify camera position as a direction. Same result as above.
ax = axs[2]
ax.axis('off')
camconf = my_brain_viewer._view_to_camera_config("L")
camconf = my_brain_viewer._zoom_camera_to_content(camconf)
my_brain_viewer.viewer.change_view(**camconf)
my_brain_viewer.offscreen_draw_to_matplotlib_axes(ax)

my_brain_viewer.viewer.window.destroy()
plt.show()