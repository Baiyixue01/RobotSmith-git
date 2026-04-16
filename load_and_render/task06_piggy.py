import os

import genesis as gs
import imageio
import numpy as np
from genesis.constants import backend as gs_backend

project_path = os.path.dirname(os.path.join(os.path.abspath(__file__), ".."))

gs.init(seed=0, precision="32", logging_level="error", backend=gs_backend.gpu)
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
        gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_fov=30,
        res=(960, 640),
    ),
    vis_options=gs.options.VisOptions(
        env_separate_rigid=True,
    ),
    show_viewer=False,
)  # scene
plane = scene.add_entity(
    morph=gs.morphs.Plane(),
)  # plane
xarm = scene.add_entity(
    gs.morphs.URDF(
        file="{}/assets/xarm7_with_gripper_reduced_dof.urdf".format(project_path),
        fixed=True,
        collision=True,
        links_to_keep=["link_tcp"],
    ),
)  # xarm
mat_rigid = gs.materials.Rigid(
    coup_friction=0.1,
    coup_softness=0.0001,
    coup_restitution=0.0001,
    sdf_cell_size=0.0001,
    sdf_min_res=64,
    sdf_max_res=64,
)  # mat_rigid
piggy = scene.add_entity(
    material=mat_rigid,
    morph=gs.morphs.Mesh(
        file="{}/task06_piggy/piggy_fab_simf_b.obj".format(project_path),
        scale=0.057,
        pos=(0.2, 0.2, 0.01),
        euler=(90, 0, 90),
        fixed=False,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color=(0.0, 0.0, 0.8),
    ),
)  # piggy
tool = scene.add_entity(
    material=mat_rigid,
    morph=gs.morphs.Mesh(
        file="{}/task06_piggy/tool.obj".format(project_path),
        scale=0.03,
        pos=(0.4, 0.2, 0.03),
        euler=(90, 0, 90),
        fixed=False,
        collision=True,
        decompose_nonconvex=False,
        convexify=False,
        decimate=False,
    ),
    surface=gs.surfaces.Default(
        color=(1.0, 0.0, 0.0),
    ),
)  # tool
cam = scene.add_camera(
    pos=(0.4, -0.9, 0.9),
    lookat=(0.4, 0.0, 0.0),
    fov=50,
    res=(1440, 1440),
    GUI=False,
)  # cam

scene.build()

for link in piggy.links:
    link._inertial_mass = 0.01
for link in tool.links:
    link._inertial_mass = 0.01


def save_img(cam, nam):
    img = cam.render()[0][0]
    imageio.imwrite(nam, img)
    return img

