import numpy as np
import genesis as gs
from utils.env_for_render import RenderEnv, euler_to_quat


class PiggyREnv(RenderEnv):
    def __init__(self, task="task06_piggy"):
        super().__init__(task)
        self.dest_pos = np.array([0.2, 0.2, 0.05])

    def add_entities(self):
        mat_rigid = gs.materials.Rigid(
            coup_friction=0.1,
            coup_softness=0.0001,
            coup_restitution=0.0001,
            sdf_cell_size=0.0001,
            sdf_min_res=64,
            sdf_max_res=64,
        )

        self.piggy = self.scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                file=self.task_asset("piggy_fab_simf_b.obj"),
                scale=0.057,
                pos=(0.2, 0.2, 0.01 + self.desk_height),
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
        )

        self.tool = self.scene.add_entity(
            material=mat_rigid,
            morph=gs.morphs.Mesh(
                file=self.task_asset("tool.obj"),
                scale=0.03,
                pos=(0.4, 0.2, 0.03 + self.desk_height),
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
        )

        for link in self.piggy.links:
            link._inertial_mass = 0.01
        for link in self.tool.links:
            link._inertial_mass = 0.01

    def add_camera_for_gallery(self):
        self.cam_gallery = self.scene.add_camera(
            pos=(1.5, 0.4, 1.5),
            lookat=(-0.1, 0.3, 1.0),
            fov=30,
            res=(1440, 1440),
            GUI=False,
        )

    def add_camera_for_trajectory(self):
        self.cam_trajectory = self.scene.add_camera(
            pos=(2.1, 0.55, 2.5),
            lookat=(-0.3, 0.55, 0.8),
            fov=30,
            res=(1440, 1440),
            GUI=False,
        )


env = PiggyREnv()
env.add_camera_for_gallery()
env.add_camera_for_trajectory()
env.scene.build()

q_hand = np.array([0.4, 0.2, 1.03, 180, 0, 90])
q_hand[3:] = np.deg2rad(q_hand[3:])
q_xarm, err = env.xarm.inverse_kinematics(
    link=env.xarm.get_link("link_tcp"),
    pos=q_hand[:3],
    quat=euler_to_quat(q_hand[3:]),
    return_error=True,
    respect_joint_limit=False,
)
q_xarm[-2:] = 0.0
env.xarm.set_dofs_position(q_xarm)
env.xarm.control_dofs_position(q_xarm)

q_tool = np.array([0.4, 0.2, 1.03, 90, 0, 90])
q_tool[3:] = np.deg2rad(q_tool[3:])
env.tool.set_dofs_position(q_tool)

q_piggy = np.array([0.2, 0.2, 1.01, 90, 0, 90])
q_piggy[3:] = np.deg2rad(q_piggy[3:])
env.piggy.set_dofs_position(q_piggy)
env.piggy.control_dofs_position(q_piggy)

env.scene.visualizer.update()
env.save_gallery_img()

for i in range(80):
    env.scene.step()
    if i % 10 == 0:
        env.save_trajectory_img()

