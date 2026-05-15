from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import transforms3d as t3d
from ._ais_benchmark_common import use_fixed_aliasbench_colors


class place_phone_other_stand(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag('left' if np.random.randint(0, 2) == 0 else 'right')
        self.start_on_left = np.random.randint(0, 2) == 0

        left_stand_center = np.array([
            np.random.uniform(-0.24, -0.04),
            np.random.uniform(-0.04, 0.08),
            0.741 + self.table_z_bias,
        ])
        right_stand_center = np.array([
            np.random.uniform(0.04, 0.24),
            np.random.uniform(-0.04, 0.08),
            0.741 + self.table_z_bias,
        ])

        start_stand_center = left_stand_center if self.start_on_left else right_stand_center
        target_stand_center = right_stand_center if self.start_on_left else left_stand_center

        stand_quat = [0.707, 0.707, 0.0, 0.0]
        sampled_left_stand_id = np.random.choice([1, 2], 1)[0]
        sampled_right_stand_id = np.random.choice([1, 2], 1)[0]
        self.left_stand_id = sampled_left_stand_id
        self.right_stand_id = sampled_right_stand_id
        self.left_stand = create_actor(
            scene=self,
            pose=sapien.Pose(left_stand_center.tolist(), stand_quat),
            modelname='078_phonestand',
            convex=True,
            model_id=self.left_stand_id,
            is_static=True,
        )
        self.right_stand = create_actor(
            scene=self,
            pose=sapien.Pose(right_stand_center.tolist(), stand_quat),
            modelname='078_phonestand',
            convex=True,
            model_id=self.right_stand_id,
            is_static=True,
        )

        self.start_stand = self.left_stand if self.start_on_left else self.right_stand
        self.target_stand = self.right_stand if self.start_on_left else self.left_stand
        self.start_stand_center = start_stand_center
        self.target_stand_center = target_stand_center

        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        tmp_phone = create_actor(
            scene=self,
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            modelname='077_phone',
            convex=True,
            model_id=self.phone_id,
        )
        phone_local_fp = np.array(tmp_phone.get_functional_point(0, 'matrix'), dtype=np.float64)
        target_fp = np.array(self.start_stand.get_functional_point(0, 'matrix'), dtype=np.float64)
        phone_world = target_fp @ np.linalg.inv(phone_local_fp)
        phone_pose = sapien.Pose(phone_world[:3, 3], t3d.quaternions.mat2quat(phone_world[:3, :3]))
        self.scene.remove_actor(tmp_phone.actor)

        self.phone = create_actor(
            scene=self,
            pose=phone_pose,
            modelname='077_phone',
            convex=True,
            model_id=self.phone_id,
        )
        self.phone.set_mass(0.01)

        self.add_prohibit_area(self.left_stand, padding=0.12)
        self.add_prohibit_area(self.right_stand, padding=0.12)
        self.add_prohibit_area(self.phone, padding=0.12)
        self.delay(2)

    def play_once(self):
        arm_tag = self.arm_tag
        target_name = "right_stand" if self.start_on_left else "left_stand"
        self._init_aliasbench_trace(intent=f"go_to_{target_name}", destination=target_name, ambiguous=False, control=True)
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag, pre_grasp_dis=0.08))
        if not self.plan_success:
            print('Failed to pick the phone from the start stand.')
            return self.info

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis='world'))
        self._set_aliasbench_trace(intent=f"go_to_{target_name}", destination=target_name, ambiguous=True, control=False)
        if not self.plan_success:
            print('Failed to lift the phone from the start stand.')
            return self.info

        self.move(self.place_actor(
            self.phone,
            arm_tag=arm_tag,
            target_pose=self.target_stand.get_functional_point(0),
            functional_point_id=0,
            dis=0,
            constrain='align',
        ))
        if not self.plan_success:
            print('Failed to place the phone on the other stand.')
            return self.info

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis='world'))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination=target_name, ambiguous=False, control=False)

        self.info['info'] = {
            '{A}': f'077_phone/base{self.phone_id}',
            '{B}': f'078_phonestand/base{self.left_stand_id}',
            '{C}': f'078_phonestand/base{self.right_stand_id}',
        }
        return self.info

    def check_success(self):
        phone_func_pose = np.array(self.phone.get_functional_point(0))
        target_func_pose = np.array(self.target_stand.get_functional_point(0))
        start_func_pose = np.array(self.start_stand.get_functional_point(0))
        at_target = np.all(np.abs(phone_func_pose[:3] - target_func_pose[:3]) < np.array([0.045, 0.04, 0.04]))
        away_from_start = np.linalg.norm(phone_func_pose[:2] - start_func_pose[:2]) > 0.10
        return (
            at_target
            and away_from_start
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
