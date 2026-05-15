from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np


class bread_skillet_plate_aba(Base_Task):

    def _to_list(self, value):
        if value is None:
            return None
        if hasattr(value, "p") and hasattr(value, "q"):
            return self._to_list(value.p) + self._to_list(value.q)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list):
            return [float(x) if np.isscalar(x) else x for x in value]
        if np.isscalar(value):
            return float(value)
        return value

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1

        self.plate_center = np.array([x_sign * 0.08, -0.14, 0.742])
        self.skillet_center = np.array([x_sign * 0.22, -0.02, 0.742])
        self.grid_a_center = self.plate_center
        self.grid_b_center = self.skillet_center

        self.plate_id = 0
        self.plate = create_actor(
            self,
            pose=sapien.Pose(self.plate_center.tolist(), [0.5, 0.5, 0.5, 0.5]),
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True,
        )
        self.skillet_id = np.random.choice([0, 1, 2, 3])
        self.skillet = create_actor(
            self,
            pose=sapien.Pose(self.skillet_center.tolist(), [0, 0, 0.707, 0.707]),
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
            is_static=True,
        )
        self.bread_id = 6
        bread_center = self.plate.get_functional_point(0, "pose").p.copy()
        bread_center[:2] = self.plate_center[:2]
        bread_center[2] = 0.750
        self.object = create_actor(
            self,
            pose=sapien.Pose(bread_center.tolist(), [0.707, 0.707, 0.0, 0.0]),
            modelname="075_bread",
            model_id=self.bread_id,
            convex=True,
        )
        self.object.set_mass(0.001)
        self.eval_visited_b = False
        self.eval_returned_a = False

        self.add_prohibit_area(self.plate, padding=0.09)
        self.add_prohibit_area(self.skillet, padding=0.09)
        self.add_prohibit_area(self.object, padding=0.03)

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent="go_to_skillet", destination="skillet", ambiguous=False, control=True)

        self._pick_object(arm_tag, stage_prefix="pick_1")
        self._set_aliasbench_trace(intent="go_to_skillet", destination="skillet", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the bread from the plate.")
            return self.info

        self._place_object_at_target(arm_tag, "skillet", target_pose=self._skillet_target_pose(), pre_dis=0.12, dis=0.03)

        if not self.plan_success:
            print("Failed to place the bread on the skillet.")
            return self.info
        self.eval_visited_b = True
        
        obj_pos = self.object.get_pose().p
        self._set_aliasbench_trace(intent="go_to_plate", destination="plate", ambiguous=False, control=True)
        self._pick_object(arm_tag, stage_prefix="pick_2")
        self._set_aliasbench_trace(intent="go_to_plate", destination="plate", ambiguous=True, control=False)

        if not self.plan_success:
            print("Failed to pick the bread from the skillet.")
            return self.info

        self._move_object_over_target(arm_tag, "plate", target_pose=self._plate_target_pose(), pre_dis=0.14)
        if not self.plan_success:
            print("Failed to move the bread over the plate.")
            return self.info
        self._release_object_at_target(arm_tag, target_pose=self._plate_target_pose(), z_offset=0.005)
        if not self.plan_success:
            print("Failed to release the bread on the plate.")
            return self.info
        self.eval_returned_a = True
        
        obj_pos = self.object.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="world"))

        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination="plate", ambiguous=False, control=False)

        self.info["info"] = {"{A}": f"075_bread/base{self.bread_id}", "{B}": f"106_skillet/base{self.skillet_id}", "{C}": f"003_plate/base{self.plate_id}"}
        return self.info

    def _pick_object(self, arm_tag, stage_prefix="pick"):
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.07, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="world"))
    def _skillet_target_pose(self):
        return self.skillet.get_functional_point(0)

    def _plate_target_pose(self):
        return self.plate.get_functional_point(0)

    def _move_object_over_target(self, arm_tag, target_name, target_pose, pre_dis=0.12):

        ee_pose = np.array(self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(), dtype=np.float64)
        object_pos = np.array(self.object.get_pose().p, dtype=np.float64)
        target_pose_arr = np.array(target_pose, dtype=np.float64)
        ee_object_offset = ee_pose[:3] - object_pos
        target_object_pos = target_pose_arr[:3].copy()
        target_object_pos[2] += pre_dis
        target_ee_pose = ee_pose.copy()
        target_ee_pose[:3] = target_object_pos + ee_object_offset
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_ee_pose.tolist()))


    def _release_object_at_target(self, arm_tag, target_pose, z_offset=0.0):
        ee_pose = np.array(self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(), dtype=np.float64)
        object_pos = np.array(self.object.get_pose().p, dtype=np.float64)
        target_pose_arr = np.array(target_pose, dtype=np.float64)
        ee_object_offset = ee_pose[:3] - object_pos
        target_object_pos = target_pose_arr[:3].copy()
        target_object_pos[2] += z_offset
        target_ee_pose = ee_pose.copy()
        target_ee_pose[:3] = target_object_pos + ee_object_offset
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target_ee_pose.tolist()))
        if not self.plan_success:
            return
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))

    def _place_object_at_target(self, arm_tag, target_name, target_pose, pre_dis, dis):

        self.move(
            self.place_actor(
                self.object,
                arm_tag=arm_tag,
                target_pose=target_pose,
                pre_dis=pre_dis,
                dis=dis,
                constrain="free",
            ))


    def check_success(self):
        pos = self.object.get_pose().p
        at_b = np.all(np.abs(pos[:2] - self.grid_b_center[:2]) < np.array([0.10, 0.10]))
        at_a = np.all(np.abs(pos[:2] - self.grid_a_center[:2]) < np.array([0.05, 0.05]))
        if (not self.eval_visited_b) and at_b:
            self.eval_visited_b = True
        if self.eval_visited_b and at_a:
            self.eval_returned_a = True
        final_ok = (
            self.eval_visited_b
            and self.eval_returned_a
            and at_a
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        return final_ok
