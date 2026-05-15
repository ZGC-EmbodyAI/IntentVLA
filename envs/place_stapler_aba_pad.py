from ._base_task import Base_Task
from .utils import *
from ._ais_benchmark_common import sample_distinct_pad_colors
import sapien
import numpy as np


class place_stapler_aba_pad(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.pad_half_size = [0.055, 0.03, 0.0005]
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1

        self.grid_a_center = np.array([x_sign * 0.03, -0.14, 0.742])
        self.grid_b_center = np.array([x_sign * 0.25, -0.02, 0.742])
        self.grid_colors = sample_distinct_pad_colors(2)

        self.grid_a = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]),
            half_size=self.pad_half_size,
            color=self.grid_colors[0],
            name="grid_A",
        )
        self.grid_b = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.grid_b_center.tolist(), [1, 0, 0, 0]),
            half_size=self.pad_half_size,
            color=self.grid_colors[1],
            name="grid_B",
        )

        obj_center = self.grid_a_center.copy()
        obj_center[2] = 0.741
        self.object_id = np.random.choice([0, 1, 2, 3, 4, 5, 6], 1)[0]
        self.object = create_actor(
            scene=self,
            pose=sapien.Pose(obj_center.tolist(), [0.5, 0.5, 0.5, 0.5]),
            modelname="048_stapler",
            convex=True,
            model_id=self.object_id,
        )

        self.eval_visited_b = False
        self.eval_returned_a = False

        self.add_prohibit_area(sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]), padding=0.09)
        self.add_prohibit_area(sapien.Pose(self.grid_b_center.tolist(), [1, 0, 0, 0]), padding=0.09)
        self.add_prohibit_area(self.object, padding=0.1)

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent="go_to_B", destination="B", ambiguous=False, control=True)
        self._pick_object(arm_tag)
        self._set_aliasbench_trace(intent="go_to_B", destination="B", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the stapler from the grid A.")
            return self.info
        self._place_object(arm_tag, self.grid_b_center)
        if not self.plan_success:
            print("Failed to place the stapler at the grid B.")
            return self.info
        self.eval_visited_b = True
        
        obj_pos = self.object.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self._set_aliasbench_trace(intent="go_to_A", destination="A", ambiguous=False, control=True)
        if not self.plan_success:
            print("Failed to move up after placing the stapler at the grid B.")
            return self.info

        self._pick_object(arm_tag)
        self._set_aliasbench_trace(intent="go_to_A", destination="A", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the stapler from the grid B.")
            return self.info
        self._place_object(arm_tag, self.grid_a_center)
        if not self.plan_success:
            print("Failed to place the stapler at the grid A.")
            return self.info
        self.eval_returned_a = True
        
        obj_pos = self.object.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination="A", ambiguous=False, control=False)

        self.info["info"] = {"{A}": f"048_stapler/base{self.object_id}"}
        return self.info

    def _pick_object(self, arm_tag):
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))

    def _place_object(self, arm_tag, grid_center):
        self.move(self.place_actor(self.object, target_pose=grid_center.tolist() + [0.707,0,0,0.707], arm_tag=arm_tag, pre_dis=0.1, dis=0.0, constrain="align"))

    def check_success(self):
        pos = self.object.get_pose().p
        at_b = np.all(np.abs(pos[:2] - self.grid_b_center[:2]) < np.array([0.08, 0.08]))
        at_a = np.all(np.abs(pos[:2] - self.grid_a_center[:2]) < np.array([0.03, 0.03]))
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
