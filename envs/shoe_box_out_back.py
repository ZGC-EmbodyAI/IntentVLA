from ._base_task import Base_Task
from .utils import *
from ._ais_benchmark_common import sample_distinct_pad_colors
import sapien
import numpy as np


class shoe_box_out_back(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.pad_half_size = [0.13, 0.05, 0.0005]
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1

        self.grid_a_center = np.array([x_sign * 0.08, -0.24, 0.742])
        self.grid_b_center = np.array([x_sign * 0.22, -0.02, 0.742])
        self.grid_colors = sample_distinct_pad_colors(1)

        self.grid_a = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]),
            half_size=self.pad_half_size,
            color=self.grid_colors[0],
            name="grid_A",
        )
        self.target_block = create_box(scene=self, pose=sapien.Pose(self.grid_a_center.tolist(), [1,0,0,0]), half_size=(0.13, 0.05, 0.0005), color=self.grid_colors[0], is_static=True, name="box")
        self.target_block.config["functional_matrix"] = [[[0.0,-1.0,0.0,0.0],[-1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0],[0.0,0.0,0.0,1.0]],[[0.0,-1.0,0.0,0.0],[-1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,0],[0.0,0.0,0.0,1.0]]]
        self.shoe_box = create_actor(self, pose=sapien.Pose(self.grid_b_center.tolist(), [0.5,0.5,-0.5,-0.5]), modelname="007_shoe-box", convex=True, is_static=True)
        self.object_id = np.random.choice([i for i in range(10)])
        shoe_center = self.grid_a_center.copy()
        shoe_center[2] = 0.741
        self.object = create_actor(scene=self, pose=sapien.Pose(shoe_center.tolist(), [0.5,0.5,-0.5,-0.5]), modelname="041_shoe", convex=True, model_id=self.object_id)

        self.eval_visited_b = False
        self.eval_returned_a = False

        self.add_prohibit_area(sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]), padding=0.09)
        self.add_prohibit_area(sapien.Pose(self.grid_b_center.tolist(), [1, 0, 0, 0]), padding=0.09)
        self.add_prohibit_area(self.object, padding=0.1)
        self.add_prohibit_area(self.shoe_box, padding=0.1)

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent="go_into_box", destination="box", ambiguous=False, control=True)
        self._pick_object(arm_tag)
        self._set_aliasbench_trace(intent="go_into_box", destination="box", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the shoe from the grid A.")
            return self.info
        self._place_object(arm_tag, self.grid_b_center)
        if not self.plan_success:
            print("Failed to place the shoe at the grid B.")
            return self.info
        self.eval_visited_b = True
        
        obj_pos = self.object.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="world"))
        self._set_aliasbench_trace(intent="go_outside", destination="outside", ambiguous=False, control=True)
        if not self.plan_success:
            print("Failed to move the arm after placing the shoe.")
            return self.info

        self._pick_object(arm_tag)
        self._set_aliasbench_trace(intent="go_outside", destination="outside", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the shoe from the grid B.")
            return self.info
        self._place_object(arm_tag, self.grid_a_center)
        if not self.plan_success:
            print("Failed to place the shoe back at the grid A.")
            return self.info
        self.eval_returned_a = True
        
        obj_pos = self.object.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="world"))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination="outside", ambiguous=False, control=False)

        self.info["info"] = {"{A}": f"041_shoe/base{self.object_id}", "{B}": "007_shoe-box/base0"}
        return self.info

    def _pick_object(self, arm_tag):
        self.move(self.grasp_actor(self.object, arm_tag=arm_tag, pre_grasp_dis=0.12, gripper_pos=0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="world"))

    def _place_object(self, arm_tag, grid_center):
        if np.allclose(grid_center[:2], self.grid_b_center[:2]):
            target_pose = self.shoe_box.get_functional_point(0)
            self._move_object_over_target(arm_tag, target_pose, hover_z=0.18)
            self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose, functional_point_id=0, pre_dis=0.18, constrain="align"))
        else:
            target_pose = self.target_block.get_functional_point(0)
            self._move_object_over_target(arm_tag, target_pose, hover_z=0.14)
            self.move(self.place_actor(self.object, arm_tag=arm_tag, target_pose=target_pose, functional_point_id=0, pre_dis=0.14, constrain="align"))
            self.move(self.open_gripper(arm_tag=arm_tag))

    def _move_object_over_target(self, arm_tag, target_pose, hover_z):
        if not self.plan_success:
            return
        ee_pose = np.array(self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(), dtype=np.float64)
        object_pose = np.array(self.object.get_pose().p, dtype=np.float64)
        target_pose = np.array(target_pose, dtype=np.float64)
        target_object_pos = target_pose[:3].copy()
        target_object_pos[2] += hover_z
        ee_pose[:3] = target_object_pos + (ee_pose[:3] - object_pose)
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=ee_pose.tolist()))

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
