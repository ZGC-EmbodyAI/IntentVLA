from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._ais_benchmark_common import use_fixed_aliasbench_colors, fixed_pad_colors


class place_block_aba_grid(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.block_half_size = 0.025
        self.grid_half_size = [0.04, 0.04, 0.002]
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1

        self.grid_a_center = np.array([x_sign * 0.03, -0.14, 0.742])
        self.grid_b_center = np.array([x_sign * 0.33, -0.02, 0.742])
        self.grid_colors = self._sample_grid_colors()

        self.grid_a = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]),
            half_size=self.grid_half_size,
            color=self.grid_colors[0],
            name="grid_A",
        )
        self.grid_b = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.grid_b_center.tolist(), [1, 0, 0, 0]),
            half_size=self.grid_half_size,
            color=self.grid_colors[1],
            name="grid_B",
        )

        block_center = self.grid_a_center.copy()
        block_center[2] = 0.741 + self.block_half_size
        self.block = create_box(
            scene=self,
            pose=sapien.Pose(block_center.tolist(), [1, 0, 0, 0]),
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1, 0.15, 0.15),
            name="target_block",
        )
        self.block.set_mass(0.03)
        self.eval_visited_b = False
        self.eval_returned_a = False

        self.add_prohibit_area(sapien.Pose(self.grid_a_center.tolist(), [1, 0, 0, 0]), padding=0.09)
        self.add_prohibit_area(sapien.Pose(self.grid_b_center.tolist(), [1, 0, 0, 0]), padding=0.09)

    def _sample_grid_colors(self):
        color_pool = [
            (0.12, 0.58, 0.95),
            (0.95, 0.55, 0.12),
            (0.24, 0.78, 0.36),
            (0.72, 0.32, 0.92),
            (0.95, 0.86, 0.16),
            (0.95, 0.28, 0.52),
        ]
        color_ids = np.random.choice(len(color_pool), 2, replace=False)
        if use_fixed_aliasbench_colors():
            return fixed_pad_colors(2)
        return [color_pool[color_ids[0]], color_pool[color_ids[1]]]

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent="go_to_B", destination="B", ambiguous=False, control=True)
        self._pick_block(arm_tag)
        self._set_aliasbench_trace(intent="go_to_B", destination="B", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the block first time.")
            return self.info

        self._place_block(arm_tag, self.grid_b_center)
        if not self.plan_success:
            print("Failed to place the block at grid B.")
            return self.info
        self.eval_visited_b = True
        
        block_pos = self.block.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self._set_aliasbench_trace(intent="go_to_A", destination="A", ambiguous=False, control=True)
        if not self.plan_success:
            print("Failed to move the arm.")
            return self.info

        self._pick_block(arm_tag)
        self._set_aliasbench_trace(intent="go_to_A", destination="A", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to pick the block second time.")
            return self.info

        self._place_block(arm_tag, self.grid_a_center)
        if not self.plan_success:
            print("Failed to place the block at grid A.")
            return self.info
        self.eval_returned_a = True
        
        block_pos = self.block.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination="A", ambiguous=False, control=False)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": f"{arm_tag} hand",
        }
        return self.info

    def _pick_block(self, arm_tag):
        self.move(
            self.grasp_actor(
                self.block,
                arm_tag=arm_tag,
                pre_grasp_dis=0.09,
                grasp_dis=0.01,
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))

    def _place_block(self, arm_tag, grid_center):
        target_center = grid_center.copy()
        target_center[2] += self.table_z_bias
        target_pose = target_center.tolist() + [0, 1, 0, 0]
        self.move(
            self.place_actor(
                self.block,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.10,
                dis=0.02,
                constrain="free",
                pre_dis_axis="fp",
            ))

    def check_success(self):
        block_pos = self.block.get_pose().p
        # print("check_success", block_pos, self.grid_a_center, self.grid_b_center)
        at_b = np.all(np.abs(block_pos[:2] - self.grid_b_center[:2]) < np.array([0.08, 0.08]))
        at_a = np.all(np.abs(block_pos[:2] - self.grid_a_center[:2]) < np.array([0.045, 0.045]))
        # print("check_success", at_b, at_a)
        if (not self.eval_visited_b) and at_b:
            self.eval_visited_b = True
        if self.eval_visited_b and at_a:
            self.eval_returned_a = True
        final_ok = (
            self.eval_visited_b
            and self.eval_returned_a
            and at_a
            and abs(block_pos[2] - (0.741 + self.block_half_size + self.table_z_bias)) < 0.025
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        return final_ok
