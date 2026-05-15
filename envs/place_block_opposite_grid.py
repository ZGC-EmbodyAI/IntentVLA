from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._ais_benchmark_common import use_fixed_aliasbench_colors, fixed_pad_colors


class place_block_opposite_grid(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.block_half_size = 0.025
        self.grid_half_size = [0.035, 0.035, 0.002]
        self.grid_color = self._sample_grid_color()
        self.grid_names = ["A", "B", "C", "D"]
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1
        self.grid_centers = {
            "A": np.array([x_sign * 0.02, -0.14, 0.742]),
            "B": np.array([x_sign * 0.24, -0.14, 0.742]),
            "C": np.array([x_sign * 0.02, 0.00, 0.742]),
            "D": np.array([x_sign * 0.24, 0.00, 0.742]),
        }
        self.opposite_grid = {
            "A": "D",
            "D": "A",
            "B": "C",
            "C": "B",
        }

        self.grids = {}
        for grid_name, center in self.grid_centers.items():
            grid_pose = sapien.Pose(center.tolist(), [1, 0, 0, 0])
            self.grids[grid_name] = create_visual_box(
                scene=self,
                pose=grid_pose,
                half_size=self.grid_half_size,
                color=self.grid_color,
                name=f"grid_{grid_name}",
            )

        self.start_grid_name = self.grid_names[np.random.randint(0, len(self.grid_names))]
        self.target_grid_name = self.opposite_grid[self.start_grid_name]
        block_center = self.grid_centers[self.start_grid_name].copy()
        block_center[2] = 0.741 + self.block_half_size
        block_pose = sapien.Pose(block_center.tolist(), [1, 0, 0, 0])
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1, 0.15, 0.15),
            name="target_block",
        )
        self.block.set_mass(0.03)

        for center in self.grid_centers.values():
            self.add_prohibit_area(sapien.Pose(center.tolist(), [1, 0, 0, 0]), padding=0.09)

    def _sample_grid_color(self):
        color_pool = [
            (0.12, 0.58, 0.95),
            (0.24, 0.78, 0.36),
            (0.72, 0.32, 0.92),
            (0.95, 0.22, 0.18),
            (0.95, 0.86, 0.16),
            (0.95, 0.55, 0.12),
            (0.10, 0.75, 0.72),
            (0.95, 0.35, 0.70),
            (0.55, 0.35, 0.18),
            (0.82, 0.82, 0.82),
        ]
        sampled_idx = np.random.randint(0, len(color_pool))
        if use_fixed_aliasbench_colors():
            return fixed_pad_colors(1)[0]
        return color_pool[sampled_idx]

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent=f"go_to_{self.target_grid_name}", destination=self.target_grid_name, ambiguous=False, control=True)

        self.move(
            self.grasp_actor(
                self.block,
                arm_tag=arm_tag,
                pre_grasp_dis=0.09,
                grasp_dis=0.01,
            ))
        if not self.plan_success:
            print("Failed to grasp the block.")
            return self.info
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10))
        self._set_aliasbench_trace(intent=f"go_to_{self.target_grid_name}", destination=self.target_grid_name, ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to lift the block.")
            return self.info

        target_center = self.grid_centers[self.target_grid_name].copy()
        target_center[2] += self.table_z_bias
        target_pose = target_center.tolist() + [0, 1, 0, 0]
        self.move(
            self.place_actor(
                self.block,
                arm_tag=arm_tag,
                target_pose=target_pose,
                functional_point_id=0,
                pre_dis=0.09,
                dis=0.02,
                constrain="free",
                pre_dis_axis="fp",
            ))
        if not self.plan_success:
            print("Failed to place the block at the target grid.")
            return self.info
        block_pos = self.block.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination=self.target_grid_name, ambiguous=False, control=False)

        self.info["info"] = {
            "{A}": "red block",
            "{B}": f"{arm_tag} hand",
        }
        return self.info

    def check_success(self):
        block_pos = self.block.get_pose().p
        target_pos = self.grid_centers[self.target_grid_name]
        final_ok = (
            np.all(np.abs(block_pos[:2] - target_pos[:2]) < np.array([0.045, 0.045]))
            and abs(block_pos[2] - (0.741 + self.block_half_size + self.table_z_bias)) < 0.025
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        return final_ok

