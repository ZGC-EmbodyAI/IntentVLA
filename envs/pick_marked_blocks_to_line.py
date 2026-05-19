from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._ais_benchmark_common import use_fixed_aliasbench_colors, fixed_pad_colors, fixed_block_colors


class pick_marked_blocks_to_line(Base_Task):
    EVAL_MARKER_OBS_FRAMES = 50


    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)
        self.eval_marker_stage_idx = -1
        self.eval_marker_obs_remaining = self.EVAL_MARKER_OBS_FRAMES
        self._hide_marker()

    def load_actors(self):
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        x_sign = -1 if self.arm_tag == "left" else 1
        self.block_half_size = 0.025
        self.pad_half_size = [0.05, 0.05, 0.002]
        self.marker_half_size = [0.045, 0.045, 0.002]
        self.marker_color = self._sample_marker_color()
        self.marker_hide_pose = sapien.Pose([0, 0, -1.0], [1, 0, 0, 0])
        self.pad_color = self._sample_pad_color()

        source_xy = [
            [x_sign * 0.28, 0.02],
            [x_sign * 0.16, 0.02],
            [x_sign * 0.04, 0.02],
        ]
        self.source_centers = []
        for x, y in source_xy:
            rand_x = x + np.random.uniform(-0.01, 0.01)
            rand_y = y + np.random.uniform(-0.01, 0.01)
            self.source_centers.append(np.array([rand_x, rand_y, 0.741 + self.block_half_size]))

        self.target_centers = [
            np.array([x_sign * 0.20, -0.10, 0.742]),
            np.array([x_sign * 0.12, -0.10, 0.742]),
            np.array([x_sign * 0.04, -0.10, 0.742]),
        ]

        self.target_pads = []
        for idx, center in enumerate(self.target_centers):
            pad = create_visual_box(
                scene=self,
                pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
                half_size=self.pad_half_size,
                color=self.pad_color,
                name=f"target_pad_{idx}",
            )
            self.target_pads.append(pad)

        block_colors = self._sample_block_colors()
        self.blocks = []
        self.block_infos = []
        for idx, (center, (color_name, color)) in enumerate(zip(self.source_centers, block_colors)):
            block = create_box(
                scene=self,
                pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
                half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
                color=color,
                name=f"marked_block_{idx}",
            )
            block.set_mass(0.03)
            self.blocks.append(block)
            self.block_infos.append(color_name)

        self.marker = create_visual_box(
            scene=self,
            pose=self.marker_hide_pose,
            half_size=self.marker_half_size,
            color=self.marker_color,
            name="target_marker",
        )

        self.target_order = np.random.permutation(len(self.blocks)).tolist()
        self.eval_stage_idx = 0
        self.eval_marker_obs_remaining = self.EVAL_MARKER_OBS_FRAMES

        for center in self.target_centers:
            self.add_prohibit_area(sapien.Pose(center.tolist(), [1, 0, 0, 0]), padding=0.08)

    def _sample_marker_color(self):
        color_pool = [
            (0.95, 0.22, 0.18),
            (0.95, 0.40, 0.20),
            (0.72, 0.32, 0.92),
            (0.10, 0.75, 0.72),
            (0.10, 0.35, 0.95),
        ]
        sampled_idx = np.random.randint(0, len(color_pool))
        if use_fixed_aliasbench_colors():
            return fixed_pad_colors(1)[0]
        return color_pool[sampled_idx]

    def _sample_pad_color(self):
        color_pool = [
            (0.12, 0.58, 0.95),
            (0.24, 0.78, 0.36),
            (0.72, 0.32, 0.92),
            (0.95, 0.22, 0.18),
            (0.95, 0.86, 0.16),
            (0.95, 0.55, 0.12),
        ]
        sampled_idx = np.random.randint(0, len(color_pool))
        if use_fixed_aliasbench_colors():
            return fixed_pad_colors(2)[1]
        return color_pool[sampled_idx]

    def _sample_block_colors(self):
        color_pool = [
            ("blue", (0.12, 0.58, 0.95)),
            ("green", (0.24, 0.78, 0.36)),
            ("purple", (0.72, 0.32, 0.92)),
            ("red", (0.95, 0.22, 0.18)),
            ("yellow", (0.95, 0.86, 0.16)),
            ("orange", (0.95, 0.55, 0.12)),
            ("cyan", (0.10, 0.75, 0.72)),
            ("pink", (0.95, 0.35, 0.70)),
        ]
        color_ids = np.random.choice(len(color_pool), 3, replace=False)
        if use_fixed_aliasbench_colors():
            return fixed_block_colors(3)
        return [color_pool[color_id] for color_id in color_ids]

    def play_once(self):
        self._init_aliasbench_trace(intent="idle", destination="none", ambiguous=False, control=False)
        self.move(self.back_to_origin(arm_tag=self.arm_tag))
        for stage_idx, target_id in enumerate(self.target_order):
            block = self.blocks[target_id]
            self._set_aliasbench_trace(intent=f"target_block_{target_id}", destination=f"slot_{stage_idx}", ambiguous=False, control=True)
            if self.eval_marker_stage_idx != stage_idx:
                self._show_eval_marker_for_stage(stage_idx)

            self._set_aliasbench_trace(intent=f"target_block_{target_id}", destination=f"slot_{stage_idx}", ambiguous=True, control=False)
            self.move(
                self.grasp_actor(
                    block,
                    arm_tag=self.arm_tag,
                    pre_grasp_dis=0.09,
                    grasp_dis=0.01,
                ))
            if not self.plan_success:
                print(f"Failed to grasp the block {stage_idx}.")
                return self.info
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.10))
            if not self.plan_success:
                print(f"Failed to lift the block {stage_idx}.")
                return self.info

            target_center = self.target_centers[stage_idx].copy()
            target_center[2] += self.table_z_bias
            target_pose = target_center.tolist() + [0, 1, 0, 0]
            self.move(
                self.place_actor(
                    block,
                    arm_tag=self.arm_tag,
                    target_pose=target_pose,
                    functional_point_id=0,
                    pre_dis=0.09,
                    dis=0.02,
                    constrain="free",
                    pre_dis_axis="fp",
                ))
            if not self.plan_success:
                print(f"Failed to place the block {stage_idx} at the target position.")
                return self.info
            self.move(self.move_by_displacement(arm_tag=self.arm_tag, z=0.07, move_axis="arm"))
            self.move(self.back_to_origin(arm_tag=self.arm_tag))
            if not self.plan_success:
                print(f"Failed to return the arm to the origin after placing block {stage_idx}.")
                return self.info

            self.eval_stage_idx = stage_idx + 1

        self._set_aliasbench_trace(intent="done", destination="complete", ambiguous=False, control=False)
        self.info["info"] = {
            "{A}": "briefly marked blocks",
            "{B}": "line of colored pads",
            "{C}": f"{self.arm_tag} hand",
        }
        return self.info

    def _set_marker_visible_for_actor(self, actor):
        marker_pose = actor.get_pose().p.copy()
        marker_pose[2] = 0.742 + self.table_z_bias
        self.marker.set_pose(sapien.Pose(marker_pose.tolist(), [1, 0, 0, 0]))

    def _hide_marker(self):
        self.marker.set_pose(self.marker_hide_pose)

    def _show_eval_marker_for_stage(self, stage_idx):
        if stage_idx < 0 or stage_idx >= len(self.target_order):
            self._hide_marker()
            return
        target_id = self.target_order[stage_idx]
        self._show_marker(self.blocks[target_id])
        self.eval_marker_stage_idx = stage_idx

    def _show_marker(self, block):
        self._set_marker_visible_for_actor(block)
        self._wait_marker_steps(500)
        self._hide_marker()

    def _wait_marker_steps(self, step_num):
        save_freq = self.save_freq
        for step_idx in range(step_num):
            self.scene.step()
            if self.render_freq and step_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and step_idx % save_freq == 0:
                self._update_render()
                self._take_picture()
        if save_freq is not None:
            self._take_picture()

    def get_obs(self):
        if self.eval_mode and self.eval_stage_idx < len(self.target_order) and self.eval_marker_obs_remaining > 0:
            target_id = self.target_order[self.eval_stage_idx]
            self._set_marker_visible_for_actor(self.blocks[target_id])
            obs = super().get_obs()
            self._hide_marker()
            self.eval_marker_obs_remaining -= 1
            return obs
        return super().get_obs()

    def check_success(self):
        expected_z = 0.741 + self.block_half_size + self.table_z_bias
        if self.eval_stage_idx < len(self.target_order):
            stage_idx = self.eval_stage_idx
            target_id = self.target_order[stage_idx]
            block_pos = self.blocks[target_id].get_pose().p
            target_pos = self.target_centers[stage_idx]
            stage_ok = (
                np.all(np.abs(block_pos[:2] - target_pos[:2]) < np.array([0.06, 0.06]))
                and abs(block_pos[2] - expected_z) < 0.035
                and self.is_left_gripper_open()
                and self.is_right_gripper_open()
            )
            if stage_ok:
                self.eval_stage_idx += 1
                if self.eval_stage_idx < len(self.target_order):
                    self.eval_marker_obs_remaining = self.EVAL_MARKER_OBS_FRAMES
                else:
                    self._hide_marker()

        final_ok = True
        for stage_idx, target_id in enumerate(self.target_order):
            block_pos = self.blocks[target_id].get_pose().p
            target_pos = self.target_centers[stage_idx]
            if not np.all(np.abs(block_pos[:2] - target_pos[:2]) < np.array([0.045, 0.045])):
                final_ok = False
                break
            if abs(block_pos[2] - expected_z) >= 0.025:
                final_ok = False
                break
        final_ok = final_ok and self.is_left_gripper_open() and self.is_right_gripper_open()
        return final_ok and self.eval_stage_idx >= len(self.target_order)
