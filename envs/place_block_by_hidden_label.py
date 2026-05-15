from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import sapien
import numpy as np
import transforms3d as t3d
from ._ais_benchmark_common import use_fixed_aliasbench_colors


class place_block_by_hidden_label(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.arm_tag = ArmTag("left" if np.random.randint(0, 2) == 0 else "right")
        self.x_sign = -1 if self.arm_tag == "left" else 1
        self.block_half_size = np.array([0.040, 0.025, 0.060])
        self.pad_half_size = [0.065, 0.055, 0.002]
        self.marker_half_size = [0.032, 0.0006, 0.052]
        self.inspect_rotation_angle = np.pi
        self.color_pool = [
            (0.95, 0.18, 0.12),
            (0.12, 0.58, 0.95),
            (0.18, 0.75, 0.28),
            (0.72, 0.32, 0.92),
            (0.95, 0.45, 0.10),
            (0.05, 0.75, 0.85),
        ]
        color_ids = np.random.choice(len(self.color_pool), size=2, replace=False)
        if use_fixed_aliasbench_colors():
            self.marker_color = self.color_pool[0]
            self.distractor_color = self.color_pool[1]
        else:
            self.marker_color = self.color_pool[int(color_ids[0])]
            self.distractor_color = self.color_pool[int(color_ids[1])]
        self.correct_target_id = np.random.randint(0, 2)

        self.block_start_center = np.array([self.x_sign * 0.12, 0.02, 0.741 + self.block_half_size[2]])
        self.inspection_center = np.array([self.x_sign * 0.12, -0.04, 0.86 + self.table_z_bias])
        target_slots = [
            np.array([self.x_sign * 0.05, -0.15, 0.742]),
            np.array([self.x_sign * 0.20, -0.15, 0.742]),
        ]
        self.target_centers = target_slots
        self.target_center = self.target_centers[self.correct_target_id]
        self.eval_seen_marker_face = False
        self.marker_face_front_threshold = -0.5
        self.marker_face_lift_margin = 0.04

        self.target_pads = []
        for target_id, center in enumerate(self.target_centers):
            color = self.marker_color if target_id == self.correct_target_id else self.distractor_color
            self.target_pads.append(create_visual_box(
                scene=self,
                pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
                half_size=self.pad_half_size,
                color=color,
                name=f"target_color_{target_id}",
            ))

        self.block = self._create_back_labeled_block()
        self.block.set_mass(0.03)

        for center in self.target_centers:
            self.add_prohibit_area(sapien.Pose(center.tolist(), [1, 0, 0, 0]), padding=0.08)
        self.add_prohibit_area(sapien.Pose(self.block_start_center.tolist(), [1, 0, 0, 0]), padding=0.08)

    def _create_back_labeled_block(self):
        scene, pose = preprocess(self, sapien.Pose(self.block_start_center.tolist(), [1, 0, 0, 0]))
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            half_size=self.block_half_size.tolist(),
            material=scene.default_physical_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0], [1, 0, 0, 0]),
            half_size=self.block_half_size.tolist(),
            material=sapien.render.RenderMaterial(base_color=[0.82, 0.82, 0.82, 1]),
        )

        builder.add_box_visual(
            pose=sapien.Pose([0, self.block_half_size[1] + self.marker_half_size[1], 0], [1, 0, 0, 0]),
            half_size=self.marker_half_size,
            material=sapien.render.RenderMaterial(base_color=[*self.marker_color, 1]),
        )

        entity = builder.build(name="hidden_label_block")
        entity.set_pose(pose)
        data = {
            "center": [0, 0, 0],
            "extents": self.block_half_size.tolist(),
            "scale": self.block_half_size.tolist(),
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, -1], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0, 0.0], [0.0, 0, -1.0, 1], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3], [4, 5, 6, 7]],
            "contact_points_mask": [True, True],
            "target_point_description": ["The center point on the bottom of the box."],
        }
        return Actor(entity, data)


    def _update_eval_marker_face_seen(self):
        expected_z = 0.741 + self.block_half_size[2] + self.table_z_bias
        block_pos = self.block.get_pose().p
        if block_pos[2] <= expected_z + self.marker_face_lift_margin:
            return
        quat_wxyz = np.array(self.block.get_pose().q, dtype=float)
        rot = t3d.quaternions.quat2mat(quat_wxyz)
        marker_face_normal = rot @ np.array([0.0, 1.0, 0.0])
        if marker_face_normal[1] < self.marker_face_front_threshold:
            self.eval_seen_marker_face = True

    def play_once(self):
        arm_tag = self.arm_tag
        self._init_aliasbench_trace(intent="inspect_marker", destination="inspection", ambiguous=False, control=True)
        self.move(self.back_to_origin(arm_tag=arm_tag))
        if not self.plan_success:
            print("Failed to move the arm to the origin.")
            return self.info

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
        if not self.plan_success:
            print("Failed to lift the block.")
            return self.info

        current_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        inspection_pose = current_pose.copy()
        inspection_pose[0] = self.inspection_center[0]
        inspection_pose[1] = self.inspection_center[1]
        inspection_pose[2] = max(current_pose[2], self.inspection_center[2])
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=inspection_pose.tolist()))
        if not self.plan_success:
            print("Failed to move the block to the inspection position.")
            return self.info

        original_pose = np.array(self.get_arm_pose(arm_tag), dtype=float)
        inspect_pose = original_pose.copy()
        inspect_pose[3:] = self._rotate_quat(original_pose[3:], axis=[0, 0, 1], angle=-self.x_sign * self.inspect_rotation_angle)
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=inspect_pose.tolist()))
        if not self.plan_success:
            print("Failed to rotate the block for inspection.")
            return self.info
        self._update_eval_marker_face_seen()
        self._wait_steps(80)

        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=original_pose.tolist()))
        self._set_aliasbench_trace(intent=f"place_on_color_{self.correct_target_id}", destination=f"target_{self.correct_target_id}", ambiguous=True, control=False)
        if not self.plan_success:
            print("Failed to return the arm to the origin.")
            return self.info

        target_center = self.target_center.copy()
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
            print("Failed to place the block at the target position.")
            return self.info
        block_pos = self.block.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.07, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag=arm_tag))
        self._set_aliasbench_trace(intent="done", destination=f"target_{self.correct_target_id}", ambiguous=False, control=False)

        self.info["info"] = {
            "{A}": "back-color-marked block",
            "{B}": f"color target {self.correct_target_id}",
            "{C}": f"{arm_tag} hand",
        }
        return self.info

    def _rotate_quat(self, quat, axis, angle):
        delta_quat = t3d.quaternions.axangle2quat(axis, angle)
        rotated = t3d.quaternions.qmult(delta_quat, quat)
        return (rotated / np.linalg.norm(rotated)).tolist()

    def _wait_steps(self, step_num):
        save_freq = self.save_freq
        for step_idx in range(step_num):
            self.scene.step()
            self._update_eval_marker_face_seen()
            if self.render_freq and step_idx % self.render_freq == 0:
                self._update_render()
                self.viewer.render()
            if save_freq is not None and step_idx % save_freq == 0:
                self._update_render()
                self._take_picture()
        self._update_eval_marker_face_seen()
        if save_freq is not None:
            self._take_picture()

    def check_success(self):
        self._update_eval_marker_face_seen()
        block_pos = self.block.get_pose().p
        expected_z = 0.741 + self.block_half_size[2] + self.table_z_bias
        final_ok = (
            np.all(np.abs(block_pos[:2] - self.target_center[:2]) < np.array([0.045, 0.045]))
            and abs(block_pos[2] - expected_z) < 0.025
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        return final_ok and self.eval_seen_marker_face
