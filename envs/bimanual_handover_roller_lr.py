from ._base_task import Base_Task
from .utils import *
from ._ais_benchmark_common import sample_pad_color, use_fixed_aliasbench_colors, fixed_pad_colors, PAD_COLOR_POOL
import sapien
import numpy as np


class bimanual_handover_roller_lr(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.pad_half_size = [0.12, 0.03, 0.00125]
        if use_fixed_aliasbench_colors():
            sampled_left_idx = np.random.randint(0, len(PAD_COLOR_POOL))
            sampled_right_idx = np.random.randint(0, len(PAD_COLOR_POOL))
            while np.linalg.norm(np.array(PAD_COLOR_POOL[sampled_left_idx]) - np.array(PAD_COLOR_POOL[sampled_right_idx])) < 0.35:
                sampled_right_idx = np.random.randint(0, len(PAD_COLOR_POOL))
            self.left_pad_color, self.right_pad_color = fixed_pad_colors(2)
        else:
            sampled_left_pad_color = sample_pad_color()
            sampled_right_pad_color = sample_pad_color()
            while np.linalg.norm(np.array(sampled_left_pad_color) - np.array(sampled_right_pad_color)) < 0.35:
                sampled_right_pad_color = sample_pad_color()
            self.left_pad_color = sampled_left_pad_color
            self.right_pad_color = sampled_right_pad_color

        self.left_pad_center = np.array([-0.20, -0.08, 0.741 + self.table_z_bias])
        self.right_pad_center = np.array([0.20, -0.08, 0.741 + self.table_z_bias])
        self.middle_center = np.array([0.0, -0.08, 0.80 + self.table_z_bias])

        self.left_pad = self._create_pad(self.left_pad_center, self.left_pad_color, 'left_pad')
        self.right_pad = self._create_pad(self.right_pad_center, self.right_pad_color, 'right_pad')

        self.start_side = 'left' if np.random.randint(0, 2) == 0 else 'right'
        self.roller_id = 2
        start_center = self.left_pad_center if self.start_side == 'left' else self.right_pad_center
        roller_pose = sapien.Pose(start_center.tolist(), [0, 0, 0.707, 0.707])
        self.roller = create_actor(
            scene=self,
            pose=roller_pose,
            modelname='102_roller',
            convex=True,
            model_id=self.roller_id,
        )

        if self.start_side == 'left':
            self.first_arm_tag = ArmTag('left')
            self.second_arm_tag = ArmTag('right')
            self.start_pad = self.left_pad
            self.final_pad = self.right_pad
            self.final_pad_center = self.right_pad_center
            self.first_contact_id = 0
            self.second_contact_id = 1
        else:
            self.first_arm_tag = ArmTag('right')
            self.second_arm_tag = ArmTag('left')
            self.start_pad = self.right_pad
            self.final_pad = self.left_pad
            self.final_pad_center = self.left_pad_center
            self.first_contact_id = 1
            self.second_contact_id = 0

        self.eval_first_grasp = False
        self.eval_visited_middle = False
        self.eval_second_grasp = False
        self.eval_handover_release = False
        self.eval_reached_final = False

        self.add_prohibit_area(self.left_pad, padding=0.08)
        self.add_prohibit_area(self.right_pad, padding=0.08)
        self.add_prohibit_area(self.roller, padding=0.06)

    def _create_pad(self, center, color, name):
        return create_box(
            scene=self,
            pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
            half_size=self.pad_half_size,
            color=color,
            name=name,
            is_static=True,
        )

    def play_once(self):
        final_side = "right" if self.start_side == "left" else "left"
        self._init_aliasbench_trace(intent=f"handover_to_{final_side}", destination="middle", ambiguous=False, control=True)
        self.move(self.back_to_origin(arm_tag=self.first_arm_tag))
        self.move(self.back_to_origin(arm_tag=self.second_arm_tag))

        if not self._first_arm_pick_and_transport():
            return self.info
        if not self._second_arm_handover_and_place():
            return self.info

        self._set_aliasbench_trace(intent=f"handover_to_{final_side}", destination=final_side, ambiguous=True, control=False)
        self.move(self.back_to_origin(arm_tag=self.second_arm_tag))
        if self.plan_success:
            self._set_aliasbench_trace(intent="done", destination=final_side, ambiguous=False, control=False)
            self.info['info'] = {'{A}': f'102_roller/base{self.roller_id}'}
        return self.info

    def _first_arm_pick_and_transport(self):
        self.move(
            self.grasp_actor(
                self.roller,
                arm_tag=self.first_arm_tag,
                pre_grasp_dis=0.08,
                grasp_dis=0.0,
                gripper_pos=0,
                contact_point_id=self.first_contact_id,
            )
        )
        if not self.plan_success:
            return False

        self.move(self.move_by_displacement(arm_tag=self.first_arm_tag, z=0.08, move_axis='world'))
        if not self.plan_success:
            return False

        roller_pos = self.roller.get_pose().p
        dx_to_middle = -roller_pos[0]
        self.move(self.move_by_displacement(arm_tag=self.first_arm_tag, x=dx_to_middle, move_axis='world'))
        if not self.plan_success:
            return False

        return True

    def _second_arm_handover_and_place(self):
        grasp_action = self.grasp_actor(
            self.roller,
            arm_tag=self.second_arm_tag,
            pre_grasp_dis=0.07,
            grasp_dis=0.0,
            gripper_pos=0,
            contact_point_id=self.second_contact_id,
        )
        self.move(grasp_action)
        if not self.plan_success:
            return False

        self.move(self.open_gripper(self.first_arm_tag))
        if not self.plan_success:
            return False

        self.move(self.back_to_origin(arm_tag=self.first_arm_tag))
        if not self.plan_success:
            return False

        self.move(self.move_by_displacement(arm_tag=self.second_arm_tag, z=0.04, move_axis='world'))
        if not self.plan_success:
            return False

        roller_pos = self.roller.get_pose().p
        dx_to_final = self.final_pad_center[0] - roller_pos[0]
        self.move(self.move_by_displacement(arm_tag=self.second_arm_tag, x=dx_to_final, move_axis='world'))
        if not self.plan_success:
            return False

        # Lower the roller onto the pad before release instead of dropping it
        # from the handover height, which can fling it beyond the target.
        dz_to_pad = self.final_pad_center[2] + 0.005 - self.roller.get_pose().p[2]
        self.move(self.move_by_displacement(arm_tag=self.second_arm_tag, z=dz_to_pad, move_axis='world'))
        if not self.plan_success:
            return False

        self.move(self.open_gripper(self.second_arm_tag))
        if not self.plan_success:
            return False
        self.move(self.move_by_displacement(arm_tag=self.second_arm_tag, z=0.06, move_axis='world'))

        return self.plan_success

    def move(self, *args, **kwargs):
        result = super().move(*args, **kwargs)
        # Expert moves do not call check_success at each physics step.
        self._update_handover_progress()
        return result

    def _roller_gripper_contacts(self, side):
        """Count distinct finger links touching the roller for one arm.

        Use entities, not link names: two single-arm robots can share names.
        """
        fingers = [joint.child_link.entity for joint, _, _ in getattr(self.robot, f"{side}_gripper")]
        articulation = getattr(self.robot, f"{side}_entity")
        fixed_names = getattr(self.robot, f"{side}_fix_gripper_name")
        for link in articulation.get_links():
            if link.name in fixed_names and link.entity not in fingers:
                fingers.append(link.entity)
        touching = set()
        for contact in self.scene.get_contacts():
            if not contact.points:
                continue
            a, b = (body.entity for body in contact.bodies)
            other = b if a == self.roller.actor else a if b == self.roller.actor else None
            for index, finger in enumerate(fingers):
                if other == finger:
                    touching.add(index)
        return len(touching) >= 2

    def _update_handover_progress(self):
        pos = self.roller.get_pose().p
        first = str(self.first_arm_tag)
        second = str(self.second_arm_tag)
        first_open = getattr(self, f"is_{first}_gripper_open")()
        second_open = getattr(self, f"is_{second}_gripper_open")()
        first_holds = not first_open and self._roller_gripper_contacts(first)
        second_holds = not second_open and self._roller_gripper_contacts(second)
        at_middle = (abs(pos[0] - self.middle_center[0]) < 0.03
                     and abs(pos[1] - self.middle_center[1]) < 0.07
                     and pos[2] > self.middle_center[2] - 0.03)

        if first_holds:
            self.eval_first_grasp = True
        if self.eval_first_grasp and first_holds and at_middle:
            self.eval_visited_middle = True
        if self.eval_visited_middle and at_middle and first_holds and second_holds:
            self.eval_second_grasp = True
        if self.eval_second_grasp and first_open and second_holds:
            self.eval_handover_release = True

        at_final = (np.all(np.abs(pos[:2] - self.final_pad_center[:2]) < np.array([0.07, 0.07]))
                    and abs(pos[2] - self.final_pad_center[2]) < 0.04)
        if self.eval_handover_release and at_final and first_open and second_open:
            self.eval_reached_final = True
        return at_final

    def check_success(self):
        at_final = self._update_handover_progress()
        return bool(
            self.eval_first_grasp
            and self.eval_visited_middle
            and self.eval_second_grasp
            and self.eval_handover_release
            and self.eval_reached_final
            and at_final
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
