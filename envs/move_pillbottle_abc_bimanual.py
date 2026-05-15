from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._ais_benchmark_common import use_fixed_aliasbench_colors, fixed_pad_colors


class move_pillbottle_abc_bimanual(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)

    def load_actors(self):
        self.pad_half_size = [0.04, 0.04, 0.0005]
        self.pad_color = self._sample_pad_color()

        self.pad_a_center = np.array([-0.18, -0.08, 0.741 + self.table_z_bias])
        self.pad_b_center = np.array([0.0, -0.08, 0.741 + self.table_z_bias])
        self.pad_c_center = np.array([0.18, -0.08, 0.741 + self.table_z_bias])

        self.pad_a = self._create_pad(self.pad_a_center, "pad_A")
        self.pad_b = self._create_pad(self.pad_b_center, "pad_B")
        self.pad_c = self._create_pad(self.pad_c_center, "pad_C")

        self.start_pad_name = "A" if np.random.randint(0, 2) == 0 else "C"
        pill_pose = sapien.Pose(
            (self.pad_a_center if self.start_pad_name == "A" else self.pad_c_center).tolist(),
            [0.5, 0.5, 0.5, 0.5],
        )

        self.pillbottle_id = np.random.choice([1, 2, 3, 4, 5], 1)[0]
        self.pillbottle = create_actor(
            scene=self,
            pose=pill_pose,
            modelname="080_pillbottle",
            convex=True,
            model_id=self.pillbottle_id,
        )
        self.pillbottle.set_mass(0.05)

        if self.start_pad_name == "A":
            self.first_arm_tag = ArmTag("left")
            self.second_arm_tag = ArmTag("right")
            self.final_pad = self.pad_c
            self.final_pad_center = self.pad_c_center
        else:
            self.first_arm_tag = ArmTag("right")
            self.second_arm_tag = ArmTag("left")
            self.final_pad = self.pad_a
            self.final_pad_center = self.pad_a_center

        self.eval_visited_middle = False
        self.eval_reached_final = False

        self.add_prohibit_area(self.pad_a, padding=0.08)
        self.add_prohibit_area(self.pad_b, padding=0.08)
        self.add_prohibit_area(self.pad_c, padding=0.08)
        self.add_prohibit_area(self.pillbottle, padding=0.05)

    def _create_pad(self, center, name):
        return create_box(
            scene=self,
            pose=sapien.Pose(center.tolist(), [1, 0, 0, 0]),
            half_size=self.pad_half_size,
            color=self.pad_color,
            name=name,
            is_static=True,
        )

    def _sample_pad_color(self):
        color_pool = [
            (0.12, 0.58, 0.95),
            (0.24, 0.78, 0.36),
            (0.72, 0.32, 0.92),
            (0.95, 0.22, 0.18),
            (0.95, 0.86, 0.16),
            (0.95, 0.55, 0.12),
            (0.95, 0.28, 0.52),
            (0.10, 0.75, 0.72),
        ]
        sampled_idx = np.random.randint(0, len(color_pool))
        if use_fixed_aliasbench_colors():
            return fixed_pad_colors(1)[0]
        return color_pool[sampled_idx]

    def play_once(self):
        final_name = "C" if self.start_pad_name == "A" else "A"
        second_arm_name = str(self.second_arm_tag)
        self._init_aliasbench_trace(intent=f"continue_with_{second_arm_name}", destination="B", ambiguous=False, control=True)
        self.move(self.back_to_origin(arm_tag=self.first_arm_tag))
        self.move(self.back_to_origin(arm_tag=self.second_arm_tag))

        self._move_pillbottle_between_pads(
            arm_tag=self.first_arm_tag,
            target_pad=self.pad_b,
            lift_height=0.05,
        )
        if not self.plan_success:
            print("Failed to move the pillbottle to the middle pad.")
            return self.info
        self.eval_visited_middle = True
        self._set_aliasbench_trace(intent=f"continue_with_{second_arm_name}", destination=final_name, ambiguous=True, control=False)
        
        self.move(self.back_to_origin(arm_tag=self.first_arm_tag))
        if not self.plan_success:
            print("Failed to return the first arm to the origin.")
            return self.info

        self._move_pillbottle_between_pads(
            arm_tag=self.second_arm_tag,
            target_pad=self.final_pad,
            lift_height=0.05,
        )
        if not self.plan_success:
            print("Failed to move the pillbottle to the final pad.")
            return self.info
        self.eval_reached_final = True
        
        self.move(self.back_to_origin(arm_tag=self.second_arm_tag))
        self._set_aliasbench_trace(intent="done", destination=final_name, ambiguous=False, control=False)

        self.info["info"] = {
            "{A}": f"080_pillbottle/base{self.pillbottle_id}",
        }
        return self.info

    def _move_pillbottle_between_pads(self, arm_tag, target_pad, lift_height):
        target_is_middle = target_pad is self.pad_b
        self.move(
            self.grasp_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                pre_grasp_dis=0.06,
                gripper_pos=0,
            ))
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_height))
        if not self.plan_success:
            return
        self.move(
            self.place_actor(
                self.pillbottle,
                arm_tag=arm_tag,
                target_pose=target_pad.get_functional_point(1),
                pre_dis=0.05,
                dis=0.0,
                functional_point_id=0,
                pre_dis_axis="fp",
            ))
        if not self.plan_success:
            return
        pillbottle_pos = self.pillbottle.get_pose().p
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.06, move_axis="arm"))

    def check_success(self):
        pos = self.pillbottle.get_pose().p
        at_middle = np.all(np.abs(pos[:2] - self.pad_b_center[:2]) < np.array([0.08, 0.08]))
        at_final = np.all(np.abs(pos[:2] - self.final_pad_center[:2]) < np.array([0.05, 0.05]))
        if (not self.eval_visited_middle) and at_middle:
            self.eval_visited_middle = True
        if self.eval_visited_middle and at_final:
            self.eval_reached_final = True
        final_ok = (
            self.eval_visited_middle
            and self.eval_reached_final
            and at_final
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        return final_ok
