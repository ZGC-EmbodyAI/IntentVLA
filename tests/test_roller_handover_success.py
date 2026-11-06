"""Asset-independent tests for the roller handover success checker."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


TASK_FILE = Path(__file__).resolve().parents[1] / "envs" / "bimanual_handover_roller_lr.py"


class FakeBase:
    def is_left_gripper_open(self):
        return self.robot.is_left_gripper_open()

    def is_right_gripper_open(self):
        return self.robot.is_right_gripper_open()


def load_task_class():
    """Compile the real class without importing SAPIEN, robot code, or assets."""
    tree = ast.parse(TASK_FILE.read_text())
    task = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                and node.name == "bimanual_handover_roller_lr")
    module = ast.fix_missing_locations(ast.Module(body=[task], type_ignores=[]))
    namespace = {"Base_Task": FakeBase, "np": np,
                 "sapien": SimpleNamespace(Pose=object)}
    exec(compile(module, str(TASK_FILE), "exec"), namespace)
    return namespace[task.name]


Task = load_task_class()


class Entity:
    """Identity-based stand-in for a SAPIEN entity."""


class Link:
    def __init__(self, name):
        self.name = name
        self.entity = Entity()


class Joint:
    def __init__(self, link):
        self.child_link = link


class Articulation:
    def __init__(self, links):
        self.links = links

    def get_links(self):
        return self.links


class Robot:
    def __init__(self):
        self.links = {}
        for side in ("left", "right"):
            moving = Link("finger_moving")
            fixed = Link("finger_fixed")
            self.links[side] = (moving, fixed)
            setattr(self, f"{side}_gripper", [(Joint(moving), 1.0, 0.0)])
            setattr(self, f"{side}_entity", Articulation([moving, fixed]))
            setattr(self, f"{side}_fix_gripper_name", ["finger_fixed"])
        self.open = {"left": True, "right": True}

    def is_left_gripper_open(self):
        return self.open["left"]

    def is_right_gripper_open(self):
        return self.open["right"]


class Scene:
    def __init__(self):
        self.contacts = []

    def get_contacts(self):
        return self.contacts


def contact(actor, finger, *, reverse=False, points=True):
    """Match scene.get_contacts(): bodies contain entities; points signal contact."""
    bodies = [SimpleNamespace(entity=actor), SimpleNamespace(entity=finger.entity)]
    if reverse:
        bodies.reverse()
    return SimpleNamespace(bodies=bodies, points=[object()] if points else [])


class RollerHandoverSuccessTest(unittest.TestCase):
    def setUp(self):
        self.task = Task()
        self.task.robot = Robot()
        self.task.scene = Scene()
        self.task.roller = SimpleNamespace(actor=Entity())
        self.task.roller.get_pose = lambda: SimpleNamespace(p=self.position)
        self.task.middle_center = np.array([0.0, -0.08, 0.80])
        self.task.left_pad_center = np.array([-0.20, -0.08, 0.741])
        self.task.right_pad_center = np.array([0.20, -0.08, 0.741])
        self.position = self.task.left_pad_center.copy()
        for flag in ("eval_first_grasp", "eval_visited_middle", "eval_second_grasp",
                     "eval_handover_release", "eval_reached_final"):
            setattr(self.task, flag, False)

    def direction(self, first):
        second = "right" if first == "left" else "left"
        self.task.first_arm_tag = first
        self.task.second_arm_tag = second
        self.task.final_pad_center = getattr(self.task, f"{second}_pad_center")
        self.position = getattr(self.task, f"{first}_pad_center").copy()
        return second

    def step(self, position, *, closed=(), touching=(), duplicate=False):
        self.position = np.array(position, dtype=float)
        self.task.robot.open = {side: side not in closed for side in ("left", "right")}
        contacts = []
        for side, indices in touching:
            for index in indices:
                finger = self.task.robot.links[side][index]
                contacts.append(contact(self.task.roller.actor, finger,
                                        reverse=bool(index % 2)))
                if duplicate:
                    contacts.append(contact(self.task.roller.actor, finger, reverse=True))
        self.task.scene.contacts = contacts
        return self.task.check_success()

    def handover_until_release(self, first):
        second = self.direction(first)
        start = getattr(self.task, f"{first}_pad_center")
        middle = self.task.middle_center
        self.assertFalse(self.step(start, closed=(first,), touching=((first, (0, 1)),)))
        self.assertFalse(self.step(middle, closed=(first,), touching=((first, (0, 1)),)))
        self.assertFalse(self.step(middle, closed=(first, second),
                                   touching=((first, (0, 1)), (second, (0, 1)))))
        self.assertFalse(self.step(middle, closed=(second,), touching=((second, (0, 1)),)))
        return second

    def test_valid_handover_in_both_directions(self):
        for first in ("left", "right"):
            with self.subTest(first=first):
                self.setUp()
                second = self.handover_until_release(first)
                final = self.task.final_pad_center
                self.assertFalse(self.step(final, closed=(second,),
                                           touching=((second, (0, 1)),)))
                self.assertTrue(self.step(final))

    def test_missing_middle_second_hand_or_first_release(self):
        for first in ("left", "right"):
            with self.subTest(first=first):
                self.setUp()
                second = self.direction(first)
                start = getattr(self.task, f"{first}_pad_center")
                final = self.task.final_pad_center
                self.step(start, closed=(first,), touching=((first, (0, 1)),))
                self.step(final, closed=(first, second),
                          touching=((first, (0, 1)), (second, (0, 1))))
                self.assertFalse(self.step(final))
                self.assertFalse(self.task.eval_visited_middle)

                self.setUp()
                second = self.direction(first)
                self.step(self.task.middle_center, closed=(first,),
                          touching=((first, (0, 1)),))
                self.assertFalse(self.step(self.task.final_pad_center))
                self.assertFalse(self.task.eval_second_grasp)

                self.setUp()
                second = self.direction(first)
                middle = self.task.middle_center
                self.step(middle, closed=(first,), touching=((first, (0, 1)),))
                self.step(middle, closed=(first, second),
                          touching=((first, (0, 1)), (second, (0, 1))))
                self.step(self.task.final_pad_center, closed=(first, second),
                          touching=((first, (0, 1)), (second, (0, 1))))
                self.assertFalse(self.step(self.task.final_pad_center))
                self.assertFalse(self.task.eval_handover_release)

    def test_airborne_final_point_does_not_succeed(self):
        for first in ("left", "right"):
            with self.subTest(first=first):
                self.setUp()
                self.handover_until_release(first)
                floating = self.task.final_pad_center + np.array([0.0, 0.0, 0.12])
                self.assertFalse(self.step(floating))
                self.assertFalse(self.task.eval_reached_final)

    def test_same_named_fingers_are_distinguished_by_entity(self):
        self.direction("left")
        self.assertEqual(self.task.robot.links["left"][0].name,
                         self.task.robot.links["right"][0].name)
        self.assertFalse(self.step(self.task.middle_center, closed=("left", "right"),
                                   touching=(("left", (0,)), ("right", (1,)))))
        self.assertFalse(self.task.eval_first_grasp)
        self.assertFalse(self.task._roller_gripper_contacts("left"))
        self.assertFalse(self.task._roller_gripper_contacts("right"))

    def test_repeated_single_finger_contact_is_not_a_grasp(self):
        self.direction("left")
        self.assertFalse(self.step(self.task.middle_center, closed=("left",),
                                   touching=(("left", (0,)),), duplicate=True))
        self.assertFalse(self.task.eval_first_grasp)
        self.assertFalse(self.task._roller_gripper_contacts("left"))
        self.assertFalse(self.step(self.task.final_pad_center))

        # A fixed link may refer to the same entity as the moving finger.
        self.setUp()
        self.direction("left")
        moving, fixed = self.task.robot.links["left"]
        fixed.entity = moving.entity
        self.assertFalse(self.step(self.task.middle_center, closed=("left",),
                                   touching=(("left", (0,)),)))
        self.assertFalse(self.task.eval_first_grasp)


if __name__ == "__main__":
    unittest.main()
