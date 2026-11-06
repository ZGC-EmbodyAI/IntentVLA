# Roller handover success checking

The original release updated handover progress only in expert helpers called
by `play_once()`. Resetting the environment before policy evaluation cleared
these flags, leaving success unreachable through `take_action()` alone
([issue #3](https://github.com/ZGC-EmbodyAI/IntentVLA/issues/3)).

The checker now observes the roller pose and actual gripper contacts in order:

1. The first hand holds the roller.
2. That hand carries it into the raised middle region.
3. Both hands hold it in that region.
4. The first hand opens while the second hand still holds it.
5. The roller reaches the opposite pad with both hands open.

A hold requires contact with at least two distinct finger links from one arm
and a gripper that is not open. Entity identity distinguishes fingers even
when two robot instances share link names. Expert moves use the same observer
instead of assigning successful phases directly.

The middle region is within 3 cm in x and 7 cm in y of `middle_center`, above
`middle_center.z - 3 cm`. Final placement retains the original 7 cm x/y
tolerance and additionally requires height within 4 cm of the pad origin.
The expert lowers toward the pad before releasing the roller, avoiding the
old high release that could bounce it beyond the target.

These checker and expert-placement changes should be identified when comparing
new evaluations or collected data with the original release.

## Regression tests

From this repository, with Python and NumPy installed:

```bash
python tests/test_roller_handover_success.py
```

The tests execute the actual checker with simulated poses and contact objects;
they require neither SAPIEN imports nor assets. They cover both directions,
missing middle/second-hand/release phases, an airborne final pose, equal finger
names across arms, and repeated contacts with a single finger.

Real-simulator development checks used an integrated XPolicyLab-era RoboTwin
checkout with SAPIEN and the aloha-agilex embodiment: seed 0 expert motion
(left to right) passed; seed 1 controlled rollout (right to left) reached
`eval_success=True` in 10 `take_action()` calls without invoking `play_once()`.
The controlled rollout generated motions with expert helpers and bypassed the
expert progress observer, so only `take_action()` / `check_success()` updated
the flags. This is a checker integration check, not a trained-policy score.
The simulator smoke used the existing batch runner's direct process-exit
handling after assertions passed to avoid a SAPIEN/CUDA teardown error.

The legacy RoboTwin runtime has not been revalidated. This patch preserves the
public task's `Base_Task` integration; it does not provide the full XPolicyLab
migration or the trace hooks already referenced by `play_once()` in the release.
