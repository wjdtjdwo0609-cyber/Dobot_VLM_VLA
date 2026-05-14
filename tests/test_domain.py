import unittest

from dobot_vla.domain.robot import DEFAULT_SAFETY_BOUNDS, DeltaAction, RobotPose, RobotState
from dobot_vla.domain.tasks import CommandCatalog


class RobotDomainTest(unittest.TestCase):
    def test_delta_action_is_clamped_to_workspace(self):
        pose = RobotPose(300, 140, 140, 80)
        action = DeltaAction(50, 50, 50, 50, 1.0)

        target = pose.apply_delta(action, DEFAULT_SAFETY_BOUNDS)

        self.assertEqual(target.to_list(), [310.0, 150.0, 150.0, 90.0])

    def test_robot_state_matches_pi0_input_shape(self):
        state = RobotState(RobotPose(200, 0, 50, 0), gripper_closed=True)

        self.assertEqual(state.to_model_input(), [200, 0, 50, 0, 1.0])


class TaskDomainTest(unittest.TestCase):
    def test_partial_korean_object_name_maps_to_training_prompt(self):
        command = CommandCatalog().command_for_object("휴지 좀")

        self.assertEqual(command, "pick up the tissue and hand it over")


if __name__ == "__main__":
    unittest.main()
