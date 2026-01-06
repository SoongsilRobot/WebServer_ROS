from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_setup_assistant_launch

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "five_dof_cobot_rollJ4",
        package_name="five_dof_cobot_roll_j4_moveit_config",
    ).to_moveit_configs()
    return generate_setup_assistant_launch(moveit_config)
