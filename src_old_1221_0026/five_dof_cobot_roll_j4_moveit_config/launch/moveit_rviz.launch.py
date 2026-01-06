from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder(
        "five_dof_cobot_rollJ4",
        package_name="five_dof_cobot_roll_j4_moveit_config",
    ).to_moveit_configs()
    return generate_moveit_rviz_launch(moveit_config)
