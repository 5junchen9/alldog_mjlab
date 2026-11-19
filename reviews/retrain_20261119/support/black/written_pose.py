"""Read exactly the raw free-joint pose just written by the reset event."""


def written_root_pose(robot, ids):
  return robot.data.data.qpos[
    ids[:, None], robot.data.indexing.free_joint_q_adr
  ].clone()
