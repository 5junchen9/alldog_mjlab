"""Black geometry and control adapter for the unchanged mjlab Go1 baseline.

Model and control values: N-W-wolf/super-dog at
1b9f15de46eb75f6f253582b66fd51536548b4d9. Original files stay byte-exact.
The D435 mount/visual/inertia are preserved; no optical calibration is invented.
"""

import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_go1.go1_constants import FULL_COLLISION
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

ROOT = Path(__file__).parent
SOURCE_URDF = ROOT / "source" / "black_description.urdf"
BLACK_ACTION_SCALE = 0.25
BLACK_DEFAULT_JOINT_POS = {
  "FL_hip_joint": 0.0,
  "FL_thigh_joint": 0.8014,
  "FL_calf_joint": -1.527,
  "FR_hip_joint": 0.0,
  "FR_thigh_joint": -0.8014,
  "FR_calf_joint": 1.527,
  "RR_hip_joint": 0.0,
  "RR_thigh_joint": -0.8014,
  "RR_calf_joint": 1.527,
  "RL_hip_joint": 0.0,
  "RL_thigh_joint": 0.8014,
  "RL_calf_joint": -1.527,
}
BLACK_JOINT_NAMES = tuple(BLACK_DEFAULT_JOINT_POS)
# Match Black body traversal for contacts, height scans and reward site IDs.
FOOT_NAMES = ("FL", "FR", "RR", "RL")


def get_spec() -> mujoco.MjSpec:
  """Import all source geometry/inertias while retaining fixed link frames."""
  root = ET.fromstring(SOURCE_URDF.read_bytes())
  extension = ET.SubElement(root, "mujoco")
  ET.SubElement(
    extension,
    "compiler",
    discardvisual="false",
    fusestatic="false",
    strippath="false",
  )
  for mesh in root.findall(".//mesh"):
    filename = mesh.attrib["filename"]
    path = ROOT / "source" / filename
    if filename == "meshes/trunk.STL":
      # MuJoCo STL decoder limits a file to 200000 faces. This OBJ retains
      # every source vertex/triangle, without simplification or mesh repair.
      path = ROOT / "derived" / "trunk.obj"
    mesh.set("filename", str(path))
  for link in root.findall("link"):
    name = link.attrib["name"]
    for i, visual in enumerate(link.findall("visual")):
      visual.set("name", f"{name}_visual{i}")
    for i, collision in enumerate(link.findall("collision")):
      suffix = "" if name.endswith("_foot") else str(i)
      collision.set("name", f"{name}_collision{suffix}")
  spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
  spec.body("base").add_freejoint(name="root_free")
  for geom in spec.geoms:
    if geom.contype or geom.conaffinity:
      geom.group = 3
  for foot in FOOT_NAMES:
    spec.body(f"{foot}_foot").add_site(name=foot, pos=(0.0, 0.0, 0.0), size=(0.005,))
  spec.body("d435_link").add_site(name="d435_mount", pos=(0.0, 0.0, 0.0), size=(0.005,))
  # Stock sensor semantics at Black's source-defined IMU frame. Go1 uses
  # its own offset site; do not copy Go1's hardware mount into Black.
  # Black imu_joint is fixed at trunk origin with identity rotation.
  spec.body("imu_link").add_site(name="imu", pos=(0.0, 0.0, 0.0), group=5)
  for name, sensor_type in (
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
  ):
    spec.add_sensor(
      name=name,
      type=sensor_type,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname="imu",
    )
  spec.add_sensor(
    name="imu_upvector",
    type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="world",
    reftype=mujoco.mjtObj.mjOBJ_SITE,
    refname="imu",
  )
  # Stock rewards refer to this sensor (zero weight by default).
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="base",
  )
  return spec


def get_black_robot_cfg() -> EntityCfg:
  """Use Black hardware values with native mjlab actuator/contact semantics."""
  return EntityCfg(
    spec_fn=get_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.45),
      joint_pos=BLACK_DEFAULT_JOINT_POS.copy(),
      joint_vel={".*": 0.0},
    ),
    collisions=(deepcopy(FULL_COLLISION),),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        BuiltinPositionActuatorCfg(
          target_names_expr=BLACK_JOINT_NAMES,
          stiffness=40.0,
          damping=1.2,
          effort_limit=20.0,
        ),
      ),
      soft_joint_pos_limit_factor=0.9,
    ),
  )
