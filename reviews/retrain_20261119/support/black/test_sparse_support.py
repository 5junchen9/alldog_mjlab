"""用真实编译碰撞体检查：底板、侧壁、旧contact槽均不能冒充有效顶面支持。"""

from types import SimpleNamespace

import mujoco
import pytest
import torch

from support.black.sparse_support import SparseSupportCatalog, support_plane


def fixture():
  feet = "".join(
    f'<geom name="{leg}_foot_collision" type="sphere" size=".02" pos="0 0 1"/>'
    for leg in ("FL", "FR", "RR", "RL")
  )
  model = mujoco.MjModel.from_xml_string(f"""<mujoco><option cone="elliptic"/><worldbody>
    {feet}
    <geom name="specialty_4_trench_10cm_near" type="box" pos="1 0 .1" size=".5 .5 .1"/>
    <geom name="specialty_4_trench_10cm_far" type="box" pos="2.1 0 .1" size=".5 .5 .1"/>
    <geom name="specialty_4_lower_floor" type="box" pos="1.5 0 -.1" size="3 2 .1"/>
    </worldbody></mujoco>""")
  catalog = SparseSupportCatalog(model, "cpu")
  data = SimpleNamespace(
    nacon=torch.tensor([1]),
    nefc=torch.tensor([1]),
    efc=SimpleNamespace(force=torch.tensor([[2.0]])),
    contact=SimpleNamespace(
      geom=torch.tensor([[0, 4]]),
      worldid=torch.tensor([0]),
      efc_address=torch.tensor([[0]]),
      pos=torch.tensor([[1.0, 0, 0.2]]),
      frame=torch.tensor([[[0.0, 0, 1], [1, 0, 0], [0, 1, 0]]]),
    ),
  )
  return catalog, data


def test_actual_top_support_and_reversed_geom_order():
  catalog, data = fixture()
  assert catalog.read(data, 1)["supported"].tolist() == [[True, False, False, False]]
  data.contact.geom[:] = torch.tensor([[4, 0]])
  assert catalog.read(data, 1)["supported"][0, 0]


def test_core_floor_failure_and_exit_floor_support():
  catalog, data = fixture()
  data.contact.geom[0, 1] = 6
  data.contact.pos[0] = torch.tensor([1.55, 0, 0])
  result = catalog.read(data, 1)
  assert result["lower_floor"][0, 0] and not result["supported"].any()
  data.contact.pos[0, 0] = 3.0
  result = catalog.read(data, 1)
  assert result["supported"][0, 0] and not result["lower_floor"].any()


def test_side_contact_zero_force_and_stale_slot_are_not_support():
  catalog, data = fixture()
  data.contact.frame[0] = torch.eye(3)
  assert not catalog.read(data, 1)["supported"].any()
  data.contact.frame[0, 0] = torch.tensor([0.0, 0, 1])
  data.efc.force.zero_()
  assert not catalog.read(data, 1)["supported"].any()
  data.efc.force.fill_(2)
  data.nacon.zero_()
  assert not catalog.read(data, 1)["supported"].any()


def test_nan_in_active_contact_fails_but_unused_slots_do_not():
  catalog, data = fixture()
  data.contact.pos[0, 0] = torch.nan
  assert catalog.read(data, 1)["invalid"][0]
  data.nacon.zero_()
  assert not catalog.read(data, 1)["invalid"].any()


def test_real_bridge_ramp_compiles_and_top_plane_rejects_bottom():
  from support.black.obstacle_geometry import subset_xml

  model = mujoco.MjModel.from_xml_string(subset_xml(["trench_10cm"]))
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  gid = model.geom("trench_10cm_entry").id
  bounds, coefficients = support_plane(model, data, gid)
  plane = torch.tensor(coefficients)
  top = torch.tensor([3.5, 9.0, 0.15], dtype=plane.dtype)
  bottom = torch.tensor([3.5, 9.0, 0.0], dtype=plane.dtype)
  assert abs(float(top @ plane[:3] + plane[3])) < 1e-6
  assert abs(float(bottom @ plane[:3] + plane[3])) > 0.1
  assert bounds[0] < 3.5 < bounds[1]


@pytest.mark.parametrize("task", ["broken_bridge", "plum_stakes"])
def test_generated_full_family_has_recognized_support_surfaces(task):
  from mjlab.terrains.terrain_generator import TerrainGenerator
  from support.black.parameters import PARAMETERS
  from support.black.specialty_env import make_specialty_configs

  cfg, _ = make_specialty_configs(
    task, num_envs=32, seed=119032, course_parameters=PARAMETERS
  )
  spec = mujoco.MjSpec()
  spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
  TerrainGenerator(cfg.scene.terrain.terrain_generator, "cpu").compile(spec)
  for leg in ("FL", "FR", "RR", "RL"):
    spec.worldbody.add_geom(
      name=f"{leg}_foot_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.02, 0, 0]
    )
  model = spec.compile()
  catalog = SparseSupportCatalog(model, "cpu")
  assert catalog.surface.any()
  assert catalog.floor.any()
