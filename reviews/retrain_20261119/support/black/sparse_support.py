"""桥桩支持证据：识别真实受力接触的碰撞体，空隙射线漏测不算故障。"""

import re

import mujoco
import numpy as np
import torch


def support_plane(model, data, gid):
  """读取实际水平箱顶或三角棱柱斜坡的朝上平面。"""
  matrix = data.geom_xmat[gid].reshape(3, 3)
  position = data.geom_xpos[gid]
  if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
    if not np.allclose(matrix, np.eye(3), atol=1e-6):
      raise ValueError("桥桩箱体须保留已定义的水平顶面")
    x, y, z = position
    sx, sy, sz = model.geom_size[gid]
    return [x - sx, x + sx, y - sy, y + sy, z + sz], [0, 0, 1, -(z + sz)]
  if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
    raise ValueError("支持目录仅接纳箱体或入口出口斜坡网格")
  mesh = model.geom_dataid[gid]
  start = model.mesh_vertadr[mesh]
  vertices = (
    model.mesh_vert[start : start + model.mesh_vertnum[mesh]] @ matrix.T + position
  )
  start = model.mesh_faceadr[mesh]
  faces = model.mesh_face[start : start + model.mesh_facenum[mesh]]
  planes = []
  for face in faces:
    a, b, c = vertices[face]
    normal = np.cross(b - a, c - a)
    length = np.linalg.norm(normal)
    if length > 1e-10 and normal[2] / length > 0.1:
      normal /= length
      planes.append(np.r_[normal, -np.dot(normal, a)])
  if not planes or not np.allclose(planes, planes[0], atol=1e-5):
    raise ValueError("入口出口网格须有唯一朝上的坡面")
  low, high = vertices.min(0), vertices.max(0)
  return [low[0], high[0], low[1], high[1], high[2]], planes[0]


class SparseSupportCatalog:
  """从编译后的模型读取箱体顶面；不把底部地板当成桥面或桩顶。"""

  def __init__(self, model, device, foot_geom_names=None):
    if foot_geom_names is None:
      foot_geom_names = tuple(
        f"{leg}_foot_collision" for leg in ("FL", "FR", "RR", "RL")
      )
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    foot = np.full(model.ngeom, -1, dtype=np.int64)
    surface = np.zeros(model.ngeom, dtype=bool)
    floor = np.zeros(model.ngeom, dtype=bool)
    bounds = np.zeros((model.ngeom, 5), dtype=np.float32)
    planes = np.zeros((model.ngeom, 4), dtype=np.float32)
    core = np.zeros((model.ngeom, 2), dtype=np.float32)
    patches = {}
    for gid in range(model.ngeom):
      name = model.geom(gid).name
      for slot, suffix in enumerate(foot_geom_names):
        if name == suffix or name.endswith("/" + suffix):
          foot[gid] = slot
      match = re.search(r"(specialty_\d+_)(.*)$", name)
      if match is None:
        continue
      prefix, local = match.groups()
      patch = patches.setdefault(prefix, {"surfaces": [], "core": [], "floor": []})
      is_floor = local == "lower_floor"
      is_support = bool(re.match(r"(?:trench_\d+cm|plum(?:_t\d+)?)_", local))
      if not (is_floor or is_support):
        continue
      if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH and not local.endswith(
        ("_entry", "_exit", "_entry_ramp", "_exit_ramp")
      ):
        raise ValueError(f"只允许明确命名的入口出口斜坡网格：{name}")
      bounds[gid], planes[gid] = support_plane(model, data, gid)
      if is_floor:
        floor[gid] = True
        patch["floor"].append(gid)
      else:
        surface[gid] = True
        patch["surfaces"].append(gid)
        if local.endswith(("_near", "_far")) or re.search(r"_r[12]_\d+$", local):
          patch["core"].append(gid)
    relevant = [patch for patch in patches.values() if patch["surfaces"]]
    if not relevant or set(foot[foot >= 0].tolist()) != {0, 1, 2, 3}:
      raise ValueError("缺少桥桩支持面或四足/四轮碰撞体")
    for patch in relevant:
      if not patch["core"] or not patch["floor"]:
        raise ValueError("桥桩核心区域或下层地板缺失")
      low = min(bounds[gid, 0] for gid in patch["core"])
      high = max(bounds[gid, 1] for gid in patch["core"])
      for gid in patch["floor"]:
        core[gid] = [low, high]
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
      raise ValueError("接触力读取要求椭圆摩擦锥")
    self.foot = torch.as_tensor(foot, device=device)
    self.surface = torch.as_tensor(surface, device=device)
    self.floor = torch.as_tensor(floor, device=device)
    self.bounds = torch.as_tensor(bounds, device=device)
    self.planes = torch.as_tensor(planes, device=device)
    self.core = torch.as_tensor(core, device=device)

  def read(self, data, count):
    """返回每足真实顶面支持、核心区落底，以及有效接触中的非有限值。"""
    geom = data.contact.geom.reshape(-1, 2)
    raw_world = data.contact.worldid.reshape(-1)
    device = raw_world.device
    world = raw_world.clamp(0, count - 1).long()
    geom_valid = ((geom >= 0) & (geom < len(self.foot))).all(-1)
    g0, g1 = geom.clamp(0, len(self.foot) - 1).long().unbind(-1)
    f0, f1 = self.foot[g0], self.foot[g1]
    slot = torch.where(f0 >= 0, f0, f1)
    other = torch.where(f0 >= 0, g1, g0)
    active = (
      (torch.arange(len(world), device=device) < data.nacon.reshape(-1)[0])
      & (raw_world >= 0)
      & (raw_world < count)
      & geom_valid
      & (slot >= 0)
      & ((f0 >= 0) ^ (f1 >= 0))
    )
    address = data.contact.efc_address.reshape(len(world), -1)[:, 0].long()
    force = data.efc.force.reshape(count, -1)
    normal = force[world, address.clamp(0, force.shape[1] - 1)]
    constrained = (address >= 0) & (address < data.nefc.reshape(-1)[world])
    position = data.contact.pos.reshape(-1, 3)
    frame = data.contact.frame.reshape(-1, 3, 3)
    finite = torch.isfinite(position).all(-1) & torch.isfinite(frame).flatten(1).all(-1)
    finite &= ~constrained | torch.isfinite(normal)
    top = self.bounds[other]
    plane = self.planes[other]
    top_contact = (
      active
      & constrained
      & finite
      & (normal > 1e-6)
      & ((frame[:, 0] * plane[:, :3]).sum(-1).abs() >= 0.95)
      & (((position * plane[:, :3]).sum(-1) + plane[:, 3]).abs() <= 0.005)
      & (position[:, 0] >= top[:, 0] - 1e-4)
      & (position[:, 0] <= top[:, 1] + 1e-4)
      & (position[:, 1] >= top[:, 2] - 1e-4)
      & (position[:, 1] <= top[:, 3] + 1e-4)
    )
    in_core = (position[:, 0] > self.core[other, 0]) & (
      position[:, 0] < self.core[other, 1]
    )
    lower = top_contact & self.floor[other] & in_core
    # 入口前和出口后的平地可支撑；核心区底板始终不能计作支持。
    supported = top_contact & (self.surface[other] | (self.floor[other] & ~in_core))
    flat = world * 4 + slot.clamp(0, 3)
    result = {}
    for name, mask in (("supported", supported), ("lower_floor", lower)):
      found = torch.zeros(count * 4, dtype=torch.long, device=device)
      found.scatter_reduce_(0, flat, mask.long(), reduce="amax", include_self=True)
      result[name] = found.reshape(count, 4).bool()
    invalid = torch.zeros(count, dtype=torch.long, device=device)
    invalid.scatter_reduce_(
      0, world, (active & ~finite).long(), reduce="amax", include_self=True
    )
    result["invalid"] = invalid.bool()
    return result
