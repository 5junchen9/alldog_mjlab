"""Bounded monotone ascent credit from actual tread contacts; no success override."""

import re

import mujoco
import numpy as np
import torch

STEP_MILLI = 25
CAP_MILLI = 150
CONFIG = dict(
  per_legal_ascent_stage=0.025,
  per_episode_cap=0.15,
  require_fore_and_hind_touch=True,
  max_root_displacement_per_control=0.25,
  tread_contact_z_tolerance=0.005,
  minimum_normal_force=1e-6,
  minimum_vertical_normal=0.95,
)


class StairCatalog:
  """Exact compiled box top faces; reverse ascent uses the opposite stair side."""

  def __init__(self, model, device):
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    stage = np.zeros((2, model.ngeom), dtype=np.int64)
    edge = np.zeros_like(stage, dtype=np.float32)
    boxes = np.zeros((model.ngeom, 5), dtype=np.float32)
    foot = np.full(model.ngeom, -1, dtype=np.int64)
    groups = {}
    names = {}
    for gid in range(model.ngeom):
      name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
      names[name] = gid
      for slot, leg in enumerate(("FL", "FR", "RR", "RL")):
        if name == leg + "_foot_collision" or name.endswith(
          "/" + leg + "_foot_collision"
        ):
          if slot in foot:
            raise ValueError("Ambiguous foot collision identity")
          foot[gid] = slot
      match = re.fullmatch(r"(.*?(stairs_\d+cm))_(up|down)_(\d+)", name)
      if match is None:
        if "stairs_" in name and name.endswith("_landing"):
          if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Landing phase evidence requires an actual box top")
          x, y, z = data.geom_xpos[gid]
          sx, sy, sz = model.geom_size[gid]
          boxes[gid] = [x - sx, x + sx, y - sy, y + sy, z + sz]
        continue
      if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("Ascent shaping only admits original box stair treads")
      if not np.allclose(data.geom_xmat[gid].reshape(3, 3), np.eye(3), atol=1e-6):
        raise ValueError("Rotated tread needs a separately verified catalog")
      x, y, z = data.geom_xpos[gid]
      sx, sy, sz = model.geom_size[gid]
      boxes[gid] = [x - sx, x + sx, y - sy, y + sy, z + sz]
      groups.setdefault((match[1], match[2], match[3]), []).append(gid)
    counts = {}
    surfaces = []
    for (prefix, case, side), gids in groups.items():
      reverse = side == "down"
      gids.sort(key=lambda gid: data.geom_xpos[gid, 0], reverse=reverse)
      heights = [float(boxes[gid, 4]) for gid in gids]
      if any(b <= a + 1e-6 for a, b in zip(heights, heights[1:], strict=False)):
        raise ValueError("Tread ascent heights are not strictly ordered")
      if case in counts and counts[case] != len(gids):
        raise ValueError("Direction/patch stage count mismatch")
      counts[case] = len(gids)
      for index, gid in enumerate(gids, 1):
        stage[int(reverse), gid] = index
        edge[int(reverse), gid] = boxes[gid, int(reverse)]
        surfaces.append(
          dict(
            geom_id=gid,
            prefix=prefix,
            case=case,
            direction=-1 if reverse else 1,
            stage=index,
            bounds_top=boxes[gid].tolist(),
          )
        )
    if not groups or set(foot[foot >= 0].tolist()) != {0, 1, 2, 3}:
      raise ValueError("Compiled stair/foot identities incomplete")
    if int(model.opt.cone) != int(mujoco.mjtCone.mjCONE_ELLIPTIC):
      raise ValueError("Normal-force reader requires unchanged elliptic cone")
    self.stage = torch.as_tensor(stage, device=device)
    self.edge = torch.as_tensor(edge, device=device)
    self.boxes = torch.as_tensor(boxes, device=device)
    self.foot = torch.as_tensor(foot, device=device)
    self.counts = counts
    self.names = names
    self.metadata = dict(
      config=CONFIG,
      stages_by_case=counts,
      compiled_treads=surfaces,
      no_geometry_modified=True,
    )

  def contact_bits(self, data, direction, expected_stage, root_x, active):
    """Fixed-capacity vector operations; no per-world Python or CPU transfer."""
    geom = data.contact.geom.reshape(-1, 2)
    world_raw = data.contact.worldid.reshape(-1)
    n = len(direction)
    world = world_raw.clamp(0, n - 1).long()
    g0 = geom[:, 0].clamp(0, len(self.foot) - 1).long()
    g1 = geom[:, 1].clamp(0, len(self.foot) - 1).long()
    foot0 = self.foot[g0]
    foot1 = self.foot[g1]
    primary = torch.where(foot0 >= 0, foot0, foot1)
    tread = torch.where(foot0 >= 0, g1, g0)
    index = (direction[world] < 0).long()
    stage = self.stage[index, tread]
    pos = data.contact.pos.reshape(-1, 3)
    frame = data.contact.frame.reshape(-1, 3, 3)
    address = data.contact.efc_address.reshape(len(world), -1)[:, 0].long()
    forces = data.efc.force.reshape(n, -1)
    normal = forces[world, address.clamp(0, forces.shape[1] - 1)]
    bounds = self.boxes[tread]
    actual_count = data.nacon.reshape(-1)[0]
    valid = (
      (torch.arange(len(world), device=world.device) < actual_count)
      & (world_raw >= 0)
      & (world_raw < n)
      & (primary >= 0)
      & (stage == expected_stage[world])
      & (stage > 0)
      & (address >= 0)
      & (address < data.nefc.reshape(-1)[world])
      & torch.isfinite(normal)
      & (normal > CONFIG["minimum_normal_force"])
      & torch.isfinite(pos).all(-1)
      & (frame[:, 0, 2].abs() >= CONFIG["minimum_vertical_normal"])
      & ((pos[:, 2] - bounds[:, 4]).abs() <= CONFIG["tread_contact_z_tolerance"])
      & (pos[:, 0] >= bounds[:, 0] - 1e-4)
      & (pos[:, 0] <= bounds[:, 1] + 1e-4)
      & (pos[:, 1] >= bounds[:, 2] - 1e-4)
      & (pos[:, 1] <= bounds[:, 3] + 1e-4)
      & (direction[world] * (root_x[world] - self.edge[index, tread]) >= 0)
      & active[world]
    )
    found = torch.zeros(n * 4, device=world.device, dtype=torch.long)
    flat = world * 4 + primary.clamp(0, 3)
    found.scatter_reduce_(0, flat, valid.long(), reduce="amax", include_self=True)
    weights = torch.tensor([1, 2, 4, 8], device=world.device)
    return (found.reshape(n, 4) * weights).sum(-1)


class StairLedger:
  fields = (
    "max_stage",
    "contact_bits",
    "paid_milli",
    "owed_milli",
    "blocked",
    "initialized",
    "previous_root",
    "birth_step",
    "direction",
    "level",
    "furthest_progress",
  )

  def __init__(self, n, device):
    self.max_stage = torch.zeros(n, device=device, dtype=torch.long)
    self.contact_bits = torch.zeros_like(self.max_stage)
    self.paid_milli = torch.zeros_like(self.max_stage)
    self.owed_milli = torch.zeros_like(self.max_stage)
    self.blocked = torch.zeros(n, device=device, dtype=torch.bool)
    self.initialized = torch.zeros_like(self.blocked)
    self.previous_root = torch.zeros(n, 3, device=device)
    self.birth_step = torch.zeros_like(self.max_stage)
    self.direction = torch.zeros_like(self.max_stage)
    self.level = torch.zeros_like(self.max_stage)
    self.furthest_progress = torch.zeros(n, device=device)
    self.last_step = -1
    self.paid_step = -1
    self.dt = 0.02
    self.migration = None

  def prepare(
    self,
    step,
    dt,
    *,
    progress,
    root,
    hits,
    stages,
    entry,
    active,
    failed,
    valid,
    birth_step,
    direction,
    level,
  ):
    if step == self.last_step:
      return
    if step < self.last_step or not 0 < dt < 1:
      raise ValueError("Invalid shaping clock")
    self.last_step = step
    self.dt = dt
    fresh = active & ~self.initialized
    finite = torch.isfinite(root).all(-1) & torch.isfinite(progress)
    changed = self.initialized & (
      (self.birth_step != birth_step)
      | (self.direction != direction)
      | (self.level != level)
    )
    jumped = self.initialized & (
      (root - self.previous_root).norm(dim=-1)
      > CONFIG["max_root_displacement_per_control"]
    )
    invalid = active & (
      failed | ~valid | ~finite | changed | jumped | (fresh & (progress >= 0.75))
    )
    self.blocked |= invalid
    self.owed_milli[invalid] = 0
    live = active & ~self.blocked
    self.birth_step = torch.where(fresh, birth_step, self.birth_step)
    self.direction = torch.where(fresh, direction, self.direction)
    self.level = torch.where(fresh, level, self.level)
    forward = (
      live
      & ~fresh
      & ((progress - self.furthest_progress) > 1e-5)
      & (direction * (root[:, 0] - self.previous_root[:, 0]) > 1e-5)
    )
    self.furthest_progress = torch.where(
      live & finite,
      torch.maximum(self.furthest_progress, progress),
      self.furthest_progress,
    )
    self.previous_root = torch.where((live & finite)[:, None], root, self.previous_root)
    self.initialized |= fresh
    self.contact_bits |= torch.where(forward & entry, hits, 0)
    fore = (self.contact_bits & 3) != 0
    hind = (self.contact_bits & 12) != 0
    legal = forward & entry & fore & hind & (self.max_stage < stages)
    self.max_stage += legal.long()
    available = (CAP_MILLI - self.paid_milli - self.owed_milli).clamp(min=0)
    self.owed_milli += legal.long() * available.clamp(max=STEP_MILLI)
    self.contact_bits = torch.where(legal, 0, self.contact_bits)

  def consume(self, step):
    if step != self.last_step:
      raise RuntimeError("Shared terminal snapshot must precede shaping reward")
    if step == self.paid_step:
      return torch.zeros_like(self.previous_root[:, 0])
    value = self.owed_milli.to(self.previous_root.dtype) / (1000 * self.dt)
    self.paid_milli += self.owed_milli
    self.owed_milli.zero_()
    self.paid_step = step
    return value

  def reset(self, ids):
    for field in self.fields:
      getattr(self, field)[ids] = 0

  def state_dict(self):
    return dict(
      version=1,
      config=CONFIG,
      step_milli=STEP_MILLI,
      cap_milli=CAP_MILLI,
      last_step=self.last_step,
      paid_step=self.paid_step,
      dt=self.dt,
      migration=self.migration,
      tensors={k: getattr(self, k).detach().cpu().clone() for k in self.fields},
    )

  def load_state_dict(self, saved):
    if saved is None:
      self.blocked.fill_(True)
      self.migration = "Prior checkpoint has no stage ledger; retire old episodes until fresh physical reset"
      return
    if (
      saved["version"] != 1
      or saved["config"] != CONFIG
      or saved["step_milli"] != STEP_MILLI
      or saved["cap_milli"] != CAP_MILLI
    ):
      raise ValueError("Shaping schema/config mismatch")
    for name in self.fields:
      value = saved["tensors"][name]
      target = getattr(self, name)
      if (
        value.shape != target.shape
        or value.dtype != target.dtype
        or not torch.isfinite(value).all()
      ):
        raise ValueError("Invalid shaping tensor " + name)
      target.copy_(value.to(target.device))
    self.last_step = saved["last_step"]
    self.paid_step = saved["paid_step"]
    self.dt = saved["dt"]
    self.migration = saved.get("migration")
    if (
      (self.max_stage < 0).any()
      or (self.paid_milli < 0).any()
      or (self.owed_milli < 0).any()
      or ((self.paid_milli + self.owed_milli) > CAP_MILLI).any()
      or (self.contact_bits < 0).any()
      or (self.contact_bits > 15).any()
    ):
      raise ValueError("Corrupt shaping budget or stage state")
