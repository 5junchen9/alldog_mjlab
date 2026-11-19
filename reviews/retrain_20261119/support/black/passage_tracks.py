"""True compiled box contact certificate, independent from the original reward ledgers."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import torch

from support.black.passage_course import HEIGHTS


class PhysicalTrack:
  def __init__(self, model, base, device):
    stage = np.zeros((2, model.ngeom), dtype=np.int64)
    lev = np.full(model.ngeom, -1, dtype=np.int64)
    edges = np.zeros((2, model.ngeom), dtype=np.float32)
    boxes = base.boxes.detach().cpu().numpy()
    spatial = mujoco.MjData(model)
    mujoco.mj_kinematics(model, spatial)
    feet = np.flatnonzero(base.foot.detach().cpu().numpy() >= 0)
    if len(feet) != 4:
      raise ValueError("Expected original four compiled feet")
    tracks = {}
    excluded = set()
    for name, g in base.names.items():
      m = re.fullmatch(r"(.*?)stairs_(\d+)cm_(up_\d+|down_\d+|landing)", name)
      if not m:
        continue
      prefix, cm, part = m.groups()
      cm = int(cm)
      if cm not in HEIGHTS:
        # Actual source can compile geometry outside this course plan.
        excluded.add(cm)
        continue
      height = HEIGHTS.index(cm)
      if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX) or not np.allclose(
        spatial.geom_xmat[g].reshape(3, 3), np.eye(3), atol=1e-6, rtol=0
      ):
        raise ValueError("Expected original axis-aligned physical tread or landing")
      if not all(
        (int(model.geom_contype[g]) & int(model.geom_conaffinity[j]))
        or (int(model.geom_conaffinity[g]) & int(model.geom_contype[j]))
        for j in feet
      ):
        raise ValueError("Missing native terrain-foot collision mask")
      tracks.setdefault((prefix, height), {})[part] = g
      lev[g] = height
    assert tracks
    self.compiled_cm = tuple(sorted({HEIGHTS[h] for _, h in tracks} | excluded))
    self.outside_route_cm = tuple(sorted(excluded))
    for (_, _height), gp in tracks.items():
      assert len(gp) == 13 and "landing" in gp
      ids = [g for k, g in gp.items() if k != "landing"]
      assert len(ids) == 12 and len(set(ids)) == 12
      for k in range(2):
        order = sorted(ids, key=lambda j: boxes[j, 0], reverse=bool(k))
        zz = [float(boxes[g, 4]) for g in order]
        assert all(zz[j + 1] > zz[j] for j in range(5))
        assert all(zz[j + 1] < zz[j] for j in range(6, 11))
        assert abs(zz[5] - zz[6]) < 1e-5
        for j, g in enumerate(order, 1):
          stage[k, g] = j
          edges[k, g] = boxes[g, k]
        stage[k, gp["landing"]] = 13
    self.stage = torch.as_tensor(stage, device=device)
    self.level = torch.as_tensor(lev, device=device)
    self.edges = torch.as_tensor(edges, device=device)
    self.boxes = base.boxes
    self.foot = base.foot
    assert int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_ELLIPTIC)

  def contacts(self, data, direction, level, want, root_x, active):
    n = len(direction)
    gid = data.contact.geom.reshape(-1, 2)
    raw = data.contact.worldid.reshape(-1)
    world = raw.clamp(0, n - 1).long()
    trans = gid.clamp(0, len(self.foot) - 1).long()
    f0, f1 = self.foot[trans].unbind(-1)
    foot = torch.where(f0 >= 0, f0, f1)
    tread = torch.where(f0 >= 0, trans[:, 1], trans[:, 0])
    side = (direction[world] < 0).long()
    step = self.stage[side, tread]
    xyz = data.contact.pos.reshape(-1, 3)
    frame = data.contact.frame.reshape(-1, 3, 3)
    addr = data.contact.efc_address.reshape(len(world), -1)[:, 0].long()
    forces = data.efc.force.reshape(n, -1)
    force = forces[world, addr.clamp(0, forces.shape[1] - 1)]
    b = self.boxes[tread]
    ok = (
      (torch.arange(len(world), device=world.device) < data.nacon.reshape(-1)[0])
      & (raw >= 0)
      & (raw < n)
      & (foot >= 0)
      & (gid >= 0).all(-1)
      & (gid < len(self.foot)).all(-1)
      & (self.level[tread] == level[world])
      & active[world]
      & (addr >= 0)
      & (addr < data.nefc.reshape(-1)[world])
      & torch.isfinite(force)
      & (force > 1e-6)
      & torch.isfinite(xyz).all(-1)
      & (frame[:, 0, 2].abs() >= 0.95)
      & ((xyz[:, 2] - b[:, 4]).abs() <= 0.005)
      & (xyz[:, 0] >= b[:, 0] - 1e-4)
      & (xyz[:, 0] <= b[:, 1] + 1e-4)
      & (xyz[:, 1] >= b[:, 2] - 1e-4)
      & (xyz[:, 1] <= b[:, 3] + 1e-4)
    )
    directional = direction[world] * (root_x[world] - self.edges[side, tread]) >= 0
    seen = torch.zeros(4 * n, device=world.device, dtype=torch.long)
    seen.scatter_reduce_(
      0,
      4 * world + foot.clamp(0, 3),
      (ok & directional & (step == want[world]) & (step <= 12)).long(),
      reduce="amax",
    )
    mask = (seen.reshape(n, 4) * torch.tensor([1, 2, 4, 8], device=world.device)).sum(
      -1
    )
    landing = torch.zeros(n, device=world.device, dtype=torch.long)
    landing.scatter_reduce_(0, world, (ok & (step == 13)).long(), reduce="amax")
    return mask, landing.bool()


class ObservedPassage:
  def __init__(self, n, device):
    self.stage = torch.zeros(n, device=device, dtype=torch.long)
    self.stage_contact_bits = torch.zeros_like(self.stage)
    self.landing = torch.zeros(n, device=device, dtype=torch.bool)
    self.pass_once = torch.zeros_like(self.landing)
    self.first_verified_pass_tick = torch.full_like(self.stage, -1)
    self.invalid = torch.zeros_like(self.landing)
    self.assigned = torch.zeros_like(self.landing)
    self.birth = torch.full_like(self.stage, -1)
    self.level = torch.full_like(self.stage, -1)
    self.direction = torch.zeros_like(self.stage)
    self.previous = torch.zeros(n, 3, device=device)
    self.seen = torch.zeros_like(self.landing)
    self.step = -1

  def on_assign(self, ids, birth, level, direction):
    for name in (
      "stage",
      "stage_contact_bits",
      "landing",
      "pass_once",
      "invalid",
      "seen",
    ):
      getattr(self, name)[ids] = 0
    self.first_verified_pass_tick[ids] = -1
    self.assigned[ids] = True
    self.birth[ids] = birth
    self.level[ids] = level
    self.direction[ids] = direction

  def observe(
    self,
    step,
    *,
    root,
    birth,
    level,
    direction,
    active,
    bad,
    entry,
    native_ascent,
    exit_ok,
    hit,
    landing,
  ):
    if step == self.step:
      return
    assert step > self.step
    self.step = step
    live = active & self.assigned
    finite = torch.isfinite(root).all(-1)
    displacement = root - self.previous
    first = live & ~self.seen
    changed = (
      (birth != self.birth) | (level != self.level) | (direction != self.direction)
    )
    self.invalid |= live & (
      bad | ~finite | changed | (~first & (displacement.norm(dim=-1) > 0.25))
    )
    live &= ~self.invalid
    moving = (direction * displacement[:, 0]) > 1e-5
    previous_stage = self.stage.clone()
    self.landing |= live & entry & (previous_stage == 6) & landing
    can_move_stage = (previous_stage < 6) | ((previous_stage >= 6) & self.landing)
    step_ok = live & ~first & moving & entry & can_move_stage & (previous_stage < 12)
    self.stage_contact_bits |= torch.where(step_ok, hit, 0)
    # Ordered stage needs a front and a rear foot, possibly reported on adjacent frames.
    front = (self.stage_contact_bits & 3) != 0
    rear = (self.stage_contact_bits & 12) != 0
    legal = step_ok & front & rear
    self.stage += legal.long()
    self.stage_contact_bits = torch.where(legal, 0, self.stage_contact_bits)
    just = (
      live
      & ~first
      & (self.stage == 12)
      & self.landing
      & (native_ascent == 6)
      & exit_ok
      & ~self.pass_once
    )
    self.first_verified_pass_tick[just] = step
    self.pass_once |= just
    self.previous = torch.where((live & finite)[:, None], root, self.previous)
    self.seen |= live

  def expire(self, lane):
    self.assigned[lane] = False
