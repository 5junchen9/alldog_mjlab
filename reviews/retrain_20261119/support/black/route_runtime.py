"""Actual route commands, footprint guards and contact phase evidence."""

import math

import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz
from support.black.safe_route import POLICY, SafeRoutes
from support.black.written_pose import written_root_pose


class RouteRuntime:
  def __init__(self, st):
    self.st = st
    self.core = SafeRoutes()
    n = st.env.num_envs
    dev = st.env.device
    self.enabled = torch.zeros(n, device=dev, dtype=torch.bool)
    self.angle = torch.zeros(n, device=dev)
    self.mid_y = torch.zeros(n, device=dev)
    self.guard_bits = torch.zeros(n, 4, device=dev, dtype=torch.bool)
    self.seen = torch.zeros(n, 3, device=dev, dtype=torch.bool)
    self.first = torch.full((n, 3), -1, device=dev, dtype=torch.long)
    self.contact_heading = torch.zeros(n, device=dev)
    self.contact_velocity = torch.zeros(n, device=dev)
    self.contact_speed = torch.zeros(n, device=dev)
    self.contact_xy = torch.zeros(n, 2, device=dev)
    self.phase_heading = torch.zeros(n, 3, device=dev)
    self.phase_velocity = torch.zeros(n, 3, device=dev)
    self.phase_speed = torch.zeros(n, 3, device=dev)
    self.phase = torch.zeros_like(st.stair_catalog.stage)
    for name, gid in st.stair_catalog.names.items():
      if "_stairs_" not in name and not name.startswith("stairs_"):
        continue
      if "_landing" in name:
        self.phase[:, gid] = 2
      elif "_up_" in name:
        self.phase[0, gid] = 1
        self.phase[1, gid] = 3
      elif "_down_" in name:
        self.phase[0, gid] = 3
        self.phase[1, gid] = 1

  def sync(self, ids):
    chosen = ids.cpu().tolist()
    self.enabled[ids] = False
    self.angle[ids] = 0
    self.mid_y[ids] = 0
    self.guard_bits[ids] = False
    self.seen[ids] = False
    self.first[ids] = -1
    selected = [i for i in chosen if i in self.core.routes]
    if selected:
      index = torch.tensor(selected, device=self.st.env.device, dtype=torch.long)
      self.enabled[index] = True
      self.angle[index] = torch.tensor(
        [math.radians(self.core.routes[i]["path_angle_degrees"]) for i in selected],
        device=self.st.env.device,
      )
      self.mid_y[index] = torch.tensor(
        [self.core.routes[i]["mid_y"] for i in selected], device=self.st.env.device
      )

  def spawn(self, ids):
    ids = ids[self.enabled[ids]]
    if not len(ids):
      return
    robot = self.st.env.scene["robot"]
    index = ids.cpu().tolist()
    pose = written_root_pose(robot, ids)
    origin = self.st.env.scene.env_origins[ids]
    values = torch.tensor(
      [
        [self.core.routes[i][k] for k in ("birth_x", "birth_y", "birth_yaw_radians")]
        for i in index
      ],
      device=self.st.env.device,
    )
    pose[:, :2] = origin[:, :2] + values[:, :2]
    zero = torch.zeros_like(values[:, 2])
    pose[:, 3:7] = quat_from_euler_xyz(zero, zero, values[:, 2])
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)

  def violation(self, positions, foot_positions, length, width, direction, heading):
    origin = self.st.env.scene.env_origins
    local = positions - origin
    line = self.mid_y + torch.tan(self.angle) * (local[:, 0] - length / 2)
    goal = torch.where(direction > 0, 0.0, math.pi) + self.angle
    error = torch.atan2(torch.sin(goal - heading), torch.cos(goal - heading))
    feet_y = (foot_positions[:, :, 1] - origin[:, None, 1]).abs() + 0.02
    # Only augmented routes get these additional conservative guards.
    bits = torch.stack(
      (
        (feet_y > (width - 0.02)[:, None]).any(-1),
        (local[:, 1] - line).abs() > POLICY["maximum_cross_track_m"],
        error.abs() > math.radians(POLICY["maximum_heading_error_degrees"]),
        ~torch.isfinite(local).all(-1) | ~torch.isfinite(error),
      ),
      -1,
    )
    self.guard_bits = bits & self.enabled[:, None]
    return self.guard_bits.any(-1)

  def observe(self):
    st = self.st
    env = st.env
    robot = env.scene["robot"]
    data = env.sim.data
    geom = data.contact.geom.reshape(-1, 2)
    wr = data.contact.worldid.reshape(-1)
    world = wr.clamp(0, env.num_envs - 1).long()
    g0 = geom[:, 0].clamp(0, len(st.stair_catalog.foot) - 1).long()
    g1 = geom[:, 1].clamp(0, len(st.stair_catalog.foot) - 1).long()
    f0 = st.stair_catalog.foot[g0]
    f1 = st.stair_catalog.foot[g1]
    tread = torch.where(f0 >= 0, g1, g0)
    phase = self.phase[(st.command.route_direction[world] < 0).long(), tread]
    address = data.contact.efc_address.reshape(len(world), -1)[:, 0].long()
    forces = data.efc.force.reshape(env.num_envs, -1)
    normal = forces[world, address.clamp(0, forces.shape[1] - 1)]
    valid = (
      (torch.arange(len(world), device=env.device) < data.nacon.reshape(-1)[0])
      & (wr >= 0)
      & (wr < env.num_envs)
    )
    valid &= (
      ((f0 >= 0) | (f1 >= 0)) & (phase > 0) & self.enabled[world] & ~st.failed[world]
    )
    bounds = st.stair_catalog.boxes[tread]
    pos = data.contact.pos.reshape(-1, 3)
    frame = data.contact.frame.reshape(-1, 3, 3)
    valid &= torch.isfinite(pos).all(-1) & torch.isfinite(frame).flatten(1).all(-1)
    valid &= frame[:, 0, 2].abs() >= 0.95
    valid &= (pos[:, 2] - bounds[:, 4]).abs() <= 0.005
    valid &= (pos[:, 0] >= bounds[:, 0] - 1e-4) & (pos[:, 0] <= bounds[:, 1] + 1e-4)
    valid &= (pos[:, 1] >= bounds[:, 2] - 1e-4) & (pos[:, 1] <= bounds[:, 3] + 1e-4)
    valid &= (
      (address >= 0)
      & (address < data.nefc.reshape(-1)[world])
      & torch.isfinite(normal)
      & (normal > 1e-6)
    )
    found = torch.zeros(env.num_envs * 3, device=env.device, dtype=torch.long)
    found.scatter_reduce_(
      0,
      world * 3 + (phase - 1).clamp(0, 2),
      valid.long(),
      reduce="amax",
      include_self=True,
    )
    touched = found.reshape(env.num_envs, 3) > 0
    first_contact = touched.any(-1) & ~self.seen.any(-1)
    self.contact_heading[first_contact] = robot.data.heading_w[first_contact]
    velocity = robot.data.root_link_lin_vel_w
    self.contact_velocity[first_contact] = torch.atan2(
      velocity[first_contact, 1], velocity[first_contact, 0]
    )
    self.contact_speed[first_contact] = velocity[first_contact, :2].norm(dim=-1)
    self.contact_xy[first_contact] = (
      robot.data.root_link_pos_w[first_contact, :2]
      - env.scene.env_origins[first_contact, :2]
    )
    fresh = touched & ~self.seen
    for phase_index in range(3):
      mask = fresh[:, phase_index]
      self.phase_heading[mask, phase_index] = robot.data.heading_w[mask]
      self.phase_velocity[mask, phase_index] = torch.atan2(
        velocity[mask, 1], velocity[mask, 0]
      )
      self.phase_speed[mask, phase_index] = velocity[mask, :2].norm(dim=-1)
    self.first = torch.where(fresh, env.common_step_counter, self.first)
    self.seen |= touched

  def flush(self, ids=None):
    if ids is None:
      ids = torch.nonzero(self.enabled).flatten()
    ids = ids[self.enabled[ids]]
    if not len(ids):
      return
    data = (
      torch.cat(
        (
          ids[:, None],
          self.seen[ids].long(),
          self.first[ids],
          self.contact_heading[ids, None],
          self.contact_velocity[ids, None],
          self.contact_speed[ids, None],
          self.contact_xy[ids],
          self.phase_heading[ids],
          self.phase_velocity[ids],
          self.phase_speed[ids],
        ),
        -1,
      )
      .cpu()
      .tolist()
    )
    for row in data:
      r = self.core.routes[int(row[0])]
      r["phase_seen"] = [bool(x) for x in row[1:4]]
      r["phase_first"] = [None if x < 0 else int(x) for x in row[4:7]]
      r["phase_contacts"] = {
        name: (
          None
          if not r["phase_seen"][j]
          else dict(
            step=r["phase_first"][j],
            body_yaw_radians=row[12 + j],
            speed_mps=row[18 + j],
            velocity_heading_radians=None if row[18 + j] < 0.01 else row[15 + j],
          )
        )
        for j, name in enumerate(("ascent", "landing", "descent"))
      }
      r["first_contact"] = (
        None
        if not any(r["phase_seen"])
        else dict(
          body_yaw_radians=row[7],
          velocity_heading_radians=None if row[9] < 0.01 else row[8],
          speed_mps=row[9],
          root_xy_relative_origin_m=row[10:12],
          angle_relative_step_normal_radians=math.atan2(
            math.sin(row[7] - (0 if r["direction"] > 0 else math.pi)),
            math.cos(row[7] - (0 if r["direction"] > 0 else math.pi)),
          ),
        )
      )

  def state_dict(self):
    self.flush()
    return self.core.state_dict()

  def load_state_dict(self, saved):
    self.core = SafeRoutes.from_state(saved)
    ids = torch.arange(self.st.env.num_envs, device=self.st.env.device)
    self.sync(ids)
    for i, r in self.core.routes.items():
      self.seen[i] = torch.tensor(r["phase_seen"], device=self.st.env.device)
      self.first[i] = torch.tensor(
        [-1 if x is None else x for x in r["phase_first"]], device=self.st.env.device
      )
      if r["first_contact"]:
        self.contact_heading[i] = r["first_contact"]["body_yaw_radians"]
        self.contact_velocity[i] = r["first_contact"]["velocity_heading_radians"] or 0.0
        self.contact_speed[i] = r["first_contact"]["speed_mps"]
        self.contact_xy[i] = torch.tensor(
          r["first_contact"]["root_xy_relative_origin_m"], device=self.st.env.device
        )
      for j, name in enumerate(("ascent", "landing", "descent")):
        phase = r.get("phase_contacts", {}).get(name)
        if phase:
          self.phase_heading[i, j] = phase["body_yaw_radians"]
          self.phase_velocity[i, j] = phase["velocity_heading_radians"] or 0.0
          self.phase_speed[i, j] = phase["speed_mps"]
