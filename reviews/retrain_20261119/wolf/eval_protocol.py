"""Wolf固定独立面板；全部原始attempt重评分才能进入下一阶段。"""

from wolf.curriculum import SPECIALTIES

SEEDS = (1911701, 1911702, 1911703)
PROTOCOL = {
  "id": "wolf_shared_actor_absolute_v2",
  "seeds": list(SEEDS),
  "control_hz": 50,
  "settle_steps": 50,
  "stop_linear_max": 0.10,
  "stop_angular_max": 0.10,
  "tilt_max": 0.35,
  "minimum_support": 2,
  "tracking_fraction": 0.5,
  "max_episode_controls": 6000,
  "required_success_fraction": 1.0,
  "general_domains": [
    "slope_up",
    "slope_down",
    "stairs_up",
    "stairs_down",
    "rough",
    "wave",
  ],
  "scope": "simulation_candidate_thresholds_not_hardware_certification",
}


def panel(course, seed, max_level=None):
  rows = []
  if course in SPECIALTIES:
    from support.black.obstacle_geometry import GROUPS

    for level, name in enumerate(GROUPS[course]):
      if max_level is not None and level > max_level:
        continue
      for direction in (-1, 1):
        rows.append(
          {
            "case": name,
            "level": level,
            "direction": direction,
            "target": [0.2, 0.0, 0.0],
            "duration": 6000,
          }
        )
  elif course == "general":
    for domain in range(1, 7):
      for level in range(1, (4 if max_level is None else max_level) + 1):
        for direction in (-1, 1):
          rows.append(
            {
              "case": f"domain_{domain}_level_{level}_{direction}",
              "domain": domain,
              "level": level,
              "direction": 1,
              "target": [direction * 0.3, 0.0, 0.0],
              "duration": 1350,
            }
          )
  else:
    axes = (0,) if course == "flat_foundation" else (0, 1, 2)
    for axis in axes:
      for sign in (-1, 1):
        for orientation in (0.0, 3.141592653589793):
          target = [0.0, 0.0, 0.0]
          target[axis] = sign * (0.3 if axis == 2 else 0.2)
          rows.append(
            {
              "case": f"axis_{axis}_{sign}_{orientation}",
              "target": target,
              "orientation": orientation,
              "direction": 1,
              "duration": 550,
            }
          )
    rows.append(
      {
        "case": "standstill",
        "target": [0.0, 0.0, 0.0],
        "orientation": 0.0,
        "direction": 1,
        "duration": 550,
      }
    )
    if course == "flat_axes":
      # 平地资格覆盖训练的中速和前进+转向/横移复合命令。
      for index, target in enumerate(
        (
          [0.8, 0.0, 0.0],
          [-0.8, 0.0, 0.0],
          [0.0, 0.5, 0.0],
          [0.0, -0.5, 0.0],
          [0.0, 0.0, 0.8],
          [0.5, 0.0, 0.6],
          [0.5, 0.0, -0.6],
          [0.4, 0.25, 0.3],
          [0.4, -0.25, -0.3],
        )
      ):
        rows.append(
          dict(
            case=f"mixed_and_speed_{index}",
            target=target,
            orientation=0.0,
            direction=1,
            duration=550,
          )
        )
  return [
    {
      **row,
      "course": course,
      "seed": seed,
      "trial_id": f"{course}/{seed}/{row['case']}/{row['direction']}",
    }
    for row in rows
  ]


def requirements(stage):
  if stage.startswith("student:"):
    course = stage.split(":", 1)[1]
    return (*requirements(course), course)
  if stage == "flat_axes":
    return ("flat_foundation",)
  if stage == "general":
    return ("flat_foundation", "flat_axes")
  if stage in SPECIALTIES:
    return ("flat_foundation", "flat_axes", "general")
  return ()
