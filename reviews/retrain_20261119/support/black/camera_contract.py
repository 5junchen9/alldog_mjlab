import mujoco
import numpy as np


def verify_camera(model, name):
  d = mujoco.MjData(model)
  mujoco.mj_forward(model, d)
  c = model.camera(name).id
  t = model.body("robot/trunk").id
  R = d.xmat[t].reshape(3, 3)
  position = R.T @ (d.cam_xpos[c] - d.xpos[t])
  axis = R.T @ (-d.cam_xmat[c].reshape(3, 3)[:, 2])
  parent = model.body(model.cam_bodyid[c]).name
  assert parent == "robot/d435_link"
  assert np.allclose(position, [0.325, 0, -0.038], atol=1e-8)
  assert np.allclose(axis, [0.85445883, 0, -0.51951911], atol=1e-7)
  fx, fy = model.cam_intrinsic[c][:2]
  w, h = model.cam_sensorsize[c]
  fov = [
    float(np.degrees(2 * np.arctan(w / (2 * fx)))),
    float(np.degrees(2 * np.arctan(h / (2 * fy)))),
  ]
  assert np.allclose(fov, [87, 58], atol=1e-5)
  assert model.cam_resolution[c].tolist() == [424, 240]
  return {
    "parent": parent,
    "position_relative_trunk_m": position.tolist(),
    "optical_axis_relative_trunk": axis.tolist(),
    "fov_degrees": fov,
    "resolution": [424, 240],
    "convention": "MuJoCo camera local -Z; static compiled nominal data relative trunk",
    "distinct_from_Codex_aligned88_8": True,
  }
