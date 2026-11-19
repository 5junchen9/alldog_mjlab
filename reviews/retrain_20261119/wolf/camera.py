"""Black成像参数与Wolf仿真候选支架；外参尚未实物标定。"""

CAMERA = {
  "name": "shared_actor_depth",
  "body": "robot/base_link",
  # 躯干碰撞盒前缘x=.215、顶面z=.06，镜头放前缘之外、顶面上方。
  "pos": [0.245, 0.0, 0.105],
  "quat": [
    0.6163438796401546,
    0.3465836436275126,
    -0.3465836436275126,
    -0.6163438796401546,
  ],
  "sensor_size": [424, 240],
  "focal_pixel": [223.40138655956403, 216.48573063257084],
  "resolution": [424, 240],
}
CAMERA_STATUS = "simulation_candidate_mount_pending_mechanical_design_and_calibration"


def add_camera(spec):
  parameters = {key: value for key, value in CAMERA.items() if key != "body"}
  spec.body(CAMERA["body"]).add_camera(**parameters)
