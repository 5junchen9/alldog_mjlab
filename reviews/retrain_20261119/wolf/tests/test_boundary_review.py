"""Wolf课程状态首入障及相机空视野的反例。"""

from copy import deepcopy

import pytest
import torch
from tensordict import TensorDict
from wolf.eval_protocol import SEEDS, panel
from wolf.eval_score import score_trial
from wolf.model import initial_models
from wolf.ppo import CourseNormalizer, build_ppo
from wolf.tests.test_contract import flat_rows


def test_course_state_survives_long_zero_history_and_ppo_save_restore():
  teacher, _, _ = initial_models(74)
  obs = TensorDict(
    {
      "history": torch.zeros(4, 318),
      "geometry": torch.zeros(4, 768),
      "critic": torch.zeros(4, 430),
      "course_state": torch.zeros(4, 3),
    },
    [4],
  )
  ppo = build_ppo(obs, teacher)
  normalizer = ppo.critic.obs_normalizer
  assert isinstance(normalizer, CourseNormalizer)
  zeros = torch.zeros(4096, 433)
  for _ in range(20):
    normalizer.update(zeros)
  first_entry = torch.zeros(4, 433)
  first_entry[:, -3:] = torch.tensor([0.9, 1 / 50, 1.0])
  normalized = normalizer(first_entry)
  torch.testing.assert_close(normalized[:, -3:], first_entry[:, -3:], rtol=0, atol=0)
  assert normalized.abs().max() <= 1
  obs["course_state"] = first_entry[:, -3:]
  with torch.no_grad():
    expected = ppo.critic(obs)
  saved = deepcopy(ppo.save())
  restored = build_ppo(obs, teacher)
  restored.load(saved, load_cfg=None, strict=True)
  with torch.no_grad():
    torch.testing.assert_close(restored.critic(obs), expected, rtol=0, atol=0)
  assert restored.critic.obs_normalizer.observation.count == 4096 * 20


@pytest.mark.parametrize("change", ["empty", "unhealthy", "old_episode", "stale"])
def test_wolf_requires_healthy_aligned_camera_but_allows_empty_view(change):
  spec = panel("flat_foundation", SEEDS[0])[0]
  rows = flat_rows(spec)
  if change == "empty":
    for row in rows:
      row["depth_has_hit"] = False
    expected = "pass"
  elif change == "unhealthy":
    rows[10]["depth_valid"] = False
    expected = "fail"
  elif change == "old_episode":
    rows[10]["capture_episode"] += 1
    expected = "invalid"
  else:
    rows[10]["capture_tick"] -= 5
    expected = "invalid"
  assert score_trial(rows, spec, None)["status"] == expected
