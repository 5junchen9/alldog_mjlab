# Wolf terrain teaching and dual-value adaptation

This change is opt-in. The original `wolf-flat`, `wolf-rough`, `wolf-flat-him`, and
`wolf-rough-him` task factories are not changed by this module. No training run,
learning curve, speedup, or locomotion success claim is reported here.

## Training integration

`methods_cfg.wolf_methods_env_cfg` calls `configure_terrain_teacher`, adding the
actual target geometry, sensors, reward terms, and hard-stage failure condition.
`MethodsPPO(DualRewardPPOMixin, HIMPPO)` collects the reward channel during the
existing PPO/HIM rollout, computes dual returns, and uses the native HIM minibatch
update. This is training-path integration, not a standalone simulation demo.

The placement contribution is included exactly once in total environment reward.
`WheelPlacementReward` publishes the same weighted, time-scaled contribution in
`extras['wolf_sparse_reward']` before automatic resets. The adapter separates
`[total - placement, placement]`, applies time-limit bootstrap to each value,
computes each GAE, normalizes the two advantages independently, then sums them
with explicit weights (default 1, 1). The actor receives one combined advantage;
two independent deterministic value networks each regress their own return.
The existing vectorized value-loss mean is the mean of the two value losses.
It is neither min-of-two-Q learning nor two critics trained on the same total.

Checkpoint metadata records channel names and advantage weights. Incompatible
full resumes fail before loading. Single-value and dual-value checkpoints are not
interchangeable; use an explicitly implemented conversion, never silent loading.
Single-device, feedforward training is supported. RND, recurrent models, and
per-minibatch advantage normalization are rejected in dual mode.

## Terrain stages

The soft and hard stages generate matched stones/beams with the same random seed
and target elevations. Raycast group 0 always represents the target scene.

- Soft: target collision is disabled; a group-3 flat collision floor supports the
  robot. The target gap is still visible and used by the placement penalty.
- Hard: target collision is enabled and the hidden flat floor is absent. A floor
  0.4 m below the target is only a catch floor. A base drop below 0.18 m relative to
  terrain origin terminates the episode, so the low floor is not a valid route.
- The stage transition requires rebuilding the environment and loading a matching
  model checkpoint. These modules do not change collision flags during rollout or
  independently schedule the soft-to-hard transition.
- Each stage additionally has ordinary terrain difficulty rows. Row curriculum
  and soft-to-hard physical staging are distinct mechanisms.
- New task commands should be restricted to the forward corridor during this
  first adaptation. This is a separate task contract, not evidence that the
  original omnidirectional skill is retained.

Each wheel has 15 downward ray samples across an 8 cm by 4 cm world-aligned patch.
Loaded-wheel unsupported fractions generate a negative placement reward with
weight 2.0. Soft support is read from target elevations; hard support additionally
must be close to the nominal wheel-bottom height (5 cm tolerance). Wolf's wheels
are not humanoid soles. This is a project-specific approximate wheel footprint,
not BeamDojo's exact humanoid foot geometry or touchdown distribution. The patch
is world-aligned and nominal wheel radius is 8 cm; radius randomization is not an
exact per-step contact-shape reconstruction.

Hard-stage base-height shaping references loaded contact support height rather
than the mean of rays into deep gaps. Soft-stage base-height shaping references
its actual flat floor. This prevents the old rough mean-height reward from
encouraging a lower base merely because nearby rays fall into holes.

## Reduced-model reference / CLF scope

`LipmClfReward` integrates a constant-height LIPM exactly over one control step.
It chooses a velocity-tracking acceleration, limits it using a conservative
friction coefficient, projects the corresponding ZMP onto the convex hull of
loaded contacts, and explicitly marks an infeasible acceleration invalid.
Per-environment reference state is maintained and cleared independently at reset.
A quadratic tracking error and finite-difference decrease residual produce the
CLF-like shaping cost (weight -0.1).

The valid domain is deliberately narrow: at least three loaded contacts, support
height spread at most 5 cm, base COM proxy height 0.2–0.8 m, conservative friction
0.3, and reference position drift at most 0.5 m. Outside it, shaping is disabled
and the local reference is reinitialized. The usable fraction is logged.

This is **not** PLANC's hybrid MLIP planner, full-body dynamic feasibility,
footstep optimization, a proven CLF, a stability certificate, or an action safety
filter. It uses the root body's COM as a proxy, ignores whole-body angular
momentum, torque/kinematic limits, changing support height, and flight/stepping
transitions. It must be labeled PLANC-inspired reduced-model shaping. Formal
whole-body feasible references remain a separate implementation and validation
requirement.

## Ablation switches and observability

`configure_terrain_teacher` exposes `soft_terrain`, `foot_placement`, and `clf`.
`wolf_methods_runner_cfg` exposes `dual_critic`. If placement is removed, also
select a single critic; dual mode fails loudly when its reward channel is absent.
Useful logs are `Methods/wheel_support_coverage`, `Methods/loaded_wheel_fraction`,
`Methods/lipm_reference_feasible`, `Methods/clf_violation`, and ordinary named
reward/termination statistics. None is a substitute for held-out task success.

## Verification actually performed

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest -q tests/test_wolf_terrain_teacher.py`:
  9 passing tests, no optimizer steps. Covers channel separation, timeout handling,
  GAE normalization, checkpoint mismatch rejection, independent parameters,
  coverage/missing rays, support-hull projection, LIPM equation, partial reset,
  baseline isolation, geometry matching, and physical ghost-vs-solid semantics.
- A CPU MuJoCo falling-sphere test sees the same 0.4 m target gap in both stages,
  but rests at z=0.05 on the hidden soft floor and z=-0.35 on the hard catch floor.
- Actual Wolf CUDA smoke: soft and hard each used 2 environments, one terrain row,
  5 zero-action steps. Both constructed, reset, stepped, and closed successfully;
  rewards and placement channels were finite, with 60 support rays per robot.
  The sampled hard run logged LIPM-feasible fraction 0.5. This is API validation,
  not performance evaluation or evidence that a policy can traverse the terrain.
- `uv run ty check` and `uv run ruff check` passed for the three new source modules.
- Direct `uv run pyright` could not start analysis: the system Node runtime rejects
  modern JavaScript syntax (`SyntaxError: Unexpected token '?'`). No environment
  dependencies were installed or changed to hide that limitation.
- Standard pytest auto-loading first encountered the pre-existing ROS Python 3.10
  plugin / Python 3.13 mismatch (`lark` missing); disabling unrelated plugin
  auto-loading let the explicit project tests run. This is not a product-code fix.

The full PPO/methods construction and mock-update test is owned by the combined
methods integration suite. No 2000-iteration formal run has been launched here.

## Source concepts

- BeamDojo: https://arxiv.org/html/2502.10363v3
- PLANC: https://arxiv.org/html/2601.06286v1

This implementation was written for the existing mjlab/RSL-RL interfaces rather
than copied from third-party training repositories. Paper-level equivalence is
not asserted.
