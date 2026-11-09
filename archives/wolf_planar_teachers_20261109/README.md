> Archive note: this is a configuration-preparation snapshot based on the fixed [Wolf v2 source archive](../wolf_methods_20261109_v2/README.md). The JSON files are pre-launch previews. Actual runs and resulting checkpoint hashes require separate records. Local tests, wrappers and machine-specific type-check configuration are intentionally excluded; their hashes are recorded in SOURCE_MANIFEST.json.

# Wolf planar HIM specialists, 2026-11-09

This separate launcher configures the frozen v2 `wolf-flat-him` task without editing
or adding registry entries. It is **not a trained or certified teacher** and does not
cover lateral motion, simultaneous vx+yaw, rough terrain or camera perception.

## Command domains

- `straight`: body vx sampled uniformly from [-1, 1] m/s; vy=yaw=0.
- `turn`: body vx=vy=0; body-z angular velocity uniform [-1.5, 1.5] rad/s.
- Both retain 10-second resampling, native Wolf command sampling and an explicit
  10% standing mask. Heading, world-frame and forward-only sampling are disabled.
- Native XY deadband stays 0.2 m/s: sampled |vx| <= 0.2 is cleared. Thus straight
  has approximately **28% zero commands** (10% explicit standing plus 18% deadband),
  with 36% forward and 36% reverse. It does **not** train nonzero speeds below 0.2.
  Turn has approximately 10% zero commands, 45% negative yaw and 45% positive yaw.
- The original `command` forward-speed curriculum is removed, not merely paused.
  No curriculum terms remain; checkpoint curriculum restore is `none`, preventing
  even an old checkpoint from silently expanding the turn profile's vx range.
- Physics, rewards, randomization, actions, observations and HIM are inherited
  unchanged: 53 values/frame, 6 frames / 318 history values, 16 actions.

## Preview and run

Always invoke through the appropriate **v2** runtime launcher, which sets
`PYTHONPATH` to its own frozen source and uses `uv run` with the verified runtime.
No training starts without `--train`. A default preview only prints the complete
JSON command/policy/training domain; it does not create logs or load checkpoints.

```sh
V2=/path/to/frozen-v2
PLANAR=/path/to/planar-launcher
export PYTORCH_ALLOC_CONF=expandable_segments:True
bash "$V2/launchers/run_jc.sh" "$PLANAR/planar_teachers.py" --profile straight
bash "$V2/launchers/run_jc.sh" "$PLANAR/planar_teachers.py" --profile turn --minibatches 8
# Only after resource admission, add --train to the desired command.
```

On Windows/WSL, use the transferred v2 `launchers/run_windows.sh` and the transferred
absolute path of this script. The script does not change `sys.path`. It verifies the
pinned v2 manifest SHA-256 and every manifest file before training or printing a plan.

Parameters: `--profile straight|turn`, `--num-envs` (1..4096, default 4096),
`--rollout 100`, `--epochs 5`, `--minibatches 4|8` (default 8), `--seed` (uint32, defaults 301/401),
`--iterations` (>=2000, default 2000), `--save-interval 50`, `--log-root PATH`,
`--checkpoint /exact/file.pt`, `--train`.

Default 4096 produces 409600 transitions/iteration. Four minibatches give 102400
transitions/minibatch; eight give 51200. These are configuration bounds, not evidence
that a GPU can fit or sustain either choice. No automatic allocation reduction,
installation, deletion, cleanup or process launching is performed in preview mode.

Experiment directories are `wolf_planar_straight_him_20261109` and
`wolf_planar_turn_him_20261109`. Run names are `wolf_planar_straight_him` and
`wolf_planar_turn_him`. TensorBoard logging is selected, with `upload_model=False`.
Native `launch_training(task_id="wolf-flat-him", args=cfg)` receives the modified
configuration, preserving the registered native HIM runner and original task IDs.

An exact checkpoint uses the native `checkpoint_path` full-HIM-state load, with
`resume=False` (no regex lookup), `warm_start=False`, and curriculum restore `none`.
A resumed `--iterations N` requests N **additional learning iterations** according to
the native runner. The launcher checks file existence, not checkpoint provenance;
verify the checkpoint is trusted and from the intended same profile before use.
No checkpoint is deserialized during preview; its SHA-256 is recorded when supplied. Save the printed JSON together with
the actual checkpoint path and SHA-256 as its domain sidecar; preview alone is not a
checkpoint validation or certification. Native params/env.yaml and agent.yaml are
written by the training runtime when an actual run begins.

Rollback: stop using this launcher and invoke the original v2 registered task, e.g.
`bash "$V2/launchers/run_jc.sh" -m mjlab.scripts.train wolf-flat-him ...`. Original
source, tasks and default configurations remain unchanged.

## Checks

Tests exercise config isolation, exact profile fields, actual native command
sampling on CPU (not simulated dynamics), both command signs/stop coverage,
deadband boundaries, frozen curriculum, resume selection, CLI/budget rejection and
plan-only versus explicit train dispatch. Training dispatch is mocked in tests.
No environment rollout, optimizer update, GPU training or convergence test is run.
See VALIDATION.md for exact commands and results.

## Resource admission update

The coordination task reports a separate JC 4096x100 / 5 epochs / 8 minibatches full
backward diagnostic passed with `PYTORCH_ALLOC_CONF=expandable_segments:True`.
Consequently this launcher defaults to 8 minibatches, 2000 iterations, straight seed 301
and turn seed 401. Both fresh runs use **no checkpoint**. No old checkpoints are
removed. The new local `run_jc.sh` wrapper sets that process-only allocator default
and delegates to v2's unchanged launcher. This launcher's tests do not independently
repeat the resource diagnostic. The domain JSON records the allocator environment.
Explicit checkpoint loading across profiles does not guarantee matching semantics.

## Pure-domain endpoint guard

The native sampler tests zero-probability branch masks with `random <= 0`. A finite
RNG can return exactly zero, activating the forward-only branch at a configured
probability of zero. The independent `PlanarVelocityCommandCfg` subclasses the
native Wolf configuration. Its command samples natively, then forces vy=0 and
straight yaw=0 / turn vx=0 and clears world/heading/forward flags. This is a command
domain correction only; it does not modify v2 or the physical/control/reward/HIM
contract. A deterministic forced-zero RNG test demonstrates the native vx=0.3
leak and confirms projection closes it. Sampling expectations above remain
approximate: native finite-RNG endpoint events can add a negligible stop/sign bias.

## Complete snapshot dependency

This directory is an overlay, not a standalone source distribution. The runnable
base is the complete validated v2 snapshot, including its 608 manifest entries,
assets and existing runtime dependencies. The launcher verifies the pinned
`snapshot_manifest.json` SHA-256 and checks every entry against files beside it.
A public source-only archive that omits tests, STL assets or that manifest cannot
run this overlay by cloning alone. Retain or transfer the full validated snapshot;
no downloads or installations are performed by this launcher. Local wrappers and
machine-specific type-check configuration are not part of a portable source archive.
