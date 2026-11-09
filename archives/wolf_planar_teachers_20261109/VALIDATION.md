# Final targeted validation, 2026-11-09

Frozen launcher SHA-256:
`4154e5dc8394a9c58ce321cf9893c58050c7384cd5c39de2f3863a60ec66f439`.

## Results

- Final focused tests: **26 passed**, 3.49 seconds, one upstream
  `torch.jit.script` deprecation warning.
- Ruff format/check: passed for the two new Python files only.
- ty: all checks passed for the new launcher and tests.
- Pyright with the installed modern Node: **0 errors, 0 warnings, 0 informations**.
- `bash -n run_jc.sh`: passed.
- Actual JC v2 runtime plan runs: both straight and turn exited 0. Native JSON
  records source manifest `11e2df1a3bf5784c81e99149459ccbffed4ea695ae39b05525908e89405e45c7`
  and **608 verified files**. The complete source manifest still matches after the
  final runs. No source files in v2, v1, or the original mjlab tree were edited.
- No training, simulated rollout or optimizer update was performed by these
  checks. The test for `--train` patches the dispatch function, not the budget.
- A temporary missing test import and then a mutable-attribute narrowing error
  were corrected before the final full focused rerun. Only the final frozen code
  has the results above. These are targeted checks, not an assertion that the
  complete upstream test suite was rerun.

## Coverage

Full config comparison after normalizing only the explicitly permitted profile
changes verifies unchanged physics, rewards, randomization, observations, actions,
HIM/network/algorithm defaults (other than requested minibatch count), and source
registry isolation. Tests cover both profile domains, native CPU samples of 32768
commands per profile, positive/negative/stop mass, deadband boundaries, repeat
heading changes, all forbidden sampling flags, no curriculum object, exact-file
resume selection, CLI fields, budget rejection, plan-only behavior, native cfg
train dispatch, source manifest and complete JSON domain fields.

The deterministic exact-zero RNG test first demonstrates native forward leakage
(vx=0.3), then verifies the independent post-sampling projection removes it for
turn and ensures straight vy=yaw=0. No change to frozen native source is needed.

## Historical local check commands (private paths generalized)

These are validation records, not a standalone setup script. `${PLANAR_ROOT}`,
`${V2_ROOT}`, `${DEV_VENV}`, `${RUNTIME_VENV}` and `${NODE_BIN}` represent the original
prepared local environment. Tests, the local `pyproject.toml` and wrappers are not
published. No package installation or training is performed by this archive.

```sh
cd ${PLANAR_ROOT}
export UV_PROJECT_ENVIRONMENT=${DEV_VENV}
export PYTHONPATH=${V2_ROOT}/src:${PLANAR_ROOT}
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
UV=/snap/bin/uv
V2=${V2_ROOT}
$UV run --project "$V2" --offline --no-sync ruff format --config pyproject.toml planar_teachers.py tests
$UV run --project "$V2" --offline --no-sync ruff check --config pyproject.toml planar_teachers.py tests
$UV run --project "$V2" --offline --no-sync pytest -q -p no:cacheprovider tests
$UV run --project "$V2" --offline --no-sync ty check --project "$PWD" planar_teachers.py tests
$UV run --project "$V2" --offline --no-sync ${NODE_BIN} ${DEV_VENV}/lib/python3.13/site-packages/pyright/dist/index.js --project pyproject.toml
bash run_jc.sh --profile turn > turn_plan.json
bash run_jc.sh --profile straight > straight_plan.json
bash -n run_jc.sh
```

The runtime previews use the validated
`${RUNTIME_VENV}/bin/python`, always through `uv run`.
No packages were installed or synchronized. No default training was started.
`turn_plan.json` and `straight_plan.json` are configuration evidence, not checkpoint
sidecars by themselves. Associate them with the actual resulting checkpoint's hash
and source/runtime/run identity before claiming a checkpoint domain.

## Final admitted first-round values

4096 environments, rollout100, epochs5, minibatches8, 2000 additional iterations,
save every50, TensorBoard, upload_model=False, allocator expandable_segments:True.
Straight seed301; turn seed401. Both checkpoint paths are null. Each requests
819200000 transitions, at409600 transitions/iteration and51200/minibatch.

Both profiles preserve 53 values/frame, 6frames/318 values,16 actions, plane terrain,
no camera and native physical/reward/HIM contracts. They remain specialized
command-domain training configurations, not general or deployment-certified teachers.

No source of old checkpoint files is automatically selected or deleted. Exact-file
resume is opt-in, records its SHA-256, and does not establish semantic equivalence
across profiles. The original registered task is the rollback path.
