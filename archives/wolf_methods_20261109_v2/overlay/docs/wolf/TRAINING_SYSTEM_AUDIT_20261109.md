# Wolf training-system audit and HIM compatibility repair

Audit date: 2026-11-09. Host: local Linux validation host. Repository: `${MJLAB_ROOT}`, HEAD `45b1e263e`. This report describes the original four Wolf tasks at the start of this audit; parallel experimental modules are not certified by this report.

## Scope and preservation

Read-only system audit followed by the specifically requested minimal HIM compatibility repair. No training, rollout collection, backward pass, or optimizer step was performed. Existing policies, logs, physics, rewards, control parameters, and default task profiles were not overwritten. Wolf source and extensive Black work were already untracked/uncommitted; no reset, clean, checkout, commit, push, dependency installation, or driver change was done.

Before-file copies, SHA256, patch and logs are under `${WOLF_EVIDENCE_ROOT}/him_ppo_audit_20261109T142357Z/`. Only `src/mjlab/rl/wolf_him/ppo.py`, the new `tests/test_wolf_him_ppo.py`, the focused changelog entry, and this audit document were changed by this worker. Other workers may add separate experimental files concurrently.

## Actual architecture

- `config/wolf/__init__.py`: `wolf-flat` and `wolf-rough` use PPO; `wolf-flat-him` and `wolf-rough-him` use HIMPPO.
- `config/wolf/env_cfg.py`: actor53 = command3 + IMU angular velocity3 + projected gravity3 + leg position12 + leg velocity12 + signed wheel velocity4 + previous raw action16. Critic56 adds privileged base linear velocity3, without actor observation corruption.
- `rl/wolf_him/spec.py`, `policy.py`, and `config/wolf/mdp/him.py`: six 53-dimensional frames externally, newest-first canonical representation. The source encoder removes each frame's three command elements, hence 300 inputs. It predicts scaled velocity3 and normalized latent16. Current actor frame53 plus those19 gives actor input72. The target encoder takes successor frame without command plus true velocity, 53 dimensions.
- This is already a HIM estimator with velocity MSE and prototype/Sinkhorn swapped-assignment contrastive loss, not a VAE. There is no VAE decoder, posterior sampling, or KL-to-prior objective. Target encoder and 32 prototypes are trained online, not a frozen certified teacher.
- `rl/wolf_him/ppo.py` keeps policy/critic optimizer separate from estimator optimizer. Policy gradients do not enter the estimator (`policy.py:get_latent` uses no-grad). Terminal successor targets are captured pre-reset by `mdp/him.py:HimTerminalTargets`, avoiding normal next-episode target contamination.
- Original Wolf code has no Huber BC, DAgger label/replay loop, two reward-channel critics, RSSM/world-model objective, planner/CLF objective, or GLAD map attention. These need separate explicit integrations. Black CTS code is not a drop-in Wolf teacher.
- Rough terrain has a ray grid (1.6 m x 1.0 m, 0.1 m resolution), but original actor and critic do not receive the map. It is used for the rough base-height reward. A scan sensor existing does not mean a visual/terrain-aware teacher exists.

## Confirmed HIM defect and repair

RSL-RL 5.5.1 updates actor/critic observation normalizers at the end of `PPO.update`, from flattened rollout observations. It no longer updates them in `process_env_step` or the runner. The custom HIMPPO override omitted this block, while Wolf critic normalization is enabled. Therefore its critic normalization statistics stayed at initialization or stale loaded values.

The repair calls both normalizer-update interfaces once after minibatch processing and before rollout clearing, matching the installed base PPO timing. HIM actor normalization remains its intentional no-op. No reward or network shape changes were made.

The override also did not implement base PPO multi-GPU gradient/LR synchronization, mixed-precision execution, minibatch advantage normalization, or recurrent minibatch handling, and silently discarded `share_cnn_encoders`. These options now fail explicitly before initialization/config mutation where applicable. Existing rejection of RND and symmetry data augmentation is retained. Default single-device, feedforward, global-advantage-normalized HIM with mirror loss is unchanged apart from the required critic-statistics update. An RSSM integration must own its sequence/state storage rather than enable unsupported recurrent flags on this original runner.

## Verification

- New tests against original code: 8 failed and 2 passed, demonstrating the normalizer omission and unsupported-option acceptance.
- Final focused suite: **55 passed**, original 45 Wolf integration/source/speed-guard cases plus 10 new HIM cases. Includes available CUDA cases. No learning occurs in the new tests; normalizer epilogue is exercised with zero optimization batches, and backward/Adam.step are explicitly forbidden.
- Ruff format/check passed for changed Python files. New tests pass `ty`.
- Original HIMPPO has **17 existing ty diagnostics** (16 errors and one warning), including TensorDict optional typing, policy/base nominal type mismatch, and upstream optimizer-factory annotation. Before/after checks show the same 17 diagnostic sites/categories. A baseline copy only reanchors relative imports for standalone checking. This is not a clean type-check claim. Full repository make check/test and pyright were not run; there is no commit.
- Isolated Wolf runtime check: PyTorch2.14.1+cu126, RSL-RL5.5.1, TensorDict0.14.2. Constructed the actual WolfHimOnPolicyRunner with default models and diagnostic 3-step synthetic storage. Normalizer count changed 0 to6 after the update epilogue; actor and critic MLP weight hashes stayed unchanged. `learn`, estimator update, backward and Adam.step were disabled. This does not validate a real PPO/estimator learning update, simulator rollout, convergence, or resume with learned optimizer state.

## Control and physics contract

Evidence: `docs/wolf/DESIGN_ALIGNMENT.md`, `source_contract.json`, `asset_zoo/robots/wolf/wolf_constants.py`, `config/wolf/env_cfg.py:_actions`.

Policy action order is FL, FR, RL, RR, each hip/thigh/calf/foot, 16 total. Leg commands are position targets, hip scale0.125 and thigh/calf0.25 around default pose; wheel commands are velocity targets, scale5 rad/s with FL/RL positive and FR/RR negative forward signs. PD is legs80/2, wheels0/0.6. Physics dt0.005s and decimation4 give 50Hz control. Nominal mass33.6686kg; force caps legs60 and wheels17 N m. These are implemented values, not new proposed settings.

The optional speed guard is a separate experiment, not a hard physical velocity constraint. Do not silently enable it for the original task or mislabel old checkpoints. Current MuJoCo/mujoco-warp3.11 differs from upstream3.14 and has documented cylinder MULTICCD limitations. No stack upgrade was attempted.

## Rewards, command sampling, and evaluation gaps

Original active rewards are x/y/yaw exponential tracking, vertical and angular-motion penalties, orientation, base height, leg/wheel action rate, hip pose, standing posture, position limits, and torques. Rewards are scaled by control dt. Several landing/slip kernels exist but are not active reward terms. There is no explicit sparse foothold/landing-coverage or planner CLF reward channel in the original configuration.

`mdp/rewards.py:44-83` uses robot-body root-link linear x/y and angular z components; yaw tracking is not a world-vertical heading-rate metric on slopes. Default commands have vx/vy in [-1,1] m/s, yaw in [-3.14,3.14] rad/s, sampled every10s. Standing, forward-only, world-frame and heading fractions are zero. Wolf clears small XY norms <=0.2 but not yaw. Thus almost every sampled command already has nonzero yaw. A proposed 75% nonzero-turn schedule would not increase this coverage. Pure forward, pure turn, stop, and amplitude-stratified tests need deliberate coverage.

The current forward-speed curriculum divides whole-episode mean tracking return by episode duration, but assigns the sample to a low/high speed group using only final command vx (`mdp/curriculums.py:205-223`). A20s episode can include different10s commands, so that grouping can misattribute tracking. Short early-termination episodes lack an independent survival gate in this curriculum. This training heuristic must not serve as teacher qualification. A new explicit curriculum should accumulate per command segment and separately test survival, tracking, action/force limits, and recovery.

## Checkpoint and teacher readiness

Original runner saves actor, critic, PPO optimizer, iteration, common_step_counter, command-curriculum state; HIM saves its optimizer too. Installed RSL-RL restores PPO learning rate from optimizer state. `command_curriculum_restore` defaults to `none`; `full` means command-curriculum statistics, not a full physics snapshot. Ordinary load does not restore warm-start provenance back into `_warm_start_info`.

No qpos/qvel/solver snapshot, episode age, current command and timer, observation/history/delay buffers, domain-randomization/event state, rough terrain placement, RNG state, or partial rollout state is persisted. Training startup randomizes episode ages. This is optimizer/policy continuation, not trajectory-identical resume. Saves use direct torch.save rather than an atomic temporary-file replacement and have no independently verified best/latest promotion policy.

Loader checks PPO/HIM mismatch and some robot metadata, but not a complete asset/control/observation/normalization/provenance contract. Missing robot metadata is tolerated. It uses weights_only=False, so arbitrary externally obtained checkpoints should not be treated as safe serialized data. Do not deserialize untrusted pickle weights merely to inspect metadata.

No Wolf path exists under the inspected repository logs. Existing explicit Wolf files in the isolated probe directory are UNTRAINED_FIXTURE and INVALID test files, not teachers. The earlier saved audit searched1795 local weight-like files without finding a matching Wolf configuration; this is not a claim that no Wolf weight exists anywhere on the computer. Preserve all Black archives and source packages.

### Selective use of old policies remains possible

Do not equate missing compatible teachers with a ban on the user's requested selective old-policy demonstrations. Implement a disabled-by-default, separately versioned adapter and eligibility mask:

1. Verify source robot, asset/axis/joint order, observation reconstruction, normalization, history, defaults, PD, action scales, timing, checkpoint hash, and task competence. Unknown values fail closed.
2. For a verified same-robot old position policy, decode old raw action to its physical joint target, then encode that target relative to the new default and scale. For wheels decode signed physical velocity and re-encode in the new signed scale. Account explicitly for axis/order transforms and clamp residuals; invalid or saturated labels must be flagged rather than silently accepted.
3. Equal physical targets do not imply equal torques or dynamics when PD, delays, or actuator models differ. Closed-loop target-profile validation is mandatory. Cross-robot policies need a real kinematic/dynamic retargeter, not the same16-dimensional vector.
4. Query eligible frozen teachers on the exact states visited by the student. Reconstruct the teacher's history and previous-action semantics from those states; do not pair labels from unrelated prerecorded trajectories. Record source identity, state/trajectory/seed, physical targets, adapted action, validity/confidence, and override mask.
5. Keep the adapter, teacher takeover probability, and BC loss independently switchable and checkpointed. Rollback must reproduce the unassisted baseline. Validate transferred skills separately from new teacher capability. This is selective advice, not a certified universal teacher.

## Resources and budget semantics

Read-only host query: RTX4060 Laptop8188MiB, approximately920MiB used,0% utilization,36 C. RAM30GiB with approximately15GiB available, no swap. Root volume had approximately100GiB available. No matching training process was active. Initial sandbox nvidia-smi could not access the driver; a permitted host query confirmed the GPU was healthy. This does not benchmark simulator throughput or prove4096 environments fit.

Default config:4096 environments, rollout100,5 epochs,4 minibatches, save interval50, max_iterations10000, seed42, PPO LR1e-4 with adaptive KL schedule. HIM estimator configured LR.001 is overridden by PPO's learning rate during every minibatch update. Each training iteration gives20 PPO optimizer steps and20 separate estimator optimizer steps for HIM; ordinary PPO has20 optimizer steps. Record these separately.

The user's normal training round minimum is2000 native training iterations, except documented clearly wrong direction or safety/resource/numerical failures. At defaults this is819.2M transitions; at128 environments with horizon100 it is25.6M. The128-environment setting is a capacity-probe candidate, not verified throughput. Keep horizon/optimizer settings fixed when comparing resource choices. RSL-RL learn adds the requested number of iterations to the starting counter, so checkpoint label1999 can represent2000 iterations from zero; labels are not iteration counts.

Use the isolated launcher `${WOLF_EVIDENCE_ROOT}/run_python.sh`, which already uses uv and isolates ROS PYTHONPATH/PYTHONHOME. CLI fields include `--agent.seed`, `--env.scene.num-envs`, `--agent.num-steps-per-env`, `--agent.algorithm.num-learning-epochs`, `--agent.algorithm.num-mini-batches`, and `--agent.max-iterations`. Prefer explicit local logging/no uploads unless external upload is authorized. Do not start concurrent training runs on the8GB GPU without capacity measurement.

## Executable dependency order

1. Preserve source/control/asset contracts and immutable original checkpoints; identify source code and complete method manifest in every new run.
2. Keep this HIM compatibility repair. Add finite-loss/gradient checks and separate counters before long learning. Establish contract-validated checkpoint load, atomic save, clear continuation semantics and warm-start provenance preservation.
3. Integrate each requested method behind independent options, with real losses/storage/optimizer wiring: Huber and student-state teacher labeling; dual reward-channel values/GAE; teacher terrain observation and attention; sequence/RSSM state/reset/terminal handling; planner reference feasibility and CLF. Do not call a network-only scaffold a working method.
4. Prove off-mode parity, gradient ownership, no privileged leakage to student, reset isolation, source identity, roundtrip save/load, finite synthetic updates and failure rollback. Explicitly distinguish unit-update tests from any training campaign; no such learning update was run by this audit.
5. Run bounded frozen capacity/physics checks before choosing environment count. Establish held-out scenario and seed matrices, teacher eligibility and independent skill/safety metrics. No training reward alone promotes a teacher.
6. Then perform complete>=2000-iteration rounds and controlled ablations using the same transition budgets, with latest/best/checkpoint evidence, wall time, memory, return components, tracking, termination, and intervention metrics. No immediate training was launched by this audit.
