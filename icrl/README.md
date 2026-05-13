# icrl Design

## Goals

`icrl/` is a clean refactor of the current `experiments/` rollout logic for a new multi-role training setup:

1. Keep Hydra-based experiment configuration.
2. Keep prompt templates decoupled in a dedicated directory.
3. Make the rollout logic explicit around `episode -> round -> role`.
4. Support future extension of verifier / critic strategies without rewriting the executor loop.

The target interaction pattern is:

1. `executor` runs one full task trajectory.
2. `verifier` judges whether the trajectory solved the task.
3. If `executor` succeeds, the episode stops.
4. If `executor` does not succeed and the round budget allows, `critic` produces feedback.
5. A new `executor` round restarts the task with the critic feedback in context.

`max_rounds >= 1`.

- `max_rounds = 1`: one `executor + verifier` pair only, so `0` improvement steps.
- `max_rounds = 2`: one improvement cycle.
- In general, `max_improvements = max_rounds - 1`.

## Core Abstractions

### Episode

An episode corresponds to one prompt/task instance and may contain multiple rounds.

Each episode owns:

- one task id / task description
- one env client lifecycle
- multiple role trajectories
- round-level records
- role-specific rewards

### Round

A round is the smallest improvement unit.

Round `r` contains:

1. one executor rollout
2. one verifier judgment
3. optionally one critic response

Critic exists only when:

- executor does not succeed
- `r < max_rounds`

### Role Trajectory

Each role trajectory is stored as an independent `Sample`, so SLIME can train them separately.

Required roles:

- `executor`
- `verifier`
- `critic`

Important consequence:

- rewards are assigned per role sample
- group relative advantage is normalized separately for each role

## Control Flow

### Round 1

1. Open env once for the episode and obtain the initial task observation.
2. Run `executor`.
3. Compute task outcome reward from env result.
4. Run `verifier` on the executor trajectory.
5. Compare verifier prediction with true executor outcome to get verifier reward.
6. If `executor` succeeds, stop.
7. Otherwise, if another round is available, run `critic`.

### Round r > 1

1. Reset env to the same task again.
2. Run a new `executor`, conditioning only on the previous round's critic feedback.
3. Run `verifier`.
4. If `executor` does not succeed and another round is available, run `critic` again.

The stopping decision is driven by `executor` task success, not verifier prediction. Verifier is still trained each round as a separate role, but it does not control whether another executor round is launched.

### Env lifecycle

The env client is opened once per episode, reset at the start of each executor round, and closed only when the full episode ends.

This avoids the earlier temporary `open -> read init_obs -> close` pattern and keeps the episode boundary explicit.

### Train vs Eval return value

The rollout control flow is the same for training and evaluation: both run the same conditional `executor -> verifier -> critic -> next executor` process.

The difference is only what gets returned to the outer rollout framework:

- train returns all role samples from the full episode
- eval returns the last executor sample

So:

- when `max_rounds = 1`, eval returns the first executor sample
- when `max_rounds >= 2`, eval returns the improved final executor sample if later rounds were executed

## Reward Design

### Executor reward

Executor uses the task outcome directly.

- For `sciworld`, the terminal reward is normalized by `/100`.
- For other envs, the original env reward is kept.

So the executor reward is the normalized outcome reward, not an extra thresholded binary value.

### Verifier reward

Verifier reward measures calibration against the normalized executor outcome:

- `1` if verifier prediction matches the executor success label
- `0` otherwise

where:

`exec_success = 1[normalized_exec_reward == 1.0]`

Formally:

`reward_verifier = 1[pred_success == exec_success]`

### Critic reward

Critic reward measures the outcome delta induced before and after critique.

Formally:

`reward_critic = next_exec_reward - current_exec_reward`

Properties:

- improvement gives positive reward
- no change gives `0`
- regression gives negative reward

This is more general than a binary improvement indicator and works naturally for non-binary outcome rewards.

## Advantage Normalization

Because one episode contains mixed roles, rewards must not be normalized together.

The new post-processing groups samples by:

`(group_index, role)`

instead of only `group_index`.

This gives role-specific group relative advantage:

- executor samples compete with executor samples from the same prompt group
- verifier samples compete only with verifier samples
- critic samples compete only with critic samples

The supported normalization mode is:

- `role_norm`: normalize within each `(group_index, role)` bucket

Reason:

- executor/verifier/critic within the same prompt group should not be normalized together

## Prompt / Context Semantics

### Executor input

The executor receives:

- the task
- the current round initial observation
- only the previous round's critic feedback, if it exists

Important:

- full `critic_history` is kept in sample metadata for analysis
- but the model input only contains `previous_critic`

This keeps the training signal focused on local improvement instead of accumulating long critic histories into the prompt.

### Verifier output

The verifier prompt currently expects:

- input contains task + trajectory summary only; it must not see the executor outcome reward
- `<verdict>correct</verdict>` if the trajectory solved the task
- `<verdict>incorrect</verdict>` otherwise

Even though the tag text is `correct/incorrect`, semantically it means trajectory success prediction.

### Critic output

The critic prompt currently expects:

- input contains task + trajectory summary + normalized executor outcome reward
- `<critic>...</critic>`

The enclosed text is fed to the next executor round as the previous critic feedback.

## Directory Layout

```text
icrl/
  README.md
  __init__.py
  hydra_runner.py
  generate.py
  prompts.py
  rewards.py
  logging_utils.py
  schema.py
  runtime.py
  templates/
    executor_system.j2
    verifier_system.j2
    critic_system.j2
  hydra_conf/
    algo/
    checkpoint/
    config.yaml
    custom/
      icrl.yaml
    eval/
    gpu/
    logging/
    misc/
    model/
    optimizer/
    paths/
    rollout/
    sglang/
```

## Module Responsibilities

### `generate.py`

Owns episode orchestration.

- round loop
- executor / verifier / critic dispatch
- env lifecycle
- reward assignment
- final sample flattening
- success-label derivation from normalized reward

### `runtime.py`

Shared low-level rollout helpers.

- token accounting
- single-turn generation call
- sample initialization/finalization
- repeated action detection

### `prompts.py`

Owns prompt rendering and template lookup.

- role -> template mapping
- environment registry reuse
- template validation

### `schema.py`

Pure dataclasses for round and episode records.

This keeps metadata conventions explicit instead of spreading them through ad hoc dict keys.

### `rewards.py`

Owns reward post-processing for training:

- raw reward extraction
- role-aware group normalization

### `logging_utils.py`

Owns rollout artifact logging for debugging and later analysis.

## Extensibility

This structure is designed for the following future changes:

1. Add retrieval / experience bank only to verifier or critic without touching executor loop.
2. Replace critic reward with a shaped delta metric.
3. Allow verifier to output uncertainty and use that for stopping rules.
4. Add more roles such as planner / rewriter / memory agent.
5. Train only a subset of roles by filtering samples before post-processing.

## Current Implementation Scope

This refactor intentionally reuses stable pieces from `experiments/`:

- env client initialization
- environment parsing
- basic multi-turn executor loop patterns
- Hydra CLI argument rendering helpers

But the Hydra config tree itself is now copied into `icrl/hydra_conf/` and maintained independently.

In particular:

- `icrl/hydra_conf/config.yaml` is the standalone Hydra entry config
- `icrl/hydra_conf/paths/local.yaml` writes outputs under `icrl/runs/...`
- `icrl/hydra_conf/custom/icrl.yaml` contains the icrl-specific custom knobs

What changes in `icrl/` is the orchestration boundary:

- no more verifier logic hidden as a special subagent path inside one loop
- reward ownership is explicit by role
- prompt templates are separated by role in a local template directory
- executor only consumes the previous critic, while full critic history is preserved for analysis
