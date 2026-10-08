# Self-play training for Mirror Battles

A network learns Mirror Battles by playing itself on the Rust engine. Both sides get the same six
from the gen7ag pool (`battle_sim.ag_sets.mirror_team`, the sampler the bot's Mirror Mode uses), with
full information.

| Piece | Where |
|---|---|
| Legal actions, player-made leads and replacements | `rust/src/choices.rs`, `rust/src/turn.rs` |
| Observation encoder (and its Python mirror) | `rust/src/obs.rs`, `battle_sim/rl/encode.py` |
| Batched environment | `rust/src/env.rs` → `pokemon_engine_rs.VecEnv` |
| Network, PPO, export, evaluation | `training/` (needs torch) |
| Playing a trained network in the Python engine | `battle_sim/rl/net_player.py` (onnxruntime only) |

Every Rust piece is checked against the Python engine: decisions, legal actions, observations and
the environment's sequencing. See `docs/rust-port-plan.md` section 8.

## The loop

- **Action space.** 14 actions: moves 0–3, the same four as Z-moves 4–7, and switch to (or lead
  with) team slot 8–13. Illegal ones are masked. A decision with a single legal answer is made by the
  environment and never reaches the network. Mega Evolution stays automatic, as in both engines.
- **Model** (`training/model.py`). One token per Pokemon plus a field token, through a 3-layer
  transformer of width 128, about 0.9M parameters. The switch heads read each team slot's token;
  the move heads read the active Pokemon's token and each move's embedding; the value head reads
  the field token.
- **Training** (`training/train.py`). PPO over the learner's own decisions. Most battles are the
  learner against itself. 20% put a past snapshot on one side, sampled from the last 30 kept, one
  saved every 20 updates. The reward is the result only: +1, -1, or 0 for a draw or the 300-turn
  cap. Advantages come from GAE with γ = 1 and λ = 0.95.

## On the RTX 3060 (WSL2)

1. **Windows side.** Install a current NVIDIA driver for Windows. WSL gets CUDA from it, so do
   **not** install a Linux NVIDIA driver inside WSL.
2. **WSL.** From PowerShell, run `wsl --install -d Ubuntu-24.04`. Then, inside Ubuntu, `nvidia-smi`
   should list the 3060. Give WSL room in `%UserProfile%\.wslconfig` if you like; half the RAM is
   the default:

   ```ini
   [wsl2]
   memory=16GB
   ```
3. **Toolchain**, inside WSL. Keep the checkouts in the Linux filesystem (`~/`), not under `/mnt/c`,
   which is many times slower.

   ```bash
   sudo apt update && sudo apt install -y build-essential pkg-config git tmux
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   curl -LsSf https://astral.sh/uv/install.sh | sh
   exec $SHELL
   ```
4. **Both repos, as siblings.** The evaluator reads the bot's champion genome from
   `../sir-meowfred/pokemon/champion.json`.

   ```bash
   cd ~ && git clone <pokemon-battle-simulator> && git clone <sir-meowfred>
   cd ~/pokemon-battle-simulator
   git checkout feature/vectorised-engine
   uv sync --group ml-cuda
   (cd rust && cargo build --release)
   ```
5. **Check.**

   ```bash
   uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"
   uv run pytest -q tests/test_rl_env.py tests/test_rl_encode.py tests/test_training.py
   uv run python tools/bench_vecenv.py --envs 1024          # the engine's ceiling on this CPU
   ```

## Running

```bash
tmux new -s train
uv run python -m training.train --run runs/mirror-1 --preset gpu --hours 10
# Ctrl-b d to detach; `tmux attach -t train` to come back.
```

- **Resume** with `--resume`. It picks up `runs/mirror-1/latest.pt`, which is rewritten every 5
  updates, and the run's own `config.json`:

  ```bash
  uv run python -m training.train --run runs/mirror-1 --resume --hours 10
  ```
- **Tune the preset** with `--envs`, `--decisions-per-update`, `--minibatch` or `--learning-rate`,
  which override it. The `gpu` preset runs 1024 battles at once, with 65,536 decisions per update
  in minibatches of 8,192.
- **Stop** with Ctrl-C. At most the updates since the last checkpoint are lost.

### What to watch in `runs/<run>/log.csv`

| Column | Healthy |
|---|---|
| `decisions_per_second` | steady; `collect_seconds` vs `learn_seconds` says which side is the bottleneck |
| `random_win_rate` | climbs well above 0.5 within the first hours (every 10 updates, greedy vs uniform random) |
| `past_win_rate` | stays above 0.5: the learner keeps beating its older selves |
| `entropy` | falls slowly; a collapse towards 0 early means the entropy bonus is too small |
| `approx_kl`, `clip_fraction` | around 0.01 and 0.1; much higher means the learning rate is too high |
| `value_loss` | falls as the value head learns who is winning |
| `timeouts`, `failed` | near 0; `failed` means the engine refused a battle and should never happen |

For a quick look: `column -s, -t < runs/mirror-1/log.csv | less -S`.

## Is it any good? Evaluate against the bot's AI

`training/evaluate.py` plays the network in the **Python** engine, through `NetPlayer`, against
`SearchPlayer` built exactly as the bot's Mirror Mode builds it: the champion genome,
`exploit_p=1.0`, `tie_band=0.10`. Each pairing is played twice with the sides swapped, as
`duel.py` does.

```bash
uv run python -m training.export runs/mirror-1/latest.pt runs/mirror-1/policy.onnx
uv run python -m training.evaluate runs/mirror-1/policy.onnx --opponent random --pairs 200
uv run python -m training.evaluate runs/mirror-1/policy.onnx --opponent search --budget 30 --pairs 100
uv run python -m training.evaluate runs/mirror-1/policy.onnx --opponent search --pairs 50   # budget 1000, as the bot plays
```

On this server a budget-30 pairing (two battles) takes about 20 seconds per worker. Budget 1000
searches over 30 times as far, so start with 30. The
network has to beat budget 1000 decisively before it replaces the bot's Mirror Mode AI. That swap
is a separate change.

## Copying results back

`runs/` is gitignored, so checkpoints travel by `rsync`:

```bash
rsync -av runs/mirror-1/{policy.onnx,latest.pt,config.json,log.csv} <server>:pokemon-battle-simulator/runs/mirror-1/
```

`policy.onnx` is all that `NetPlayer` needs. `latest.pt` lets training resume elsewhere.

## Training on this server instead

This server has 4 cores and no GPU, and it runs the bot. The `server` preset uses 3 threads for
the environment and 3 for torch, which take turns rather than run together, and leaves a core for
the bot. Run it at low priority:

```bash
nice -n 10 uv run python -m training.train --run runs/server-1 --preset server
```

Expect roughly 650 decisions a second in total, which is about 7 battles a second, and most of the
time goes to the backward pass. The environment alone manages ~75k decisions a second
(`tools/bench_vecenv.py`), so the GPU is what changes the picture. For a hard cap rather than a
priority, use `systemd-run --user --scope -p CPUQuota=250% uv run python -m training.train ...`.
