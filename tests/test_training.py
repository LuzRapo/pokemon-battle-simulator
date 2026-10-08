"""The trainer's own arithmetic, and a network making it all the way into a Python-engine battle."""

import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

# Imported after the skip, since these need torch and the skip must run first.
from battle_sim.ag_sets import mirror_team  # noqa: E402 - see the skip above
from battle_sim.players import RandomPlayer  # noqa: E402 - see the skip above
from battle_sim.rl.net_player import NetPlayer  # noqa: E402 - see the skip above
from battle_sim.runner import run_battle  # noqa: E402 - see the skip above
from battle_sim.rust_bridge import LIBRARY  # noqa: E402 - see the skip above
from training.export import export  # noqa: E402 - see the skip above
from training.model import ModelConfig, PolicyValueNet  # noqa: E402 - see the skip above
from training.train import Config, Decisions, Rollouts, _advantages, train  # noqa: E402 - see the skip above

needs_bridge = pytest.mark.skipif(not LIBRARY.exists(), reason="run `cargo build --release` in rust/")
TINY = ModelConfig(width=32, layers=1, heads=2, feedforward=64)


def test_advantages_are_gae_with_the_reward_at_each_trajectorys_end() -> None:
    """Two interleaved trajectories: 7 won (+1), 9 lost (-1). γ = 1, λ = 0.95."""
    trajectory = np.array([7, 9, 7, 9, 7])
    value = np.array([0.1, 0.0, 0.2, -0.5, 0.3], dtype=np.float32)
    reward = np.array([1, -1, 1, -1, 1], dtype=np.float32)
    advantage, returns = _advantages(trajectory, value, reward, 0.95)
    won = [0.1 + 0.95 * (0.1 + 0.95 * 0.7), 0.1 + 0.95 * 0.7, 0.7]
    lost = [-0.5 + 0.95 * -0.5, -0.5]
    np.testing.assert_allclose(advantage, [won[0], lost[0], won[1], lost[1], won[2]], rtol=1e-6)
    np.testing.assert_allclose(returns, advantage + value)


def _decisions(trajectory: list[int], action: list[int]) -> Decisions:
    rows = len(trajectory)
    return Decisions(
        ids=np.zeros((rows, 12, 7), dtype=np.int32),
        pokemon=np.zeros((rows, 12, 1), dtype=np.float32),
        field=np.zeros((rows, 1), dtype=np.float32),
        mask=np.ones((rows, 14), dtype=np.bool_),
        action=np.array(action, dtype=np.int64),
        logp=np.zeros(rows, dtype=np.float32),
        value=np.zeros(rows, dtype=np.float32),
        trajectory=np.array(trajectory, dtype=np.int64),
    )


def test_unfinished_trajectories_wait_for_the_next_update() -> None:
    rollouts = Rollouts()
    rollouts.add(_decisions([1, 2, 1], [0, 1, 2]))
    rollouts.finish(1, 1.0)
    assert rollouts.ready == 2
    assert rollouts.take(0.95).decisions.action.tolist() == [0, 2]
    rollouts.add(_decisions([2], [3]))
    rollouts.finish(2, -1.0)
    assert rollouts.take(0.95).decisions.action.tolist() == [1, 3]


def test_an_exported_network_plays_a_whole_battle_in_the_python_engine(tmp_path: Path) -> None:
    model = tmp_path / "policy.onnx"
    export(PolicyValueNet(TINY).eval(), model)  # `export` also checks onnxruntime agrees with torch
    team = mirror_team(random.Random(0))
    result = run_battle(team, team, NetPlayer(model), RandomPlayer(0), seed=0, max_turns=300)
    assert result.turns > 0


@needs_bridge
def test_a_short_run_trains_checkpoints_and_resumes(tmp_path: Path) -> None:
    config = Config(
        envs=8,
        env_threads=1,
        torch_threads=1,
        decisions_per_update=256,
        minibatch=128,
        epochs=1,
        snapshot_every=1,
        checkpoint_every=1,
        eval_every=2,
        eval_battles=4,
        model=TINY,
    )
    train(tmp_path, config, hours=None, updates=2, resume=False)
    train(tmp_path, config, hours=None, updates=1, resume=True)
    log = (tmp_path / "log.csv").read_text().splitlines()
    assert [row.split(",")[0] for row in log[1:]] == ["1", "2", "3"]
    assert (tmp_path / "latest.pt").exists()
    assert len(list((tmp_path / "snapshots").glob("*.pt"))) == 4  # the untrained one, then one per update
