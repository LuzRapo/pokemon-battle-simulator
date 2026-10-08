"""PPO self-play for Mirror Battles, on the Rust engine's `VecEnv`."""

import argparse
import csv
import json
import random
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray
from torch import Tensor, nn

from battle_sim.ag_sets import mirror_team
from battle_sim.differential import encode_spec
from battle_sim.rust_bridge import RustVecEnv, database, load
from training.model import ModelConfig, PolicyValueNet, parameter_count


@dataclass(frozen=True)
class Config:
    envs: int = 96
    env_threads: int = 3
    torch_threads: int = 3
    decisions_per_update: int = 16_384
    epochs: int = 3
    minibatch: int = 4_096
    learning_rate: float = 3e-4
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gae_lambda: float = 0.95
    max_turns: int = 300
    past_opponent: float = 0.2  # share of battles with a past snapshot on one side
    snapshot_every: int = 20  # updates
    snapshots_kept: int = 30
    checkpoint_every: int = 5
    eval_every: int = 10
    eval_battles: int = 200
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)


# This server has four cores and a Discord bot on one of them; the PC has a GPU and a few more.
PRESETS: dict[str, Config] = {
    "server": Config(),
    "gpu": Config(envs=1024, env_threads=0, torch_threads=4, decisions_per_update=65_536, minibatch=8_192),
    "smoke": Config(envs=32, decisions_per_update=4_096, minibatch=1_024, eval_every=5, eval_battles=100),
}

# What the network reads for a batch of decisions: ids, Pokemon floats, field floats, legal mask.
type NetInputs = tuple[NDArray[np.int64], NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]


@dataclass(frozen=True)
class Decisions:
    """Learner decisions as parallel columns, one row per decision."""

    ids: NDArray[np.int32]
    pokemon: NDArray[np.float32]
    field: NDArray[np.float32]
    mask: NDArray[np.bool_]
    action: NDArray[np.int64]
    logp: NDArray[np.float32]
    value: NDArray[np.float32]
    trajectory: NDArray[np.int64]

    def rows(self, keep: NDArray[np.bool_]) -> "Decisions":
        return Decisions(
            self.ids[keep],
            self.pokemon[keep],
            self.field[keep],
            self.mask[keep],
            self.action[keep],
            self.logp[keep],
            self.value[keep],
            self.trajectory[keep],
        )

    @staticmethod
    def join(parts: Sequence["Decisions"]) -> "Decisions":
        return Decisions(
            np.concatenate([part.ids for part in parts]),
            np.concatenate([part.pokemon for part in parts]),
            np.concatenate([part.field for part in parts]),
            np.concatenate([part.mask for part in parts]),
            np.concatenate([part.action for part in parts]),
            np.concatenate([part.logp for part in parts]),
            np.concatenate([part.value for part in parts]),
            np.concatenate([part.trajectory for part in parts]),
        )


@dataclass(frozen=True)
class Batch:
    """Finished decisions with their advantages and returns."""

    decisions: Decisions
    advantage: NDArray[np.float32]
    returns: NDArray[np.float32]


@dataclass
class Seating:
    """Which trajectory each side of each battle slot feeds (-1 a snapshot, -2 empty), and its opponent."""

    trajectory_of: NDArray[np.int64]
    opponent_of: list[Path | None]
    next_battle: int = 0


class Rollouts:
    """Every learner decision since the last update, tagged with its trajectory."""

    def __init__(self) -> None:
        self.chunks: list[Decisions] = []
        self.rows: dict[int, int] = {}  # trajectory -> decisions recorded so far
        self.results: dict[int, float] = {}  # trajectory -> its reward, once its battle is over
        self.dropped: set[int] = set()
        self.ready = 0  # decisions belonging to trajectories with a result

    def add(self, decisions: Decisions) -> None:
        self.chunks.append(decisions)
        for trajectory, count in zip(*np.unique(decisions.trajectory, return_counts=True), strict=True):
            self.rows[int(trajectory)] = self.rows.get(int(trajectory), 0) + int(count)

    def finish(self, trajectory: int, reward: float) -> None:
        self.results[trajectory] = reward
        self.ready += self.rows.pop(trajectory, 0)

    def drop(self, trajectory: int) -> None:
        """A battle the engine could not finish: its decisions teach nothing."""
        self.dropped.add(trajectory)
        self.rows.pop(trajectory, None)

    def take(self, gae_lambda: float) -> Batch:
        """The finished trajectories, with advantages and returns; unfinished ones stay behind."""
        columns = Decisions.join(self.chunks)
        finished = np.isin(columns.trajectory, list(self.results))
        kept = ~finished & ~np.isin(columns.trajectory, list(self.dropped))
        done = columns.rows(finished)
        reward = np.array([self.results[int(t)] for t in done.trajectory], dtype=np.float32)
        advantage, returns = _advantages(done.trajectory, done.value, reward, gae_lambda)
        self.chunks = [columns.rows(kept)]
        self.results, self.dropped, self.ready = {}, set(), 0
        return Batch(done, advantage, returns)


def _advantages(
    trajectory: NDArray[np.int64], value: NDArray[np.float32], reward: NDArray[np.float32], gae_lambda: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """GAE with γ = 1 and the reward only at each trajectory's last decision."""
    advantage = np.zeros_like(value)
    following: dict[int, tuple[float, float]] = {}  # trajectory -> (next value, running advantage)
    for row in range(len(value) - 1, -1, -1):
        key = int(trajectory[row])
        if key in following:
            next_value, running = following[key]
            delta = next_value - value[row]
        else:
            running = 0.0
            delta = reward[row] - value[row]
        running = float(delta + gae_lambda * running)
        advantage[row] = running
        following[key] = (float(value[row]), running)
    return advantage, advantage + value


class Trainer:
    def __init__(self, run: Path, config: Config, device: torch.device) -> None:
        self.run = run
        self.config = config
        self.device = device
        torch.manual_seed(config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.teams = random.Random(config.seed)
        self.model = PolicyValueNet(config.model).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.snapshots: list[Path] = []
        self.opponents: dict[Path, PolicyValueNet] = {}
        self.update = 0
        self.decisions = 0
        self.battles = 0
        self.seconds = 0.0

    # --- persistence -------------------------------------------------------------------------

    def save(self) -> None:
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "model_config": self.config.model.to_dict(),
            "update": self.update,
            "decisions": self.decisions,
            "battles": self.battles,
            "seconds": self.seconds,
            "snapshots": [str(path.relative_to(self.run)) for path in self.snapshots],
        }
        partial = self.run / "latest.pt.partial"
        torch.save(state, partial)
        partial.replace(self.run / "latest.pt")

    def restore(self) -> None:
        state = torch.load(self.run / "latest.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.update, self.decisions, self.battles = state["update"], state["decisions"], state["battles"]
        self.seconds = state["seconds"]
        self.snapshots = [self.run / path for path in state["snapshots"]]
        self.rng = np.random.default_rng([self.config.seed, self.update])
        self.teams = random.Random(f"{self.config.seed}:{self.update}")

    def snapshot(self) -> None:
        path = self.run / "snapshots" / f"{self.update:06d}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save({"model": self.model.state_dict(), "model_config": self.config.model.to_dict()}, path)
        self.snapshots.append(path)
        while len(self.snapshots) > self.config.snapshots_kept:
            # Keep the oldest as a fixed yardstick; thin out the middle.
            gone = self.snapshots.pop(1)
            self.opponents.pop(gone, None)
            gone.unlink(missing_ok=True)

    def opponent(self, path: Path) -> PolicyValueNet:
        if path not in self.opponents:
            net = PolicyValueNet(self.config.model).to(self.device)
            net.load_state_dict(torch.load(path, map_location=self.device, weights_only=True)["model"])
            self.opponents[path] = net.eval()
        return self.opponents[path]

    # --- playing -----------------------------------------------------------------------------

    def team_json(self) -> list[str]:
        encoded = json.dumps([encode_spec(spec) for spec in mirror_team(self.teams)])
        return [encoded, encoded]

    @torch.no_grad()
    def act(self, net: nn.Module, batch: NetInputs, greedy: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        ids, pokemon, field_, mask = (torch.from_numpy(a).to(self.device) for a in batch)
        logits, value = net(ids, pokemon, field_, mask)
        action = logits.argmax(dim=-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)
        return action.cpu(), logp.cpu(), value.cpu()

    def collect(self, env: RustVecEnv, rollouts: Rollouts, seating: Seating) -> dict[str, float]:
        """Play until `decisions_per_update` learner decisions belong to finished battles."""
        stats = {"battles": 0, "turns": 0, "past_played": 0, "past_won": 0, "timeouts": 0, "failed": 0}
        trajectory_of, opponent_of = seating.trajectory_of, seating.opponent_of
        self.model.eval()
        while rollouts.ready < self.config.decisions_per_update:
            for index in env.empty():
                self.start(env, index, seating)
            rows, sides, _, ids, pokemon, field_, mask = env.observe()
            actions = np.zeros(len(rows), dtype=np.int64)
            owner = trajectory_of[rows, sides]
            learner = owner >= 0
            if learner.any():
                picked: NetInputs = (ids[learner], pokemon[learner], field_[learner], mask[learner])
                action, logp, value = self.act(self.model, picked)
                actions[learner] = action.numpy()
                rollouts.add(
                    Decisions(
                        ids=picked[0].astype(np.int32),
                        pokemon=picked[1],
                        field=picked[2],
                        mask=picked[3],
                        action=action.numpy(),
                        logp=logp.numpy(),
                        value=value.numpy(),
                        trajectory=owner[learner],
                    )
                )
                self.decisions += int(learner.sum())
            for path in {opponent_of[r] for r in rows[~learner]}:
                assert path is not None
                chosen = ~learner & np.array([opponent_of[r] == path for r in rows])
                batch: NetInputs = (ids[chosen], pokemon[chosen], field_[chosen], mask[chosen])
                actions[chosen] = self.act(self.opponent(path), batch)[0].numpy()
            env.act(rows, sides, actions)
            for index, ending, winner, turns in env.collect():
                self.finish(index, ending, winner, turns, rollouts, seating, stats)
        return {k: float(v) for k, v in stats.items()}

    def start(self, env: RustVecEnv, index: int, seating: Seating) -> None:
        env.reset(index, self.team_json(), int(self.rng.integers(2**63)))
        battle = seating.next_battle
        seating.next_battle += 1
        seating.trajectory_of[index] = [2 * battle, 2 * battle + 1]
        seating.opponent_of[index] = None
        if self.snapshots and self.rng.random() < self.config.past_opponent:
            seating.opponent_of[index] = self.snapshots[int(self.rng.integers(len(self.snapshots)))]
            seating.trajectory_of[index, int(self.rng.integers(2))] = -1

    def finish(
        self,
        index: int,
        ending: str,
        winner: int | None,
        turns: int,
        rollouts: Rollouts,
        seating: Seating,
        stats: dict[str, int],
    ) -> None:
        stats["battles"] += 1
        stats["turns"] += turns
        self.battles += 1
        past = seating.opponent_of[index] is not None
        for side in (0, 1):
            trajectory = int(seating.trajectory_of[index, side])
            if trajectory < 0:
                continue
            if ending.startswith("failed"):
                rollouts.drop(trajectory)
                continue
            rollouts.finish(trajectory, 0.0 if winner is None else (1.0 if winner == side else -1.0))
            if past:
                stats["past_played"] += 1
                stats["past_won"] += int(winner == side)
        stats["timeouts"] += int(ending == "timeout")
        stats["failed"] += int(ending.startswith("failed"))
        seating.trajectory_of[index] = [-2, -2]

    # --- learning ----------------------------------------------------------------------------

    def learn(self, rollouts: Rollouts) -> dict[str, float]:
        batch = rollouts.take(self.config.gae_lambda)
        decisions = batch.decisions
        columns = {
            "ids": decisions.ids.astype(np.int64),
            "pokemon": decisions.pokemon,
            "field": decisions.field,
            "mask": decisions.mask,
            "action": decisions.action,
            "logp": decisions.logp,
            "value": decisions.value,
            "advantage": batch.advantage,
            "returns": batch.returns,
        }
        tensors = {name: torch.from_numpy(values).to(self.device) for name, values in columns.items()}
        advantage = tensors["advantage"]
        tensors["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        self.model.train()
        totals: dict[str, float] = {}
        steps = 0
        for _ in range(self.config.epochs):
            for rows in _minibatches(len(advantage), self.config.minibatch, self.rng):
                losses = self.loss({name: values[rows] for name, values in tensors.items()})
                self.optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                for name, value in losses.items():
                    totals[name] = totals.get(name, 0.0) + float(value.detach())
                steps += 1
        totals = {name: value / max(steps, 1) for name, value in totals.items()}
        totals["samples"] = float(len(advantage))
        return totals

    def loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        c = self.config
        logits, value = self.model(batch["ids"], batch["pokemon"], batch["field"], batch["mask"])
        distribution = torch.distributions.Categorical(logits=logits)
        logp = distribution.log_prob(batch["action"])
        ratio = torch.exp(logp - batch["logp"])
        advantage = batch["advantage"]
        policy = -torch.min(ratio * advantage, ratio.clamp(1 - c.clip, 1 + c.clip) * advantage).mean()
        clipped_value = batch["value"] + (value - batch["value"]).clamp(-c.clip, c.clip)
        value_loss = torch.max((value - batch["returns"]) ** 2, (clipped_value - batch["returns"]) ** 2).mean()
        # Only legal actions carry probability, so this is the entropy over the legal ones.
        entropy = distribution.entropy().mean()
        loss = policy + c.value_coef * value_loss - c.entropy_coef * entropy
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (logp - batch["logp"])).mean()
            clip_fraction = ((ratio - 1).abs() > c.clip).float().mean()
        return {
            "loss": loss,
            "policy_loss": policy,
            "value_loss": value_loss,
            "entropy": entropy,
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
        }

    # --- measuring ---------------------------------------------------------------------------

    def evaluate_against_random(self) -> float:
        """The learner, greedy, against a uniformly random legal player: each side half the time."""
        env = load().VecEnv(database(), min(self.config.eval_battles, 256), max_turns=self.config.max_turns)
        teams = random.Random(f"eval:{self.update}")
        rng = np.random.default_rng(self.update)
        learner_side: dict[int, int] = {}
        started = won = scored = 0
        self.model.eval()
        while scored < self.config.eval_battles:
            for index in env.empty():
                if started < self.config.eval_battles:
                    encoded = json.dumps([encode_spec(spec) for spec in mirror_team(teams)])
                    env.reset(index, [encoded, encoded], started)
                    learner_side[index] = started % 2
                    started += 1
            rows, sides, _, ids, pokemon, field_, mask = env.observe()
            ours = np.array([learner_side[r] == s for r, s in zip(rows, sides, strict=True)], dtype=bool)
            actions = np.where(mask, rng.random(mask.shape), -1.0).argmax(axis=1)
            if ours.any():
                picked: NetInputs = (ids[ours], pokemon[ours], field_[ours], mask[ours])
                actions[ours] = self.act(self.model, picked, greedy=True)[0].numpy()
            env.act(rows, sides, actions)
            for index, _, winner, _ in env.collect():
                scored += 1
                won += int(winner == learner_side[index])
        return won / scored


def _minibatches(size: int, batch: int, rng: np.random.Generator) -> Iterator[Tensor]:
    order = torch.from_numpy(rng.permutation(size))
    for start in range(0, size, batch):
        yield order[start : start + batch]


def train(run: Path, config: Config, hours: float | None, updates: int | None, resume: bool) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config.torch_threads)
    trainer = Trainer(run, config, device)
    if resume:
        trainer.restore()
    else:
        trainer.snapshot()  # the untrained network: the first past opponent and a fixed yardstick
    logger.info(f"{parameter_count(trainer.model):,} parameters on {device}; update {trainer.update}")

    env = load().VecEnv(database(), config.envs, max_turns=config.max_turns, threads=config.env_threads)
    seating = Seating(np.full((config.envs, 2), -2, dtype=np.int64), [None] * config.envs)
    rollouts = Rollouts()
    log_path = run / "log.csv"
    fresh = not log_path.exists()
    handle = log_path.open("a", newline="")
    writer: csv.DictWriter[str] | None = None
    started = time.monotonic()
    last_update = trainer.update
    while (hours is None or time.monotonic() - started < hours * 3600) and (
        updates is None or trainer.update - last_update < updates
    ):
        tick = time.monotonic()
        played = trainer.collect(env, rollouts, seating)
        collected = time.monotonic()
        learned = trainer.learn(rollouts)
        trainer.update += 1
        trainer.seconds += time.monotonic() - tick
        row: dict[str, float | int | str] = {
            "update": trainer.update,
            "decisions": trainer.decisions,
            "battles": trainer.battles,
            "hours": round(trainer.seconds / 3600, 4),
            "decisions_per_second": round(learned["samples"] / (time.monotonic() - tick), 1),
            "collect_seconds": round(collected - tick, 2),
            "learn_seconds": round(time.monotonic() - collected, 2),
            "turns_per_battle": round(played["turns"] / max(played["battles"], 1), 1),
            "past_win_rate": round(played["past_won"] / played["past_played"], 3) if played["past_played"] else "",
            "timeouts": int(played["timeouts"]),
            "failed": int(played["failed"]),
            **{name: round(value, 5) for name, value in learned.items()},
            "random_win_rate": "",
        }
        if trainer.update % config.eval_every == 0:
            row["random_win_rate"] = round(trainer.evaluate_against_random(), 3)
        if trainer.update % config.snapshot_every == 0:
            trainer.snapshot()
        if trainer.update % config.checkpoint_every == 0:
            trainer.save()
        if writer is None:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if fresh:
                writer.writeheader()
        writer.writerow(row)
        handle.flush()
        logger.info(
            f"update {row['update']}: {row['decisions_per_second']} dec/s, value {row['value_loss']}, "
            f"entropy {row['entropy']}, kl {row['approx_kl']}, vs past {row['past_win_rate']}, "
            f"vs random {row['random_win_rate']}"
        )
    trainer.save()
    handle.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="directory for this run's files")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="server")
    parser.add_argument("--resume", action="store_true", help="continue the run in --run")
    parser.add_argument("--hours", type=float, help="stop after this long (default: never)")
    parser.add_argument("--updates", type=int, help="stop after this many updates")
    for f in fields(Config):
        if f.name != "model":
            parser.add_argument(f"--{f.name.replace('_', '-')}", type=type(f.default), default=None)
    args = parser.parse_args()

    run: Path = args.run
    saved = run / "config.json"
    if args.resume:
        stored = json.loads(saved.read_text())
        model = ModelConfig(**stored.pop("model"))
        config = replace(Config(**stored), model=model)
    else:
        if saved.exists():
            raise SystemExit(f"{run} already holds a run; pass --resume, or choose another --run")
        config = PRESETS[args.preset]
    overrides = {f.name: vars(args)[f.name] for f in fields(Config) if f.name != "model"}
    config = replace(config, **{name: value for name, value in overrides.items() if value is not None})
    run.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(asdict(config), indent=1))
    train(run, config, args.hours, args.updates, args.resume)


if __name__ == "__main__":
    main()
