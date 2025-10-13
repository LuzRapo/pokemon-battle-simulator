import random
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class RNG:
    seed: int | None = None
    _engine: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._engine = random.Random(self.seed)

    def random_probability(self) -> float:
        """Return a random float in the range [0.0, 1.0)."""
        return self._engine.random()

    def random_integer(self, minimum: int, maximum: int) -> int:
        """Return a random integer N such that minimum <= N < maximum."""
        return self._engine.randint(minimum, maximum)

    def random_choice(self, options: Sequence[Any]) -> Any:
        """Return a random element from a non-empty sequence."""
        if not options:
            raise ValueError("Cannot choose from an empty sequence.")
        return self._engine.choice(options)

    def shuffle_items(self, items: list[Any]) -> None:
        """Shuffle a list of items in-place."""
        self._engine.shuffle(items)

    def roll_chance(self, probability: float) -> bool:
        """Return True with the given probability (0.0 to 1.0)."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")
        return self.random_probability() < probability

    def roll_percentage(self, percent_chance: float) -> bool:
        """Return True with the given percentage chance (0 to 100)."""
        if not 0.0 <= percent_chance <= 100.0:
            raise ValueError("Percentage chance must be between 0 and 100.")
        return self.roll_chance(percent_chance / 100.0)

    def get_state(self) -> tuple[Any, ...]:
        """Return the internal RNG state for exact reproducibility."""
        return self._engine.getstate()

    def set_state(self, state: tuple[Any, ...]) -> None:
        """Restore a previously saved RNG state."""
        self._engine.setstate(state)

    def reseed(self, new_seed: int) -> None:
        """Reset the generator with a new seed value."""
        self.seed = new_seed
        self._engine.seed(new_seed)
