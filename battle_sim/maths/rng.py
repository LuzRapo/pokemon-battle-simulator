import random
from dataclasses import dataclass, field


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
        return self._engine.randrange(minimum, maximum)

    def roll_chance(self, probability: float) -> bool:
        """Return True with the given probability (0.0 to 1.0)."""
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")
        return self.random_probability() < probability
