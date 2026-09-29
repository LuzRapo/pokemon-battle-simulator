"""The policy/value network: one token per Pokemon plus one for the field, through a transformer.

Inputs are exactly `battle_sim.rl.encode`'s arrays (and `VecEnv.observe()`'s), batched:

    ids      int64   [B, 12, 7]     species, ability, item, four moves — `rust/data/vocab.json` ids
    pokemon  float32 [B, 12, F]     own team in slots 0..5, the opponent's in 6..11
    field    float32 [B, FIELD]
    mask     bool    [B, 14]        which of `choices`' 14 actions are legal

Outputs are masked logits over the 14 actions and a value in [-1, 1] — the expected result for the
side deciding (+1 a win, -1 a loss).

The heads follow the action space's shape rather than flattening everything into one vector:
switching to team slot `i` is scored from slot `i`'s own token, and using move `k` (plain or as a
Z-move) from the active Pokemon's token alongside move `k`'s own embedding. The field token, which
attends to everything, carries the value and a whole-board context to both.
"""

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from battle_sim.rl import encode
from battle_sim.rl.vocab import vocabulary

ACTIONS = 14
MOVES = 4
SWITCH = 8
# Where `active` sits among a Pokemon's floats (see `encode._floats`): present, HP, max HP, five
# stats, the stages, the statuses, status turns, fainted — then active.
ACTIVE = 1 + 1 + 1 + 5 + len(encode.STAGES) + len(encode.STATUSES) + 1 + 1


@dataclass(frozen=True)
class ModelConfig:
    width: int = 128
    layers: int = 3
    heads: int = 4
    feedforward: int = 256

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class PolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        d = config.width
        tables = vocabulary()
        self.species = nn.Embedding(len(tables["species"]) + 1, d, padding_idx=0)
        self.ability = nn.Embedding(len(tables["abilities"]) + 1, d, padding_idx=0)
        self.item = nn.Embedding(len(tables["items"]) + 1, d, padding_idx=0)
        self.move = nn.Embedding(len(tables["moves"]) + 1, d, padding_idx=0)
        self.floats = nn.Sequential(nn.Linear(encode.POKEMON_FLOATS, d), nn.GELU(), nn.Linear(d, d))
        self.slot = nn.Embedding(encode.SLOTS, d)  # whose Pokemon, and which team index
        self.field = nn.Sequential(nn.Linear(encode.FIELD_FLOATS, d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(
            d, config.heads, config.feedforward, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.switch_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.move_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 2))  # plain, Z
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ids: Tensor, pokemon: Tensor, field: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch = ids.shape[0]
        moves = self.move(ids[:, :, 3:])  # [B, 12, 4, d]
        tokens = (
            self.species(ids[:, :, 0])
            + self.ability(ids[:, :, 1])
            + self.item(ids[:, :, 2])
            + moves.mean(dim=2)
            + self.floats(pokemon)
            + self.slot.weight.unsqueeze(0)
        )
        sequence = torch.cat([self.field(field).unsqueeze(1), tokens], dim=1)  # [B, 13, d]
        out = self.norm(self.encoder(sequence))
        board, own = out[:, 0], out[:, 1:7]

        switch = self.switch_head(torch.cat([own, board.unsqueeze(1).expand_as(own)], dim=-1)).squeeze(-1)
        # The active Pokemon's token and moves; before the leads are out nothing is active, and the
        # mask allows no move then anyway.
        active = pokemon[:, :6, ACTIVE]  # [B, 6], one-hot or all zero
        active_token = torch.einsum("bs,bsd->bd", active, own)
        active_moves = torch.einsum("bs,bsmd->bmd", active, moves[:, :6])  # [B, 4, d]
        context = torch.cat([active_token, board], dim=-1).unsqueeze(1).expand(batch, MOVES, -1)
        move = self.move_head(torch.cat([context, active_moves], dim=-1))  # [B, 4, 2]

        logits = torch.cat([move[..., 0], move[..., 1], switch], dim=-1)
        logits = logits.masked_fill(~mask, -1e9)
        value = torch.tanh(self.value_head(board)).squeeze(-1)
        return logits, value


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
