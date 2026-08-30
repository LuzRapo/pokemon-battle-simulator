"""Z-moves: once per battle, a held Z-Crystal upgrades one of the holder's moves.

Two kinds exist, and only one is modelled here:

- **Generic type crystals** (Firium Z, Flyinium Z, ...) upgrade any damaging move of their type. The
  vendored entry carries the right name and type but placeholder power (`basePower: 1`) and a
  `category` that ignores the base move, so both are taken from the base move instead: category
  directly, power through `_Z_POWER_TABLE`. The crystal -> type mapping is read out of the data
  rather than hand-listed, so it cannot drift.
- **Signature crystals** (Kommonium Z, Mimikium Z, ...) upgrade one specific move and carry real
  power in the data — but *which* move they attach to is not in the vendored move data at all (it
  lives in Showdown's items data, which this project does not vendor). Rather than hand-maintain 17
  pairings, those crystals stay inert. They are ~35% of observed Z-move use in the replay corpus, so
  this is a real gap, recorded rather than papered over.
"""

from dataclasses import replace

from battle_sim.database.loader import get_all_z_moves, normalize_id
from battle_sim.models.moves import DamageEffect, Move
from battle_sim.teams import item_showdown_name
from battle_sim.utils import Item, Type

_PLACEHOLDER_POWER = 1  # what the generic type Z-moves ship with instead of a real value

# Gen 7's fixed conversion: the base move's power decides the Z-move's, in bands.
_Z_POWER_TABLE: tuple[tuple[int, int], ...] = (
    (55, 100),
    (65, 120),
    (75, 140),
    (85, 160),
    (95, 175),
    (100, 180),
    (110, 185),
    (125, 190),
    (130, 195),
)
_MAX_Z_POWER = 200


def z_power(base_power: int) -> int:
    for threshold, power in _Z_POWER_TABLE:
        if base_power <= threshold:
            return power
    return _MAX_Z_POWER


def _damage_effect(move: Move) -> DamageEffect | None:
    return next((effect for effect in move.effects if isinstance(effect, DamageEffect)), None)


def _generic_template(item: Item) -> Move | None:
    """This crystal's Z-move, but only if it is a *generic* type crystal.

    Generic-ness is decided by the crystal's own entry carrying placeholder power, never by its type:
    signature crystals share types with generic ones (Ghostium/Decidium/Mimikium/Marshadium Z are all
    Ghost), so keying on type would hand Mimikium Z the generic Ghost Z-move.
    """
    z_move = get_all_z_moves().get(normalize_id(item_showdown_name(item)))
    if z_move is None:
        return None
    effect = _damage_effect(z_move)
    return z_move if effect is not None and effect.power == _PLACEHOLDER_POWER else None


def crystal_type(item: Item) -> Type | None:
    """The type a generic Z-Crystal upgrades, or None if it is not a generic crystal."""
    template = _generic_template(item)
    return template.type if template is not None else None


def z_move_for(item: Item, base: Move) -> Move | None:
    """The Z-move this crystal makes of this move, or None if the pairing does nothing."""
    template = _generic_template(item)
    if template is None or base.type is not template.type:
        return None
    base_effect = _damage_effect(base)
    if base_effect is None or base_effect.power is None:
        return None  # status moves get a bonus effect instead of a Z-attack: not modelled
    template_effect = _damage_effect(template)
    assert template_effect is not None  # _generic_template only returns moves that have one
    effect = replace(
        template_effect,
        power=z_power(base_effect.power),
        category=base_effect.category,
        contact=base_effect.contact,
    )
    return replace(
        template,
        category=base_effect.category,
        effects=[effect],
        accuracy_probability=None,  # Z-moves never miss
    )
