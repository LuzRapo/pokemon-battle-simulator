from typing import TypeAlias

from battle_sim.utils import Type

TypePair: TypeAlias = tuple[Type, Type | None]  # e.g. (Type.FIRE, Type.FLYING) or (Type.NORMAL, None)


def monotype_effectiveness(attacking_type: Type, defending_type: Type) -> float:
    """Return the effectiveness multiplier of one attacking type against one defending type."""
    return TYPE_CHART.get(attacking_type, {}).get(defending_type, 1.0)


def type_effectiveness(attacking_type: Type, defending_types: TypePair) -> float:
    """Return total type effectiveness multiplier for a move hitting a TypePair."""
    primary_type, secondary_type = defending_types

    effectiveness_multiplier = monotype_effectiveness(attacking_type, primary_type)

    if secondary_type is not None and secondary_type != primary_type:
        effectiveness_multiplier *= monotype_effectiveness(attacking_type, secondary_type)

    return effectiveness_multiplier


# Only store entries that differ from 1.0; assume 1.0 if missing.
TYPE_CHART: dict[Type, dict[Type, float]] = {
    Type.NORMAL: {
        Type.ROCK: 0.5,
        Type.STEEL: 0.5,
        Type.GHOST: 0.0,
    },
    Type.FIRE: {
        Type.GRASS: 2.0,
        Type.ICE: 2.0,
        Type.BUG: 2.0,
        Type.STEEL: 2.0,
        Type.FIRE: 0.5,
        Type.WATER: 0.5,
        Type.ROCK: 0.5,
        Type.DRAGON: 0.5,
    },
    Type.WATER: {
        Type.FIRE: 2.0,
        Type.GROUND: 2.0,
        Type.ROCK: 2.0,
        Type.WATER: 0.5,
        Type.GRASS: 0.5,
        Type.DRAGON: 0.5,
    },
    Type.ELECTRIC: {
        Type.WATER: 2.0,
        Type.FLYING: 2.0,
        Type.ELECTRIC: 0.5,
        Type.GRASS: 0.5,
        Type.DRAGON: 0.5,
        Type.GROUND: 0.0,
    },
    Type.GRASS: {
        Type.WATER: 2.0,
        Type.GROUND: 2.0,
        Type.ROCK: 2.0,
        Type.FIRE: 0.5,
        Type.GRASS: 0.5,
        Type.POISON: 0.5,
        Type.FLYING: 0.5,
        Type.BUG: 0.5,
        Type.DRAGON: 0.5,
        Type.STEEL: 0.5,
    },
    Type.ICE: {
        Type.GRASS: 2.0,
        Type.GROUND: 2.0,
        Type.FLYING: 2.0,
        Type.DRAGON: 2.0,
        Type.FIRE: 0.5,
        Type.WATER: 0.5,
        Type.ICE: 0.5,
        Type.STEEL: 0.5,
    },
    Type.FIGHTING: {
        Type.NORMAL: 2.0,
        Type.ICE: 2.0,
        Type.ROCK: 2.0,
        Type.DARK: 2.0,
        Type.STEEL: 2.0,
        Type.POISON: 0.5,
        Type.FLYING: 0.5,
        Type.PSYCHIC: 0.5,
        Type.BUG: 0.5,
        Type.FAIRY: 0.5,
        Type.GHOST: 0.0,
    },
    Type.POISON: {
        Type.GRASS: 2.0,
        Type.FAIRY: 2.0,
        Type.POISON: 0.5,
        Type.GROUND: 0.5,
        Type.ROCK: 0.5,
        Type.GHOST: 0.5,
        Type.STEEL: 0.0,
    },
    Type.GROUND: {
        Type.FIRE: 2.0,
        Type.ELECTRIC: 2.0,
        Type.POISON: 2.0,
        Type.ROCK: 2.0,
        Type.STEEL: 2.0,
        Type.GRASS: 0.5,
        Type.BUG: 0.5,
        Type.FLYING: 0.0,
    },
    Type.FLYING: {
        Type.GRASS: 2.0,
        Type.FIGHTING: 2.0,
        Type.BUG: 2.0,
        Type.ELECTRIC: 0.5,
        Type.ROCK: 0.5,
        Type.STEEL: 0.5,
    },
    Type.PSYCHIC: {
        Type.FIGHTING: 2.0,
        Type.POISON: 2.0,
        Type.PSYCHIC: 0.5,
        Type.STEEL: 0.5,
        Type.DARK: 0.0,
    },
    Type.BUG: {
        Type.GRASS: 2.0,
        Type.PSYCHIC: 2.0,
        Type.DARK: 2.0,
        Type.FIRE: 0.5,
        Type.FIGHTING: 0.5,
        Type.POISON: 0.5,
        Type.FLYING: 0.5,
        Type.GHOST: 0.5,
        Type.STEEL: 0.5,
        Type.FAIRY: 0.5,
    },
    Type.ROCK: {
        Type.FIRE: 2.0,
        Type.ICE: 2.0,
        Type.FLYING: 2.0,
        Type.BUG: 2.0,
        Type.FIGHTING: 0.5,
        Type.GROUND: 0.5,
        Type.STEEL: 0.5,
    },
    Type.GHOST: {
        Type.PSYCHIC: 2.0,
        Type.GHOST: 2.0,
        Type.DARK: 0.5,
        Type.NORMAL: 0.0,
    },
    Type.DRAGON: {
        Type.DRAGON: 2.0,
        Type.STEEL: 0.5,
        Type.FAIRY: 0.0,
    },
    Type.DARK: {
        Type.PSYCHIC: 2.0,
        Type.GHOST: 2.0,
        Type.FIGHTING: 0.5,
        Type.DARK: 0.5,
        Type.FAIRY: 0.5,
    },
    Type.STEEL: {
        Type.ICE: 2.0,
        Type.ROCK: 2.0,
        Type.FAIRY: 2.0,
        Type.FIRE: 0.5,
        Type.WATER: 0.5,
        Type.ELECTRIC: 0.5,
        Type.STEEL: 0.5,
    },
    Type.FAIRY: {
        Type.FIGHTING: 2.0,
        Type.DRAGON: 2.0,
        Type.DARK: 2.0,
        Type.FIRE: 0.5,
        Type.POISON: 0.5,
        Type.STEEL: 0.5,
    },
}
