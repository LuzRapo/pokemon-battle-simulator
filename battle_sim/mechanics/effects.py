"""Wires active Pokemon to their ability/item event handlers."""

from dataclasses import dataclass, field

from battle_sim.mechanics.events import EventBus
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Ability, Item


@dataclass(eq=False, slots=True)
class EffectOwner:
    """Compared by identity (eq=False): off_owner matches with `is`, so the registry keeps each instance."""

    name: str


@dataclass(slots=True)
class EffectRegistry:
    _owners: dict[tuple[int, Ability | Item], EffectOwner] = field(default_factory=dict)

    def acquire(self, pokemon: Pokemon, source: Ability | Item) -> EffectOwner:
        key = (id(pokemon), source)
        owner = self._owners.get(key)
        if owner is None:
            owner = EffectOwner(name=f"{pokemon.nickname}:{source.name}")
            self._owners[key] = owner
        return owner

    def release_all(self, pokemon: Pokemon) -> list[EffectOwner]:
        keys = [key for key in self._owners if key[0] == id(pokemon)]
        return [self._owners.pop(key) for key in keys]

    def release(self, pokemon: Pokemon, source: Ability | Item) -> EffectOwner | None:
        return self._owners.pop((id(pokemon), source), None)

    def is_registered(self, pokemon: Pokemon, source: Ability | Item) -> bool:
        return (id(pokemon), source) in self._owners


def register_active(bus: EventBus, registry: EffectRegistry, pokemon: Pokemon) -> None:
    from battle_sim.mechanics.abilities import ABILITY_BINDERS
    from battle_sim.mechanics.items import ITEM_BINDERS

    ability_binder = ABILITY_BINDERS.get(pokemon.ability)
    if ability_binder is not None:
        ability_binder(bus, pokemon, registry.acquire(pokemon, pokemon.ability))
    item_binder = ITEM_BINDERS.get(pokemon.item)
    if item_binder is not None:
        item_binder(bus, pokemon, registry.acquire(pokemon, pokemon.item))


def unregister_active(bus: EventBus, registry: EffectRegistry, pokemon: Pokemon) -> None:
    for owner in registry.release_all(pokemon):
        bus.off_owner(owner)


def rewire_active(bus: EventBus, registry: EffectRegistry, pokemon: Pokemon) -> None:
    """Re-binds handlers after the pokemon's ability or item changed mid-battle."""
    unregister_active(bus, registry, pokemon)
    register_active(bus, registry, pokemon)


def register_ability(bus: EventBus, registry: EffectRegistry, pokemon: Pokemon) -> None:
    from battle_sim.mechanics.abilities import ABILITY_BINDERS

    binder = ABILITY_BINDERS.get(pokemon.ability)
    if binder is not None and not registry.is_registered(pokemon, pokemon.ability):
        binder(bus, pokemon, registry.acquire(pokemon, pokemon.ability))


def unregister_ability(bus: EventBus, registry: EffectRegistry, pokemon: Pokemon) -> None:
    owner = registry.release(pokemon, pokemon.ability)
    if owner is not None:
        bus.off_owner(owner)
