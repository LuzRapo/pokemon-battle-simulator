"""One binder per held item, registered via @item; wired/unwired on switch-in/out by mechanics.effects."""

from collections.abc import Callable

from battle_sim.mechanics.effects import EffectOwner
from battle_sim.mechanics.events import (
    Event,
    EventBus,
    EventContext,
    EventPriority,
    HandlerResult,
    Payload,
    ResidualOrder,
)
from battle_sim.mechanics.log import BattleLog
from battle_sim.mechanics.stages import apply_stage_changes
from battle_sim.models.log_events import (
    AirBalloonPopped,
    AirBalloonRevealed,
    BerryWeakened,
    FloatedOnAirBalloon,
    ItemChipDamage,
    ItemHealed,
    SelfSwitchPending,
    SurvivedAtOneHp,
)
from battle_sim.models.moves import DamageEffect
from battle_sim.models.pokemon import Pokemon
from battle_sim.models.type_matchups import type_effectiveness
from battle_sim.utils import Category, Item, Stats, Status, Terrain, Type, Weather

type ItemBinder = Callable[[EventBus, Pokemon, EffectOwner], None]

ITEM_BINDERS: dict[Item, ItemBinder] = {}


def item(kind: Item) -> Callable[[ItemBinder], ItemBinder]:
    def register(binder: ItemBinder) -> ItemBinder:
        ITEM_BINDERS[kind] = binder
        return binder

    return register


MONOCLE_MOD_4096 = 6144  # 1.5x
MONOCLE_POWER_CAP = 60  # only the weak moves, exactly as Technician reads it


@item(Item.MEOWFREDS_MONOCLE)
def _bind_meowfreds_monocle(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """Technician in an item: anything at 60 base power or less hits half again as hard.

    Deliberately not from the games; neither is he. It replaced a flat 1.25x on everything, which
    made a strong attacker stronger and had nothing to say about what he should be attacking with.
    This rewards the weak, spammable moves he actually carries.
    """

    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.MEOWFREDS_MONOCLE or context.move is None:
            return None
        damage = next((e for e in context.move.effects if isinstance(e, DamageEffect)), None)
        if damage is not None and damage.power is not None and damage.power <= MONOCLE_POWER_CAP:
            payload.setdefault("power_mods_4096", []).append(MONOCLE_MOD_4096)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)


@item(Item.LIFE_ORB)
def _bind_life_orb(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon and pokemon.item is Item.LIFE_ORB:
            payload.setdefault("final_mods_4096", []).append(5324)
        return None

    def recoil(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.LIFE_ORB:
            return None
        if payload["total_dealt"] <= 0 or pokemon.is_fainted():
            return None
        if payload.get("magic_guard", False):
            return None
        chip = max(1, pokemon.stat_totals.HP // 10)
        pokemon.apply_damage(chip)
        context.log.add(
            ItemChipDamage(side=payload["attacker_index"], pokemon=pokemon.nickname, item=Item.LIFE_ORB, amount=chip)
        )
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)
    bus.on(Event.ON_ACTION_RESOLVE, recoil, priority=EventPriority.ITEM, owner=owner)


def _bind_choice_attack_item(kind: Item, category: Category) -> ItemBinder:
    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon and pokemon.item is kind and payload["category"] is category:
                payload.setdefault("final_mods_4096", []).append(6144)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)

    return binder


ITEM_BINDERS[Item.CHOICE_BAND] = _bind_choice_attack_item(Item.CHOICE_BAND, Category.PHYSICAL)
ITEM_BINDERS[Item.CHOICE_SPECS] = _bind_choice_attack_item(Item.CHOICE_SPECS, Category.SPECIAL)


@item(Item.EXPERT_BELT)
def _bind_expert_belt(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_super_effective(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.EXPERT_BELT:
            return None
        defender = context.defender
        assert defender is not None  # damage events always carry the defender
        if type_effectiveness(payload["move_type"], defender.types) >= 2:
            payload.setdefault("final_mods_4096", []).append(4915)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_super_effective, priority=EventPriority.ITEM, owner=owner)


@item(Item.ASSAULT_VEST)
def _bind_assault_vest(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def bulk_special(context: EventContext, payload: Payload) -> HandlerResult | None:
        if (
            context.defender is pokemon
            and pokemon.item is Item.ASSAULT_VEST
            and payload["category"] is Category.SPECIAL
        ):
            payload.setdefault("defense_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, bulk_special, priority=EventPriority.ITEM, owner=owner)


@item(Item.HEAVY_DUTY_BOOTS)
def _bind_heavy_duty_boots(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def block_hazards(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.HEAVY_DUTY_BOOTS:
            return None
        return HandlerResult(cancel=True, updated_payload={"hazards_blocked": True})

    bus.on(Event.ON_ENTRY_HAZARD, block_hazards, priority=EventPriority.ITEM, owner=owner)


@item(Item.FOCUS_SASH)
def _bind_focus_sash(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def hang_on(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.FOCUS_SASH:
            return None
        max_hp = pokemon.stat_totals.HP
        if max_hp == pokemon.live_stats.HP and payload["damage"] >= max_hp:
            pokemon.consume_item()
            context.log.add(
                SurvivedAtOneHp(side=payload["defender_index"], pokemon=pokemon.nickname, cause="focus_sash")
            )
            return HandlerResult(updated_payload={"damage": max_hp - 1})
        return None

    # Historically the sash is consumed in preference to Sturdy, so it outranks the ABILITY band here.
    bus.on(Event.ON_BEFORE_HIT, hang_on, priority=EventPriority.ABILITY + 100, owner=owner)


@item(Item.ROCKY_HELMET)
def _bind_rocky_helmet(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def spike_back(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.ROCKY_HELMET:
            return None
        if not payload["contact"]:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted():
            return None
        recoil = max(1, attacker.stat_totals.HP // 6)
        attacker.apply_damage(recoil)
        context.log.add(
            ItemChipDamage(
                side=payload["attacker_index"], pokemon=attacker.nickname, item=Item.ROCKY_HELMET, amount=recoil
            )
        )
        return None

    bus.on(Event.ON_AFTER_HIT, spike_back, priority=EventPriority.ITEM, owner=owner)


@item(Item.AIR_BALLOON)
def _bind_air_balloon(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """The balloon does two things: it keeps its holder off the ground, and it bursts when hit.

    The first used to live only inside `calculate_damage`, as a bare `return 0` — which meant the
    engine announced "It's super effective!" for a 4x Earthquake and then silently did nothing at
    all, because effectiveness is logged before damage is worked out and a zero deals no blow worth
    reporting. A trainer watched their Garchomp lose a won match to what looked like a no-op.

    So it is cancelled here instead, before a move is applied, exactly as Levitate cancels one. The
    check in `calculate_damage` stays where it is: `analysis.damage_range` never runs the event bus,
    so without it the AI would still believe Earthquake hurts a Pokemon holding a balloon.
    """

    def announce(context: EventContext, payload: Payload) -> HandlerResult | None:
        # The games say so on the way in, and that is the whole of how an opponent learns not to
        # reach for a Ground move. Silent, the immunity is not a counter — it is a hidden trapdoor.
        if pokemon.item is Item.AIR_BALLOON:
            context.log.add(AirBalloonRevealed(side=payload.get("side_index", 0), pokemon=pokemon.nickname))
        return None

    def float_over(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.AIR_BALLOON:
            return None
        if payload["move_type"] is not Type.GROUND:
            return None
        context.log.add(FloatedOnAirBalloon(side=payload["defender_index"], pokemon=pokemon.nickname))
        return HandlerResult(cancel=True, updated_payload={"absorbed": True, "immune_reason": "air_balloon"})

    # The last HP this Pokemon was seen at, so any drop at all can be noticed however it happened.
    # Watching the total rather than listening for one kind of damage is what makes this cover the
    # awkward ones — entry hazards land after `ON_ENTRY_HAZARD` has already fired, and recoil, Life
    # Orb, Rough Skin and confusion all arrive by their own routes.
    seen = [pokemon.live_stats.HP]

    def pop(context: EventContext, payload: Payload) -> HandlerResult | None:
        """Burst the balloon the moment its holder has lost any HP at all.

        A house rule, and a deliberate one: the games burst a balloon only when its holder is *hit
        by an attack*, leaving it whole through Stealth Rock, poison and sandstorm alike. Sam asked
        for the broader version, and match bcf1ceb4 is why — an Aggron walked into Stealth Rock, took
        nine, and still floated over a 4x Earthquake that would have won the match.
        """
        if pokemon.item is not Item.AIR_BALLOON:
            seen[0] = pokemon.live_stats.HP
            return None
        now = pokemon.live_stats.HP
        if now < seen[0]:
            pokemon.consume_item()
            context.log.add(AirBalloonPopped(side=payload.get("defender_index", 0), pokemon=pokemon.nickname))
        seen[0] = now
        return None

    bus.on(Event.ON_SWITCH_IN, announce, priority=EventPriority.ITEM, owner=owner)
    bus.on(Event.ON_BEFORE_MOVE, float_over, priority=EventPriority.ITEM, owner=owner)
    # Every point a turn can pause at after HP has moved: a hit lands immediately, and the rest —
    # hazards on the way in, residual chip, recoil — are swept up at the next of these to come round.
    for moment in (Event.ON_AFTER_HIT, Event.ON_ACTION_RESOLVE, Event.ON_RESIDUAL, Event.ON_TURN_END):
        bus.on(moment, pop, priority=EventPriority.ITEM, owner=owner)


@item(Item.LEFTOVERS)
def _bind_leftovers(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def recover(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.LEFTOVERS or pokemon.is_fainted():
            return None
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 16))
        if healed > 0:
            context.log.add(ItemHealed(side=payload["side_index"], pokemon=pokemon.nickname, item=Item.LEFTOVERS))
        return None

    bus.on(Event.ON_RESIDUAL, recover, priority=ResidualOrder.ITEM_RECOVERY, owner=owner)


@item(Item.BLACK_SLUDGE)
def _bind_black_sludge(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def sludge(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is not pokemon or pokemon.item is not Item.BLACK_SLUDGE or pokemon.is_fainted():
            return None
        side_index = payload["side_index"]
        if Type.POISON in pokemon.types:
            healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 16))
            if healed > 0:
                context.log.add(ItemHealed(side=side_index, pokemon=pokemon.nickname, item=Item.BLACK_SLUDGE))
        else:
            dealt = pokemon.apply_damage(max(1, pokemon.stat_totals.HP // 16))
            context.log.add(
                ItemChipDamage(side=side_index, pokemon=pokemon.nickname, item=Item.BLACK_SLUDGE, amount=dealt)
            )
        return None

    bus.on(Event.ON_RESIDUAL, sludge, priority=ResidualOrder.ITEM_RECOVERY, owner=owner)


# Consulted at engine core points rather than via handlers:
# - WEATHER_ROCKS / Light Clay / Terrain Extender extend durations where weather/terrain/screens are set
# - Loaded Dice alters the multi-hit roll in damage_apply._planned_hits
# - Wide Lens / Scope Lens fold into the accuracy and crit checks
# - Covert Cloak blocks secondaries in the move-effect loop
# - Clear Amulet / White Herb live in mechanics.stages
# - Lum/Chesto Berry cure on infliction in status_apply
# - Booster Energy is consumed by the paradox ability handlers
# - Rusted Sword/Shield are forme items: the Crowned formes in species data carry the effect

WEATHER_ROCKS: dict[Weather, Item] = {
    Weather.RAIN: Item.DAMP_ROCK,
    Weather.SUN: Item.HEAT_ROCK,
    Weather.SANDSTORM: Item.SMOOTH_ROCK,
    Weather.SNOW: Item.ICY_ROCK,
}

_SEED_BY_TERRAIN: dict[Terrain, tuple[Item, Stats]] = {
    Terrain.GRASSY: (Item.GRASSY_SEED, Stats.DEFENCE),
    Terrain.ELECTRIC: (Item.ELECTRIC_SEED, Stats.DEFENCE),
    Terrain.PSYCHIC: (Item.PSYCHIC_SEED, Stats.SP_DEFENCE),
    Terrain.MISTY: (Item.MISTY_SEED, Stats.SP_DEFENCE),
}


def consume_terrain_seed(pokemon: Pokemon, side_index: int, terrain: Terrain, log: BattleLog) -> None:
    """+1 Def/SpD and the seed is spent the moment its terrain is up (switch-in or terrain start)."""
    entry = _SEED_BY_TERRAIN.get(terrain)
    if entry is None:
        return
    seed, stat = entry
    if pokemon.item is not seed or pokemon.is_fainted():
        return
    pokemon.consume_item()
    apply_stage_changes(pokemon, side_index, {stat: 1}, log, inflicted_by_opponent=False, source="seed")


def _bind_terrain_seed(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def sprout(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.actor is pokemon:
            consume_terrain_seed(pokemon, payload["side_index"], context.battle.field.terrain, context.log)
        return None

    bus.on(Event.ON_SWITCH_IN, sprout, priority=EventPriority.ITEM, owner=owner)


ITEM_BINDERS[Item.GRASSY_SEED] = _bind_terrain_seed
ITEM_BINDERS[Item.ELECTRIC_SEED] = _bind_terrain_seed
ITEM_BINDERS[Item.PSYCHIC_SEED] = _bind_terrain_seed
ITEM_BINDERS[Item.MISTY_SEED] = _bind_terrain_seed


def _bind_type_boost(kind: Item, move_type: Type) -> ItemBinder:
    """Plates, Black Glasses, etc.: 1.2x damage on the matching type."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon and pokemon.item is kind and payload["move_type"] is move_type:
                payload.setdefault("final_mods_4096", []).append(4915)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)

    return binder


ITEM_BINDERS[Item.IRON_PLATE] = _bind_type_boost(Item.IRON_PLATE, Type.STEEL)
ITEM_BINDERS[Item.BLACK_GLASSES] = _bind_type_boost(Item.BLACK_GLASSES, Type.DARK)
ITEM_BINDERS[Item.MYSTIC_WATER] = _bind_type_boost(Item.MYSTIC_WATER, Type.WATER)
ITEM_BINDERS[Item.METAL_COAT] = _bind_type_boost(Item.METAL_COAT, Type.STEEL)
ITEM_BINDERS[Item.NEVER_MELT_ICE] = _bind_type_boost(Item.NEVER_MELT_ICE, Type.ICE)
ITEM_BINDERS[Item.SILK_SCARF] = _bind_type_boost(Item.SILK_SCARF, Type.NORMAL)
ITEM_BINDERS[Item.EARTH_PLATE] = _bind_type_boost(Item.EARTH_PLATE, Type.GROUND)
ITEM_BINDERS[Item.SPOOKY_PLATE] = _bind_type_boost(Item.SPOOKY_PLATE, Type.GHOST)
ITEM_BINDERS[Item.PIXIE_PLATE] = _bind_type_boost(Item.PIXIE_PLATE, Type.FAIRY)


def _bind_legend_orb(kind: Item, bearers: frozenset[str], boosted: frozenset[Type]) -> ItemBinder:
    """Soul Dew / Griseous Core: 1.2x on two types, only for the legendary line that owns it."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon or pokemon.item is not kind or pokemon.name not in bearers:
                return None
            if payload["move_type"] in boosted:
                payload.setdefault("final_mods_4096", []).append(4915)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)

    return binder


ITEM_BINDERS[Item.SOUL_DEW] = _bind_legend_orb(
    Item.SOUL_DEW, frozenset({"Latios", "Latias"}), frozenset({Type.PSYCHIC, Type.DRAGON})
)
ITEM_BINDERS[Item.GRISEOUS_CORE] = _bind_legend_orb(
    Item.GRISEOUS_CORE, frozenset({"Giratina", "Giratina-Origin"}), frozenset({Type.GHOST, Type.DRAGON})
)
# Gen 4-7 called it the Griseous Orb, which is the name every Gen 7 set is written with; identical item.
ITEM_BINDERS[Item.GRISEOUS_ORB] = _bind_legend_orb(
    Item.GRISEOUS_ORB, frozenset({"Giratina", "Giratina-Origin"}), frozenset({Type.GHOST, Type.DRAGON})
)


def _bind_ogerpon_mask(kind: Item, bearer: str) -> ItemBinder:
    """The Ogerpon masks: 1.2x power on every move of the matching forme."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def boost(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is pokemon and pokemon.item is kind and pokemon.name == bearer:
                payload.setdefault("power_mods_4096", []).append(4915)
            return None

        bus.on(Event.ON_DAMAGE_CALC, boost, priority=EventPriority.ITEM, owner=owner)

    return binder


ITEM_BINDERS[Item.WELLSPRING_MASK] = _bind_ogerpon_mask(Item.WELLSPRING_MASK, "Ogerpon-Wellspring")
ITEM_BINDERS[Item.CORNERSTONE_MASK] = _bind_ogerpon_mask(Item.CORNERSTONE_MASK, "Ogerpon-Cornerstone")


@item(Item.EVIOLITE)
def _bind_eviolite(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def bolster(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is pokemon and pokemon.item is Item.EVIOLITE and not pokemon.fully_evolved:
            payload.setdefault("defense_mods_4096", []).append(6144)
        return None

    bus.on(Event.ON_DAMAGE_CALC, bolster, priority=EventPriority.ITEM, owner=owner)


def _bind_resist_berry(kind: Item, weakened: Type) -> ItemBinder:
    """Chople/Shuca/Colbur: halve one super-effective hit, then the berry is spent."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def weaken(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.defender is not pokemon or pokemon.item is not kind:
                return None
            if payload["move_type"] is not weakened or payload["behind_substitute"]:
                return None
            if type_effectiveness(weakened, pokemon.types) < 2:
                return None
            pokemon.consume_item()
            payload.setdefault("final_mods_4096", []).append(2048)
            context.log.add(BerryWeakened(side=payload["defender_index"], pokemon=pokemon.nickname, item=kind))
            return None

        bus.on(Event.ON_DAMAGE_CALC, weaken, priority=EventPriority.ITEM, owner=owner)

    return binder


ITEM_BINDERS[Item.CHOPLE_BERRY] = _bind_resist_berry(Item.CHOPLE_BERRY, Type.FIGHTING)
ITEM_BINDERS[Item.SHUCA_BERRY] = _bind_resist_berry(Item.SHUCA_BERRY, Type.GROUND)
ITEM_BINDERS[Item.COLBUR_BERRY] = _bind_resist_berry(Item.COLBUR_BERRY, Type.DARK)


@item(Item.SITRUS_BERRY)
def _bind_sitrus_berry(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def ripen(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.SITRUS_BERRY or pokemon.is_fainted():
            return None
        if 2 * pokemon.live_stats.HP > pokemon.stat_totals.HP:
            return None
        pokemon.consume_item()
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 4))
        if healed > 0:
            context.log.add(
                ItemHealed(side=payload["defender_index"], pokemon=pokemon.nickname, item=Item.SITRUS_BERRY)
            )
        return None

    bus.on(Event.ON_AFTER_HIT, ripen, priority=EventPriority.ITEM, owner=owner)


@item(Item.WIKI_BERRY)
def _bind_wiki_berry(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    """A pinch berry: a third of its holder's health back, once, at a quarter or less.

    Sitrus is the same shape with different numbers — half health, a quarter healed. The confusion a
    disliked nature brings is not modelled; it only fires on a Sp. Atk-lowering nature and would
    otherwise mean carrying flavour data for one berry.
    """

    def ripen(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.WIKI_BERRY or pokemon.is_fainted():
            return None
        if 4 * pokemon.live_stats.HP > pokemon.stat_totals.HP:
            return None
        pokemon.consume_item()
        healed = pokemon.apply_healing(max(1, pokemon.stat_totals.HP // 3))
        if healed > 0:
            context.log.add(ItemHealed(side=payload["defender_index"], pokemon=pokemon.nickname, item=Item.WIKI_BERRY))
        return None

    bus.on(Event.ON_AFTER_HIT, ripen, priority=EventPriority.ITEM, owner=owner)


@item(Item.WEAKNESS_POLICY)
def _bind_weakness_policy(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def trigger(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.WEAKNESS_POLICY or pokemon.is_fainted():
            return None
        if payload["dealt"] <= 0 or type_effectiveness(payload["move_type"], pokemon.types) < 2:
            return None
        pokemon.consume_item()
        apply_stage_changes(
            pokemon,
            payload["defender_index"],
            {Stats.ATTACK: 2, Stats.SP_ATTACK: 2},
            context.log,
            inflicted_by_opponent=False,
            source="weakness_policy",
        )
        return None

    bus.on(Event.ON_AFTER_HIT, trigger, priority=EventPriority.ITEM, owner=owner)


def _bind_status_orb(kind: Item, status: Status) -> ItemBinder:
    """Toxic Orb / Flame Orb: self-inflicts at the end of the turn."""

    def binder(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
        def afflict(context: EventContext, payload: Payload) -> HandlerResult | None:
            if context.actor is not pokemon or pokemon.item is not kind or pokemon.is_fainted():
                return None
            if pokemon.status is not Status.NONE:
                return None
            from battle_sim.engine.status_apply import _apply_main_status  # lazy: avoids a mechanics->engine cycle

            teams = (context.battle.sides[0].team, context.battle.sides[1].team)
            _apply_main_status(status, pokemon, payload["side_index"], context.rng, context.log, teams)
            return None

        bus.on(Event.ON_RESIDUAL, afflict, priority=ResidualOrder.ORB, owner=owner)

    return binder


ITEM_BINDERS[Item.TOXIC_ORB] = _bind_status_orb(Item.TOXIC_ORB, Status.TOXIC)
ITEM_BINDERS[Item.FLAME_ORB] = _bind_status_orb(Item.FLAME_ORB, Status.BURN)


ITEM_BINDERS[Item.WISE_GLASSES] = _bind_choice_attack_item(Item.WISE_GLASSES, Category.SPECIAL)
ITEM_BINDERS[Item.MUSCLE_BAND] = _bind_choice_attack_item(Item.MUSCLE_BAND, Category.PHYSICAL)
ITEM_BINDERS[Item.MIRACLE_SEED] = _bind_type_boost(Item.MIRACLE_SEED, Type.GRASS)
ITEM_BINDERS[Item.SILVER_POWDER] = _bind_type_boost(Item.SILVER_POWDER, Type.BUG)
ITEM_BINDERS[Item.SPLASH_PLATE] = _bind_type_boost(Item.SPLASH_PLATE, Type.WATER)
ITEM_BINDERS[Item.STONE_PLATE] = _bind_type_boost(Item.STONE_PLATE, Type.ROCK)
ITEM_BINDERS[Item.ADAMANT_CRYSTAL] = _bind_legend_orb(
    Item.ADAMANT_CRYSTAL, frozenset({"Dialga", "Dialga-Origin"}), frozenset({Type.STEEL, Type.DRAGON})
)
ITEM_BINDERS[Item.LUSTROUS_GLOBE] = _bind_legend_orb(
    Item.LUSTROUS_GLOBE, frozenset({"Palkia", "Palkia-Origin"}), frozenset({Type.WATER, Type.DRAGON})
)


@item(Item.EJECT_BUTTON)
def _bind_eject_button(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def eject(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.EJECT_BUTTON or pokemon.is_fainted():
            return None
        if payload["dealt"] <= 0:
            return None
        side_index = payload["defender_index"]
        side = context.battle.sides[side_index]
        if not any(i != side.active[0] and not p.is_fainted() for i, p in enumerate(side.team)):
            return None
        pokemon.consume_item()
        side.needs_switch = True
        context.log.add(SelfSwitchPending(side=side_index, pokemon=pokemon.nickname))
        return None

    bus.on(Event.ON_AFTER_HIT, eject, priority=EventPriority.ITEM, owner=owner)


@item(Item.RED_CARD)
def _bind_red_card(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def show_the_card(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.RED_CARD or pokemon.is_fainted():
            return None
        if payload["dealt"] <= 0:
            return None
        attacker = context.actor
        assert attacker is not None  # hit events always carry the attacker
        if attacker.is_fainted():
            return None
        pokemon.consume_item()
        from battle_sim.engine.switching import _force_random_switch  # lazy: avoids a mechanics->engine cycle

        _force_random_switch(context.battle, payload["attacker_index"], context.log)
        return None

    bus.on(Event.ON_AFTER_HIT, show_the_card, priority=EventPriority.ITEM, owner=owner)


@item(Item.STARF_BERRY)
def _bind_starf_berry(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    _STARF_STATS = (Stats.ATTACK, Stats.DEFENCE, Stats.SP_ATTACK, Stats.SP_DEFENCE, Stats.SPEED)

    def panic_boost(context: EventContext, payload: Payload) -> HandlerResult | None:
        if context.defender is not pokemon or pokemon.item is not Item.STARF_BERRY or pokemon.is_fainted():
            return None
        if 4 * pokemon.live_stats.HP > pokemon.stat_totals.HP:
            return None
        pokemon.consume_item()
        stat = _STARF_STATS[context.rng.random_integer(0, len(_STARF_STATS))]
        apply_stage_changes(
            pokemon, payload["defender_index"], {stat: 2}, context.log, inflicted_by_opponent=False, source="seed"
        )
        return None

    bus.on(Event.ON_AFTER_HIT, panic_boost, priority=EventPriority.ITEM, owner=owner)


@item(Item.PUNCHING_GLOVE)
def _bind_punching_glove(bus: EventBus, pokemon: Pokemon, owner: EffectOwner) -> None:
    def boost_punches(context: EventContext, payload: Payload) -> HandlerResult | None:
        move = context.move
        if context.actor is pokemon and move is not None and move.punching:
            payload.setdefault("power_mods_4096", []).append(4506)
        return None

    bus.on(Event.ON_DAMAGE_CALC, boost_punches, priority=EventPriority.ITEM, owner=owner)


# -- What a damage estimate has to know without running the bus ----------------------
# Every multiplier above is applied from an ON_DAMAGE_CALC handler, which `analysis.damage_range`
# never runs — so the estimator priced a Choice Band hit exactly like a bare one, and an Eviolite
# wall exactly like a naked NFE. Measured against the engine over 2,000 random hits, an Eviolite
# defender took 0.73x what was predicted and a Choice Band attacker dealt 1.25x. These are the most
# common items in the game, so that error was in a large share of every decision the search made.
#
# Mirrored here rather than in the estimator so the numbers sit beside the handlers they copy, the
# same arrangement as `ability_absorbs`. 6144/4096 = 1.5, 5324 = 1.3, 4915 = 1.2.
_CHOICE_BOOSTS: dict[Item, Category] = {Item.CHOICE_BAND: Category.PHYSICAL, Item.CHOICE_SPECS: Category.SPECIAL}
# The legendary-line orbs, mirrored from the `_bind_legend_orb` registrations below: 1.2x on two
# types, and only for the line that owns the orb.
_LEGEND_ORBS: dict[Item, tuple[frozenset[str], frozenset[Type]]] = {
    Item.SOUL_DEW: (frozenset({"Latios", "Latias"}), frozenset({Type.PSYCHIC, Type.DRAGON})),
    Item.GRISEOUS_CORE: (frozenset({"Giratina", "Giratina-Origin"}), frozenset({Type.GHOST, Type.DRAGON})),
    Item.GRISEOUS_ORB: (frozenset({"Giratina", "Giratina-Origin"}), frozenset({Type.GHOST, Type.DRAGON})),
}


def _monocle_power_mods(attacker: Pokemon, base_power: int | None) -> list[int]:
    """The Monocle's Technician boost, for the estimator. Kept beside the binder it mirrors."""
    if attacker.item is not Item.MEOWFREDS_MONOCLE or base_power is None or base_power > MONOCLE_POWER_CAP:
        return []
    return [MONOCLE_MOD_4096]


def static_damage_modifiers(
    move_type: Type, category: Category, attacker: Pokemon, defender: Pokemon, base_power: int | None = None
) -> dict[str, list[int]]:
    """The item multipliers a damage estimate can work out on its own, keyed as the payload wants.

    Only what is decidable from the two Pokemon and the move: no berries (they fire once and are
    spent), nothing that depends on the order of a turn that has not happened yet.
    """
    final: list[int] = []
    defense: list[int] = []
    power: list[int] = _monocle_power_mods(attacker, base_power)
    if _CHOICE_BOOSTS.get(attacker.item) is category:
        final.append(6144)
    if attacker.item is Item.LIFE_ORB:
        final.append(5324)

    if attacker.item is Item.EXPERT_BELT and type_effectiveness(move_type, defender.types) >= 2:
        final.append(4915)
    orb = _LEGEND_ORBS.get(attacker.item)
    if orb is not None and attacker.name in orb[0] and move_type in orb[1]:
        final.append(4915)
    if defender.item is Item.EVIOLITE and not defender.fully_evolved:
        defense.append(6144)
    if defender.item is Item.ASSAULT_VEST and category is Category.SPECIAL:
        defense.append(6144)
    modifiers: dict[str, list[int]] = {}
    if final:
        modifiers["final_mods_4096"] = final
    if defense:
        modifiers["defense_mods_4096"] = defense
    if power:
        modifiers["power_mods_4096"] = power
    return modifiers
