from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Any, Protocol

from battle_sim.maths.rng import RNG
from battle_sim.mechanics.log import BattleLog
from battle_sim.models.actions import Action
from battle_sim.models.moves import Move
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Target

if TYPE_CHECKING:
    from battle_sim.mechanics.battle import BattleState

Payload = dict[str, Any]
"""Mutable per-emit scratch space merged from handler results.

Keys the emitting engine code GUARANTEES for an event are read with ``payload[...]``
so a protocol breach crashes at the read. Keys that handlers MAY contribute are
read with ``payload.get(key, neutral_default)``.

Guaranteed by the engine per event:
- ``move_type: Type`` / ``category: Category`` — ``ON_DAMAGE_CALC``, ``ON_BEFORE_MOVE``,
  ``ON_BEFORE_HIT``, ``ON_AFTER_HIT``.
- ``damage: int`` — the computed hit damage, ``ON_BEFORE_HIT`` (handlers may rewrite it).
- ``dealt: int`` / ``contact: bool`` — ``ON_AFTER_HIT``.
- ``total_dealt: int`` — ``ON_ACTION_RESOLVE``.
- ``side_index`` / ``attacker_index`` / ``defender_index: int`` — per event as applicable.

Optional handler contributions:
- ``attack_mods_4096`` / ``defense_mods_4096`` / ``power_mods_4096`` /
  ``pre_screen_mods_4096`` / ``final_mods_4096: list[int]`` — 4096-based
  multipliers folded by the damage formula at fixed positions.
- ``stab_4096: int`` — overrides the STAB constant (Adaptability).
- ``ignore_burn: bool`` — Guts.
- ``ignore_attack_stages`` / ``ignore_defense_stages`` / ``bypass_screens: bool`` — Unaware, Infiltrator.
- ``absorbed: bool`` / ``immune_reason: str`` — set alongside ``cancel`` on ``ON_BEFORE_MOVE``.
- ``magic_guard: bool`` — set at SYSTEM priority so chip/recoil handlers skip.
- ``hazards_blocked: bool`` — set alongside ``cancel`` on ``ON_ENTRY_HAZARD``.
"""


class Event(Enum):
    """Every member is either emitted by the engine or exercised in tests — no aspirational hooks."""

    ON_TURN_START = auto()
    ON_TURN_END = auto()

    ON_BEFORE_ACTION = auto()
    ON_ACTION_RESOLVE = auto()
    ON_AFTER_ACTION = auto()

    ON_SWITCH_IN = auto()
    ON_SWITCH_OUT = auto()  # the outgoing pokemon is still wired; not emitted for fainted switches

    ON_DAMAGE_CALC = auto()
    ON_BEFORE_MOVE = auto()  # once per damaging move, before application — absorption/immunity cancels
    ON_BEFORE_HIT = auto()  # once per hit, carries "damage" — survival clamps rewrite it
    ON_AFTER_HIT = auto()

    ON_ENTRY_HAZARD = auto()
    ON_RESIDUAL = auto()

    ON_FAINT = auto()


class EventPriority(IntEnum):
    SYSTEM = 10_000
    FIELD = 8_000
    SIDE = 6_000
    POKEMON_VOLATILE = 4_000
    ABILITY = 2_000
    ITEM = 1_000
    MOVE = 500
    DEFAULT = 0


class ResidualOrder(IntEnum):
    """ON_RESIDUAL priorities mirroring Showdown's onResidualOrder (higher fires earlier here)."""

    MAGIC_GUARD = 10_000  # the suppression flag must precede every chip
    PARADOX = 9_500  # Protosynthesis/Quark Drive re-evaluate after the field-duration tick
    WEATHER = 9_000  # sandstorm chip (PS: weather upkeep)
    WEATHER_ABILITY = 8_500  # Ice Body / Dry Skin weather heal+chip
    TERRAIN = 8_400  # grassy terrain recovery
    CURE = 8_200  # Hydration (PS order 5.1)
    ITEM_RECOVERY = 8_000  # Leftovers / Black Sludge (PS order 5)
    LEECH_SEED = 7_000  # PS order 8
    POISON_HEAL = 6_500  # must flag before the status chip
    STATUS = 6_000  # poison/toxic/burn (PS orders 9-10)
    NIGHTMARE = 5_000  # PS order 11
    PARTIAL_TRAP = 4_900  # PS order 13: Wrap and friends chip before Salt Cure's band
    SALT_CURE = 4_800  # PS order 13
    CURSE = 4_700  # PS order 12 (close enough; distinct band keeps ordering deterministic)
    BAD_DREAMS = 4_500  # PS order 11 (after Nightmare)
    YAWN = 3_000  # PS order 19
    LOCKED_MOVE = 2_000  # rampage countdown; fatigue confusion on natural end
    ORB = 1_500  # Toxic/Flame Orb self-infliction (PS order 26.2)
    SPEED_BOOST = 1_000  # PS order 27


@dataclass
class EventContext:
    """battle and log are always present; the rest varies by event (e.g. no actor on ON_TURN_START)."""

    rng: RNG
    battle: "BattleState"
    log: BattleLog
    actor: Pokemon | None = None
    defender: Pokemon | None = None
    target: Target | None = None
    action: Action | None = None
    move: Move | None = None


@dataclass(slots=True)
class HandlerResult:
    cancel: bool = False
    updated_payload: Payload = field(default_factory=dict)


class EventHandler(Protocol):
    def __call__(self, context: EventContext, payload: Payload) -> HandlerResult | None: ...


class EventOwner(Protocol):
    """An identity token for grouped teardown via `off_owner`; `name` exists for debugging."""

    name: str


@dataclass(slots=True)
class Subscription:
    event: Event
    handler: EventHandler
    priority: int = EventPriority.DEFAULT
    owner: EventOwner | None = None
    once: bool = False


class EventBus:
    __slots__ = ("_subscriptions", "_dedupe_keys")

    def __init__(self) -> None:
        self._subscriptions: dict[Event, list[Subscription]] = {}
        self._dedupe_keys: set[tuple[int, int, int, int, bool]] = set()

    def on(
        self,
        event: Event,
        handler: EventHandler,
        priority: int = EventPriority.DEFAULT,
        owner: EventOwner | None = None,
        once: bool = False,
    ) -> Subscription:
        key = (event.value, id(handler), id(owner) if owner is not None else 0, int(priority), once)
        if key in self._dedupe_keys:
            return Subscription(event=event, handler=handler, priority=int(priority), owner=owner, once=once)

        sub = Subscription(event=event, handler=handler, priority=int(priority), owner=owner, once=once)
        bucket = self._subscriptions.setdefault(event, [])
        bucket.append(sub)
        self._dedupe_keys.add(key)
        bucket.sort(key=lambda s: -s.priority)  # stable: equal priorities keep registration order
        return sub

    def subscribed_events(self) -> list["Event"]:
        """Every event something is currently listening on, in the enum's own order.

        Exists for introspection rather than for resolving a turn: an ability or an item is only
        described by the handlers its binder registers, so this is how `/what_is` answers "when does
        this actually do something" without anybody maintaining a second description of it.
        """
        return [event for event in Event if self._subscriptions.get(event)]

    def off(self, subscription: Subscription) -> None:
        bucket = self._subscriptions.get(subscription.event, [])
        if subscription in bucket:
            bucket.remove(subscription)
            key = (
                subscription.event.value,
                id(subscription.handler),
                id(subscription.owner) if subscription.owner is not None else 0,
                int(subscription.priority),
                subscription.once,
            )
            if key in self._dedupe_keys:
                self._dedupe_keys.remove(key)

    def off_owner(self, owner: EventOwner) -> None:
        for event, bucket in self._subscriptions.items():
            remaining: list[Subscription] = []
            for sub in bucket:
                if sub.owner is owner:
                    key = (event.value, id(sub.handler), id(owner), int(sub.priority), sub.once)
                    if key in self._dedupe_keys:
                        self._dedupe_keys.remove(key)
                    continue
                remaining.append(sub)
            self._subscriptions[event] = remaining

    def emit(self, event: Event, context: EventContext, payload: Payload | None = None) -> Payload:
        current: Payload = dict(payload or {})
        to_remove_once: list[Subscription] = []

        for sub in list(self._subscriptions.get(event, ())):
            result = sub.handler(context, current)

            if sub.once:
                to_remove_once.append(sub)

            if result is None:
                continue

            if result.updated_payload:
                current.update(result.updated_payload)

            if result.cancel:
                break

        for sub in to_remove_once:
            self.off(sub)

        return current
