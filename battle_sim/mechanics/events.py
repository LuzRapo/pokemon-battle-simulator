from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Protocol

from battle_sim.maths.rng import RNG
from battle_sim.models.actions import Action
from battle_sim.models.pokemon import Pokemon
from battle_sim.utils import Target

Payload = dict[str, Any]


class Event(Enum):
    ON_TURN_START = auto()
    ON_TURN_END = auto()

    ON_BEFORE_ACTION = auto()
    ON_ACTION_START = auto()
    ON_ACTION_RESOLVE = auto()
    ON_AFTER_ACTION = auto()

    ON_SWITCH_DECLARED = auto()
    ON_SWITCH_OUT = auto()
    ON_SWITCH_IN = auto()

    ON_STATUS_APPLY = auto()
    ON_STATUS_REMOVE = auto()

    ON_DAMAGE_CALC = auto()
    ON_BEFORE_HIT = auto()
    ON_AFTER_HIT = auto()

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


@dataclass
class EventContext:
    # TODO: battle: Battle
    rng: RNG
    actor: Pokemon | None = None
    target: Target | None = None
    action: Action | None = None


@dataclass(slots=True)
class HandlerResult:
    cancel: bool = False
    updated_payload: Payload = field(default_factory=dict)


class EventHandler(Protocol):
    def __call__(self, context: EventContext, payload: Payload) -> HandlerResult | None: ...


class EventOwner(Protocol):
    name: str

    def on_register(self, bus: "EventBus") -> None: ...
    def on_unregister(self, bus: "EventBus") -> None: ...


class EventRegistrationError(RuntimeError): ...


class EventUnregistrationError(RuntimeError): ...


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
        self._subscriptions: list[Subscription] = []
        self._dedupe_keys: set[tuple[int, int, int, int, bool]] = set()

    def on(
        self,
        event: Event,
        handler: EventHandler,
        *,
        priority: int = EventPriority.DEFAULT,
        owner: EventOwner | None = None,
        once: bool = False,
    ) -> Subscription:
        key = (event.value, id(handler), id(owner) if owner is not None else 0, int(priority), once)
        if key in self._dedupe_keys:
            return Subscription(event=event, handler=handler, priority=int(priority), owner=owner, once=once)

        sub = Subscription(event=event, handler=handler, priority=int(priority), owner=owner, once=once)
        self._subscriptions.append(sub)
        self._dedupe_keys.add(key)

        self._subscriptions.sort(key=lambda s: (-s.priority, s.event.value))
        return sub

    def off(self, subscription: Subscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)
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
        if not self._subscriptions:
            return

        remaining: list[Subscription] = []
        for sub in self._subscriptions:
            if sub.owner is owner:
                key = (
                    sub.event.value,
                    id(sub.handler),
                    id(owner),
                    int(sub.priority),
                    sub.once,
                )
                if key in self._dedupe_keys:
                    self._dedupe_keys.remove(key)
                continue
            remaining.append(sub)
        self._subscriptions = remaining

    def emit(self, event: Event, context: EventContext, payload: Payload | None = None) -> Payload:
        current: Payload = dict(payload or {})
        to_remove_once: list[Subscription] = []

        for sub in list(self._subscriptions):
            if sub.event is not event:
                continue

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
