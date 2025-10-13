import pytest

from battle_sim.maths.rng import RNG
from battle_sim.mechanics.events import Event, EventBus, EventContext, EventPriority, HandlerResult
from battle_sim.models.actions import Action, ActionType
from battle_sim.models.moves import MoveSlot
from battle_sim.utils import Target


def make_context(garchomp_factory):
    chompilly = garchomp_factory("Chompilly")
    action = Action(action=ActionType.USE_MOVE, target=Target.SINGLE_OPPONENT, move=MoveSlot.FIRST)
    return EventContext(rng=RNG(seed=123), actor=chompilly, target=action.target, action=action)


@pytest.fixture
def context(garchomp_factory):
    return make_context(garchomp_factory)


def test_subscription_ordering_and_execution(context):
    bus = EventBus()
    order = []

    def h1(ctx, payload):
        order.append("h1")
        return HandlerResult()

    def h2(ctx, payload):
        order.append("h2")
        return HandlerResult()

    bus.on(Event.ON_ACTION_START, h2, priority=EventPriority.DEFAULT)
    bus.on(Event.ON_ACTION_START, h1, priority=EventPriority.MOVE)
    bus.emit(Event.ON_ACTION_START, context)
    assert order == ["h1", "h2"]


def test_payload_merging(context):
    bus = EventBus()

    def add_a(ctx, payload):
        return HandlerResult(updated_payload={"a": 1})

    def add_b(ctx, payload):
        assert payload["a"] == 1
        return HandlerResult(updated_payload={"b": 2})

    out = bus.emit(Event.ON_TURN_START, context, {})
    assert out == {}

    bus.on(Event.ON_TURN_START, add_a)
    bus.on(Event.ON_TURN_START, add_b)
    out = bus.emit(Event.ON_TURN_START, context, {})
    assert out == {"a": 1, "b": 2}


def test_cancellation_stops_later_handlers(context):
    bus = EventBus()
    called = []

    def first(ctx, payload):
        called.append("first")
        return HandlerResult(cancel=True)

    def second(ctx, payload):
        called.append("second")
        return HandlerResult()

    bus.on(Event.ON_TURN_END, first)
    bus.on(Event.ON_TURN_END, second)
    bus.emit(Event.ON_TURN_END, context)
    assert called == ["first"]


def test_once_subscription_removed_after_emit(context):
    bus = EventBus()
    hits = []

    def once_handler(ctx, payload):
        hits.append(1)
        return HandlerResult()

    bus.on(Event.ON_BEFORE_ACTION, once_handler, once=True)
    bus.emit(Event.ON_BEFORE_ACTION, context)
    bus.emit(Event.ON_BEFORE_ACTION, context)
    assert hits == [1]


def test_deduplication_same_handler_not_added_twice(context):
    bus = EventBus()
    calls = 0

    def handler(ctx, payload):
        nonlocal calls
        calls += 1
        return None

    sub1 = bus.on(Event.ON_AFTER_ACTION, handler)
    sub2 = bus.on(Event.ON_AFTER_ACTION, handler)

    assert sub1 == sub2
    assert sub1 is not sub2

    out = bus.emit(Event.ON_AFTER_ACTION, context)
    assert out == {}
    assert calls == 1


def test_off_owner_removes_all_owned_subscriptions(context):
    bus = EventBus()

    class DummyOwner:
        name = "X"

        def on_register(self, b): ...
        def on_unregister(self, b): ...

    owner = DummyOwner()
    called = []

    def h(ctx, payload):
        called.append("h")
        return HandlerResult()

    bus.on(Event.ON_FAINT, h, owner=owner)
    bus.on(Event.ON_FAINT, h)
    bus.off_owner(owner)
    bus.emit(Event.ON_FAINT, context)
    assert called == ["h"]


def test_emit_returns_initial_payload_if_no_handlers(context):
    bus = EventBus()
    out = bus.emit(Event.ON_TURN_START, context, {"x": 1})
    assert out == {"x": 1}
