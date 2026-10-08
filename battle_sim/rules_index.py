"""What the engine knows about a move, ability or item, and where its rules live — for explaining."""

import ast
import inspect
from collections.abc import Callable
from dataclasses import fields
from enum import Enum
from functools import cache
from pathlib import Path

from battle_sim.export_data import _rule_files
from battle_sim.models.moves import MoveEffect
from battle_sim.utils import field_values

_ROOT = Path(__file__).resolve().parent


@cache
def _mentions() -> dict[str, tuple[str, ...]]:
    """`Ability.X` / `Item.X` / a quoted move name -> the rule files that mention it, relative."""
    found: dict[str, set[str]] = {}
    for file in _rule_files():
        where = str(file.relative_to(_ROOT))
        for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
            key = None
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id in ("Ability", "Item"):
                    key = f"{node.value.id}.{node.attr}"
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 40:
                key = f"move:{node.value}"
            if key is not None:
                found.setdefault(key, set()).add(where)
    return {key: tuple(sorted(files)) for key, files in found.items()}


def ability_mentions(name: str) -> tuple[str, ...]:
    """Rule files that read this ability (by its enum name, e.g. `OVERCOAT`)."""
    return _mentions().get(f"Ability.{name}", ())


def item_mentions(name: str) -> tuple[str, ...]:
    return _mentions().get(f"Item.{name}", ())


def move_mentions(move_name: str) -> tuple[str, ...]:
    """Rule files that special-case this move by name — a move with hand-written rules."""
    return _mentions().get(f"move:{move_name}", ())


def _plain(value: object) -> object:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {_plain(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def describe_effect(effect: MoveEffect) -> str:
    """One effect as `Kind(field=value, ...)`, leaving out fields still at their defaults."""
    shown = []
    values = field_values(effect)
    for spec in fields(effect):
        value = values[spec.name]
        if spec.default is not inspect.Parameter.empty and value == spec.default:
            continue
        shown.append(f"{spec.name}={_plain(value)}")
    return f"{type(effect).__name__}({', '.join(shown)})"


def binder_note(binder: Callable[..., object]) -> str:
    """The first paragraph of a binder's own docstring: what its author said it does."""
    doc = inspect.getdoc(binder) or ""
    return doc.split("\n\n")[0].replace("\n", " ").strip()
