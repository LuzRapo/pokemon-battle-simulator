"""Write a trained network out as ONNX, which `battle_sim.rl.net_player.NetPlayer` runs without torch.

uv run python -m training.export runs/first/latest.pt runs/first/policy.onnx
"""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from battle_sim.rl import encode
from training.model import ACTIONS, ModelConfig, PolicyValueNet


def load_checkpoint(path: Path) -> PolicyValueNet:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyValueNet(ModelConfig(**state["model_config"]))
    model.load_state_dict(state["model"])
    return model.eval()


def export(model: PolicyValueNet, out: Path) -> None:
    batch = 3
    example = (
        torch.ones(batch, encode.SLOTS, encode.POKEMON_IDS, dtype=torch.int64),
        torch.zeros(batch, encode.SLOTS, encode.POKEMON_FLOATS),
        torch.zeros(batch, encode.FIELD_FLOATS),
        torch.ones(batch, ACTIONS, dtype=torch.bool),
    )
    names = ["ids", "pokemon", "field", "mask"]
    rows = torch.export.Dim("rows")
    torch.onnx.export(
        model,
        example,
        str(out),
        input_names=names,
        output_names=["logits", "value"],
        dynamic_shapes={name: {0: rows} for name in names},
        dynamo=True,
        external_data=False,
    )
    _check(model, out, example)


def _check(model: PolicyValueNet, out: Path, example: tuple[torch.Tensor, ...]) -> None:
    """The exported graph must compute what the torch model does, on inputs that are not all zero."""
    generator = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 50, example[0].shape, generator=generator)
    pokemon = torch.rand(example[1].shape, generator=generator)
    field = torch.rand(example[2].shape, generator=generator)
    mask = torch.rand(example[3].shape, generator=generator) > 0.5
    mask[:, 0] = True
    with torch.no_grad():
        logits, value = model(ids, pokemon, field, mask)
    session = onnxruntime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    feeds = {"ids": ids.numpy(), "pokemon": pokemon.numpy(), "field": field.numpy(), "mask": mask.numpy()}
    got_logits, got_value = session.run(None, feeds)
    np.testing.assert_allclose(got_logits, logits.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(got_value, value.numpy(), rtol=1e-4, atol=1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="latest.pt or a snapshot")
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    export(load_checkpoint(args.checkpoint), args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
