"""CryoAtlas addition: export a trusted CNN checkpoint for inference.

Extracts the model/module state dictionary without optimizer state and
refuses to overwrite an existing destination.
"""
import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("module", checkpoint.get("model", checkpoint))
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("checkpoint does not contain a model state_dict")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Exported {len(state)} tensors to {args.output}")


if __name__ == "__main__":
    main()
