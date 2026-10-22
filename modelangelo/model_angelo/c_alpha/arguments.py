"""CryoAtlas addition: configurable CNN training arguments.

Defines data paths, optimization, loss schedules, logging, and checkpoint
options. The launcher enables phosphorus supervision explicitly; numerical
module defaults are retained for checkpoint-era compatibility.
"""
import argparse


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Train the CNN C-alpha/phosphorus segmentation model."
    )

    p.add_argument("--dataset-list", type=str, required=True)
    p.add_argument("--valid-dataset-list", type=str, required=True)
    p.add_argument("--log-dir", type=str, required=True)

    p.add_argument("--box-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=400_000)
    p.add_argument(
        "--scheduler-steps",
        type=int,
        default=None,
        help="Cosine-scheduler horizon. Defaults to --num-steps. Set this "
             "independently for short runs that must preserve the original "
             "learning-rate trajectory.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--valid-every", type=int, default=1_000)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    p.add_argument("--log-every", type=int, default=50)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--use-cosine-annealing", action="store_true")

    # Tversky schedule
    p.add_argument("--tversky-warmup-steps", type=int, default=10_000)
    p.add_argument("--tversky-ramp-steps", type=int, default=30_000)
    p.add_argument("--tversky-ramp-shape", type=str, default="cosine",
                   choices=["linear", "cosine"])
    p.add_argument("--tversky-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)

    # Per-channel loss weights
    p.add_argument("--channel-weights", type=str, default="1.0,0.0,0.5",
                   help="Comma-separated weights for (Ca, P, backbone) channels "
                        "in focal+tversky losses. The module default disables "
                        "phosphorus supervision; use a nonzero P weight for "
                        "nucleotide-containing training data.")

    # Late-stage focal boost
    p.add_argument("--focal-late-boost-start", type=int, default=50_000,
                   help="Step at which focal_weight starts ramping above 1.0.")
    p.add_argument("--focal-late-boost-end", type=int, default=80_000,
                   help="Step at which focal_weight reaches its max.")
    p.add_argument("--focal-late-boost-max", type=float, default=1.5,
                   help="Final focal_weight after the late-stage boost ramp. "
                        "Set 1.0 to disable the ramp entirely.")

    p.add_argument("--positional-encoding-dim", type=int, default=0)
    p.add_argument("--dont-use-data-augmentation", action="store_true")
    p.add_argument("--use-global-normalization", action="store_true")
    p.add_argument("--max-noise", type=float, default=0.3)

    p.add_argument("--bfactor-range", type=float, default=25.0)
    p.add_argument("--random-crop-prob", type=float, default=0.25)

    p.add_argument("--dont-load", action="store_true")
    p.add_argument("--match-model", action="store_true")
    p.add_argument("--debug", action="store_true")

    p.add_argument("--num-workers", type=int, default=None)

    p.add_argument("--grad-clip", type=float, default=0.5)

    p.add_argument("--valid-crops", type=int, default=200)
    p.add_argument("--mrc-pred-every", type=int, default=0)
    p.add_argument("--mrc-pred-pickles", type=str, default="")
    p.add_argument("--mrc-pred-n-maps", type=int, default=5)
    p.add_argument("--mrc-pred-stride", type=int, default=32)

    p.add_argument("--wandb-project", type=str, default="modelangelo-stage1-v2-1")
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--no-wandb", action="store_true")

    return p.parse_args()
