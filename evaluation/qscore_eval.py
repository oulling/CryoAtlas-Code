"""
Q-Score Evaluation Script
Evaluate map-model fit using ChimeraX Q-score.

Usage:
    python qscore_eval.py -p <model.pdb/cif> -m <map.mrc> [-o output.log] [--verbose]

Metrics:
    - Q-score (overall mean)
"""

import subprocess
import argparse
import re
import os


def add_args(parser):
    parser.add_argument(
        "--predicted-structure",
        "--p",
        "-p",
        type=str,
        required=True,
        help="Path to model structure (PDB/CIF format)",
    )
    parser.add_argument(
        "--map-file",
        "--m",
        "-m",
        type=str,
        required=True,
        help="Path to density map (.mrc/.map format)",
    )
    parser.add_argument(
        "--output-file",
        "--o",
        "-o",
        help="If set, path to save output file",
    )
    parser.add_argument(
        "--chimerax-bin",
        type=str,
        default="chimerax",
        help="Path to ChimeraX binary (default: chimerax)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints the results to the console",
    )

    return parser


def main(parsed_args):
    structure = parsed_args.predicted_structure
    map_file = parsed_args.map_file
    chimerax = parsed_args.chimerax_bin

    if not os.path.exists(structure):
        raise FileNotFoundError("Structure file not found: %s" % structure)
    if not os.path.exists(map_file):
        raise FileNotFoundError("Map file not found: %s" % map_file)

    chimerax_cmd = (
        "open %s ; open %s ; qscore #1 toVolume #2 useGui false ; exit"
        % (structure, map_file)
    )
    command = [chimerax, "--nogui", "--cmd", chimerax_cmd]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output_log = result.stdout + result.stderr

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("*" * 52 + "\n")
                f.write("Credit from ChimeraX: Q-score\n")
                f.write("*" * 52 + "\n")
                f.write(output_log)

        if parsed_args.verbose:
            print("*" * 52)
            print("Credit from ChimeraX: Q-score")
            print("*" * 52)
            print(output_log)

        ## Parse output ##
        qscore_match = re.search(r"Overall mean Q-Score:\s*([\d.]+)", output_log)

        if not qscore_match:
            raise ValueError("Failed to parse Q-score from ChimeraX output")

        output = {
            "qscore": float(qscore_match.group(1)),
        }

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("\n")
                for key, value in output.items():
                    f.write("%s: %s\n" % (key, value))

        if parsed_args.verbose:
            for key, value in output.items():
                print("%s: %s" % (key, value))

        return output

    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate map-model fit using ChimeraX Q-score"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
