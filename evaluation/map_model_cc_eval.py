"""
Map-Model CC Evaluation Script
Evaluate map-model fit using Phenix map_correlations.

Usage:
    python map_model_cc_eval.py -p <model.pdb/cif> -m <map.mrc> [-o output.log] [--verbose]

Metrics:
    - CC_mask
    - CC_volume
    - CC_peaks
    - CC_box
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
        "--verbose",
        action="store_true",
        help="If set, prints the results to the console",
    )

    return parser


def main(parsed_args):
    structure = parsed_args.predicted_structure
    map_file = parsed_args.map_file

    if not os.path.exists(structure):
        raise FileNotFoundError("Structure file not found: %s" % structure)
    if not os.path.exists(map_file):
        raise FileNotFoundError("Map file not found: %s" % map_file)

    command = ["phenix.map_correlations", structure, map_file]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output_log = result.stdout + result.stderr

        if result.returncode != 0 and "CC_mask" not in output_log:
            raise RuntimeError(
                "phenix.map_correlations failed with exit code %d:\n%s"
                % (result.returncode, result.stderr)
            )

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("*" * 52 + "\n")
                f.write("Credit from PHENIX: phenix.map_correlations\n")
                f.write("*" * 52 + "\n")
                f.write(output_log)

        if parsed_args.verbose:
            print("*" * 52)
            print("Credit from PHENIX: phenix.map_correlations")
            print("*" * 52)
            print(output_log)

        ## Parse output ##
        cc_mask_match = re.search(r"CC_mask\s*:\s*([\d.]+)", output_log)
        cc_volume_match = re.search(r"CC_volume\s*:\s*([\d.]+)", output_log)
        cc_peaks_match = re.search(r"CC_peaks\s*:\s*([\d.]+)", output_log)
        cc_box_match = re.search(r"CC_box\s*:\s*([\d.]+)", output_log)

        matches = [cc_mask_match, cc_volume_match, cc_peaks_match, cc_box_match]
        names = ["cc_mask", "cc_volume", "cc_peaks", "cc_box"]
        missing = [n for n, m in zip(names, matches) if m is None]
        if missing:
            raise ValueError(
                "Failed to parse map_correlations output. Missing: %s" % missing
            )

        output = {
            "cc_mask": float(cc_mask_match.group(1)),
            "cc_volume": float(cc_volume_match.group(1)),
            "cc_peaks": float(cc_peaks_match.group(1)),
            "cc_box": float(cc_box_match.group(1)),
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
        description="Evaluate map-model fit using Phenix map_correlations"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
