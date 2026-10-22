"""
USalign Evaluation Script
Evaluate structure similarity using US-align.

Usage:
    python usalign_eval.py -p <predicted.pdb/cif> -t <target.pdb/cif> [-o output.log] [--verbose]

Metrics:
    - TM-score (normalized by target length)
    - RMSD (Ca RMSD of aligned residues)
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
        help="Path to predicted structure (PDB/CIF format)",
    )
    parser.add_argument(
        "--target-structure",
        "--t",
        "-t",
        type=str,
        required=True,
        help="Path to target structure (PDB/CIF format)",
    )
    parser.add_argument(
        "--output-file",
        "--o",
        "-o",
        help="If set, path to save output file",
    )
    parser.add_argument(
        "--usalign-bin",
        type=str,
        default=None,
        help="Path to USalign binary (default: auto-detect in script directory)",
    )
    parser.add_argument(
        "--distance-threshold",
        "-d",
        type=float,
        default=3.0,
        help="Distance threshold in Angstrom for aligned residues (default: 3.0)",
    )
    parser.add_argument(
        "--superpose-output",
        type=str,
        default=None,
        help="If set, save superposed structure to this path (for methods needing alignment)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints the results to the console",
    )

    return parser


def main(parsed_args):
    # Locate USalign binary
    if parsed_args.usalign_bin:
        usalign = parsed_args.usalign_bin
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        usalign = os.path.join(script_dir, "USalign")

    if not os.path.exists(usalign):
        raise FileNotFoundError(
            "USalign binary not found: %s\n"
            "Please specify --usalign-bin or place USalign in the script directory." % usalign
        )

    d_threshold = parsed_args.distance_threshold

    superpose_output = getattr(parsed_args, 'superpose_output', None)

    command = [
        usalign,
        parsed_args.predicted_structure,
        parsed_args.target_structure,
        "-mm", "1",
        "-ter", "0",
        "-d", str(d_threshold),
        "-outfmt", "-1",
    ]

    if superpose_output:
        command.extend(["-o", superpose_output])

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output_log = result.stdout

        if result.returncode != 0 and "TM-score" not in output_log:
            raise RuntimeError(
                "USalign failed with exit code %d:\n%s" % (result.returncode, result.stderr)
            )

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, 'a') as f:
                f.write("*" * 52 + "\n")
                f.write("Credit from Zhang Lab: US-align (Version 20260328)\n")
                f.write("Reference: C Zhang, M Shine, AM Pyle, Y Zhang. (2022) Nat Methods\n")
                f.write("*" * 52 + "\n")
                f.write(output_log)

        if parsed_args.verbose:
            print("*" * 52)
            print("Credit from Zhang Lab: US-align")
            print("*" * 52)
            print(output_log)

        ## Parse output ##
        rmsd_match = re.search(r"RMSD=\s*([0-9.]+)", output_log)
        if not rmsd_match:
            raise ValueError("RMSD not found in the output")
        rmsd = float(rmsd_match.group(1))

        tm_score_match = re.search(
            r"TM-score= ([0-9.]+) \(normalized by length of Structure_2", output_log)
        if not tm_score_match:
            raise ValueError("TM-score not found in the output")
        tm_score = float(tm_score_match.group(1))

        output = {
            "tm_score": round(tm_score, 3),
            "ca_rmsd": round(rmsd, 3),
        }

        # Record superposed structure path if generated
        if superpose_output:
            sup_cif = superpose_output + ".cif"
            if os.path.exists(sup_cif):
                output["_superposed_structure"] = sup_cif

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, 'a') as f:
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
        description="Evaluate structure similarity using US-align"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
