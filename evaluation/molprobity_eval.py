"""
MolProbity Evaluation Script
Evaluate the stereochemical quality of a protein structure using Phenix MolProbity.

Usage:
    python molprobity_eval.py -p <predicted_model.pdb/cif> [-o output.log] [--verbose]

Metrics:
    - Ramachandran outliers / favored (%)
    - Rotamer outliers (%)
    - C-beta deviations
    - Clashscore
    - RMS(bonds), RMS(angles)
    - MolProbity score
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

    if not os.path.exists(structure):
        raise FileNotFoundError(f"Structure file not found: {structure}")

    command = ["phenix.molprobity", structure]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output_log = result.stdout + result.stderr

        if result.returncode != 0 and "Summary" not in output_log:
            raise RuntimeError(
                f"phenix.molprobity failed with exit code {result.returncode}:\n"
                f"{result.stderr}"
            )

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("*" * 52 + "\n")
                f.write("Credit from PHENIX: phenix.molprobity\n")
                f.write("*" * 52 + "\n")
                f.write(output_log)

        if parsed_args.verbose:
            print("*" * 52)
            print("Credit from PHENIX: phenix.molprobity")
            print("*" * 52)
            print(output_log)

        ## Parse Summary ##
        rama_outlier_match = re.search(
            r"Ramachandran outliers\s*=\s*([\d.]+)\s*%", output_log
        )
        rama_favored_match = re.search(
            r"favored\s*=\s*([\d.]+)\s*%", output_log
        )
        rotamer_outlier_match = re.search(
            r"Rotamer outliers\s*=\s*([\d.]+)\s*%", output_log
        )
        cbeta_match = re.search(
            r"C-beta deviations\s*=\s*(\d+)", output_log
        )
        clashscore_match = re.search(
            r"Clashscore\s*=\s*([\d.]+)", output_log
        )
        rms_bonds_match = re.search(
            r"RMS\(bonds\)\s*=\s*([\d.]+)", output_log
        )
        rms_angles_match = re.search(
            r"RMS\(angles\)\s*=\s*([\d.]+)", output_log
        )
        molprobity_score_match = re.search(
            r"MolProbity score\s*=\s*([\d.]+)", output_log
        )

        matches = [
            rama_outlier_match, rama_favored_match, rotamer_outlier_match,
            cbeta_match, clashscore_match, rms_bonds_match,
            rms_angles_match, molprobity_score_match,
        ]
        if not all(matches):
            missing = []
            names = [
                "rama_outlier", "rama_favored", "rotamer_outlier",
                "cbeta_deviations", "clashscore", "rms_bonds",
                "rms_angles", "molprobity_score",
            ]
            for name, m in zip(names, matches):
                if m is None:
                    missing.append(name)
            raise ValueError(
                f"Failed to parse MolProbity summary. Missing: {missing}"
            )

        output = {
            "rama_outlier": float(rama_outlier_match.group(1)),
            "rama_favored": float(rama_favored_match.group(1)),
            "rotamer_outlier": float(rotamer_outlier_match.group(1)),
            "cbeta_deviations": int(cbeta_match.group(1)),
            "clashscore": float(clashscore_match.group(1)),
            "rms_bonds": float(rms_bonds_match.group(1)),
            "rms_angles": float(rms_angles_match.group(1)),
            "molprobity_score": float(molprobity_score_match.group(1)),
        }

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("\n")
                for key, value in output.items():
                    f.write(f"{key}: {value}\n")

        if parsed_args.verbose:
            for key, value in output.items():
                print(f"{key}: {value}")

        return output

    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate structure quality using Phenix MolProbity"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
