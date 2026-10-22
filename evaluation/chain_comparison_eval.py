"""
Chain Comparison Evaluation Script
Evaluate protein model recovery using Phenix chain_comparison.

Usage:
    python chain_comparison_eval.py -p <predicted.cif> -t <target.cif> [-o output.log] [--verbose]

Metrics:
    - residue_coverage: fraction of built residues within 3A of target (FOUND)
    - sequence_match: fraction of close residues with correct residue type (SEQ MATCH)
    - sequence_recall: residue_coverage * sequence_match
    - completeness: len_predict / len_target
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
        "--verbose",
        action="store_true",
        help="If set, prints the results to the console",
    )

    return parser


def main(parsed_args):
    predicted = parsed_args.predicted_structure
    target = parsed_args.target_structure

    if not os.path.exists(predicted):
        raise FileNotFoundError("Predicted structure not found: %s" % predicted)
    if not os.path.exists(target):
        raise FileNotFoundError("Target structure not found: %s" % target)

    # phenix.chain_comparison needs two different file paths
    # If same file, create a temporary symlink/copy
    import shutil
    import tempfile
    tmp_copy = None
    if os.path.abspath(predicted) == os.path.abspath(target):
        suffix = os.path.splitext(predicted)[1]
        tmp_fd, tmp_copy = tempfile.mkstemp(suffix=suffix)
        os.close(tmp_fd)
        shutil.copy2(predicted, tmp_copy)
        predicted = tmp_copy

    # phenix.chain_comparison target_file query_file
    command = ["phenix.chain_comparison", target, predicted]

    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        output_log = result.stdout + result.stderr

        if result.returncode != 0 and "FOUND" not in output_log:
            raise RuntimeError(
                "phenix.chain_comparison failed with exit code %d:\n%s"
                % (result.returncode, result.stderr)
            )

        if parsed_args.output_file is not None:
            with open(parsed_args.output_file, "a") as f:
                f.write("*" * 52 + "\n")
                f.write("Credit from PHENIX: phenix.chain_comparison\n")
                f.write("*" * 52 + "\n")
                f.write(output_log)

        if parsed_args.verbose:
            print("*" * 52)
            print("Credit from PHENIX: phenix.chain_comparison")
            print("*" * 52)
            print(output_log)

        ## Parse output ##
        # Parse the result line (last data line before "Finished")
        # Format: RMSD N_close N_far N_forward N_reverse N_mixed FOUND CA_SCORE SEQ_MATCH SEQ_SCORE MEAN_LENGTH FRAGMENTS BAD_CONNECTIONS
        pattern = r"_target\s+(\d+\.\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+)"
        match = re.search(pattern, output_log)

        if not match:
            raise ValueError("Failed to parse chain_comparison output")

        found = float(match.group(7))           # FOUND (%)
        seq_match_pct = float(match.group(9))   # SEQ MATCH (%)

        # Parse original structure lengths (full, before deduplication)
        orig_pattern = r"Original residues in target:\s*(\d+)\s+In query:\s*(\d+)"
        orig_match = re.search(orig_pattern, output_log)
        if not orig_match:
            raise ValueError("Failed to parse original structure lengths")
        len_target = int(orig_match.group(1))
        len_predict = int(orig_match.group(2))

        # Calculate metrics
        residue_coverage = found / 100.0
        sequence_match = seq_match_pct / 100.0
        sequence_recall = residue_coverage * sequence_match
        completeness = len_predict / len_target if len_target > 0 else 0.0

        output = {
            "residue_coverage": round(residue_coverage, 4),
            "sequence_match": round(sequence_match, 4),
            "sequence_recall": round(sequence_recall, 4),
            "completeness": round(completeness, 4),
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
    finally:
        if tmp_copy and os.path.exists(tmp_copy):
            os.remove(tmp_copy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate protein model recovery using Phenix chain_comparison"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
