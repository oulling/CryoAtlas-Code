"""
Benchmark Evaluation Entry Point
Comprehensive evaluation of cryo-EM atomic model building methods.

Usage:
    python evaluate.py -p <predicted.pdb/cif> -t <target.pdb/cif> -o <output.log> [options]

Modules:
    --usalign         (default: True)  USalign TM-score and Ca RMSD
    --chain-comparison(default: True)  Phenix chain_comparison (residue_coverage, sequence_match, etc.)
    --molprobity      (default: False) Phenix MolProbity stereochemistry metrics
    --map-cc          (default: False) Phenix map-model CC metrics (requires --map-file)
    --qscore          (default: False) ChimeraX Q-score (requires --map-file)

Superposition:
    --need-superpose  (default: False) If True, first superpose predicted onto target using USalign,
                      then use the superposed structure for all subsequent evaluations.
                      Required for methods like AF3/E3CryoFold that predict in their own coordinate frame.
"""

import argparse
import os
import json

from usalign_eval import main as usalign_main
from molprobity_eval import main as molprobity_main
from map_model_cc_eval import main as map_model_cc_main
from qscore_eval import main as qscore_main
from chain_comparison_eval import main as chain_comparison_main


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
        required=True,
        help="Path to save output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, prints the results to the console",
    )
    # Superposition
    parser.add_argument(
        "--need-superpose",
        type=str2bool,
        default=False,
        help="If True, superpose predicted onto target via USalign before other evaluations. "
             "Required for AF3/E3CryoFold etc. (default: False)",
    )
    # Module switches
    parser.add_argument(
        "--usalign",
        type=str2bool,
        default=True,
        help="If True, run USalign evaluation (default: True)",
    )
    parser.add_argument(
        "--chain-comparison",
        type=str2bool,
        default=True,
        help="If True, run phenix.chain_comparison (default: True)",
    )
    parser.add_argument(
        "--molprobity",
        type=str2bool,
        default=False,
        help="If True, run MolProbity evaluation (default: False)",
    )
    parser.add_argument(
        "--map-cc",
        type=str2bool,
        default=False,
        help="If True, run map-model CC evaluation (default: False)",
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=None,
        help="Path to density map (.mrc/.map), required for --map-cc and --qscore",
    )
    parser.add_argument(
        "--qscore",
        type=str2bool,
        default=False,
        help="If True, run ChimeraX Q-score evaluation (default: False)",
    )
    parser.add_argument(
        "--chimerax-bin",
        type=str,
        default="chimerax",
        help="Path to ChimeraX binary (default: chimerax)",
    )
    # USalign options
    parser.add_argument(
        "--usalign-bin",
        type=str,
        default=None,
        help="Path to USalign binary",
    )
    parser.add_argument(
        "--distance-threshold",
        "-d",
        type=float,
        default=3.0,
        help="Distance threshold for USalign (default: 3.0 Angstrom)",
    )

    return parser


def main(parsed_args):
    given_output_file = parsed_args.output_file
    dir_path = os.path.dirname(given_output_file)
    if not dir_path:
        dir_path = "."
    file_name_without_ext = os.path.splitext(os.path.basename(given_output_file))[0]

    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # Log file for detailed trace
    log_file = os.path.join(dir_path, "%s_TraceLog.log" % file_name_without_ext)
    parsed_args.output_file = log_file

    # The structure used for downstream evaluations
    # (may be replaced by superposed structure if --need-superpose)
    eval_structure = parsed_args.predicted_structure

    usalign_output = {}
    chain_comparison_output = {}
    molprobity_output = {}
    map_cc_output = {}
    qscore_output = {}

    # --- USalign (always run first if enabled, handles superposition) ---
    if parsed_args.usalign or parsed_args.need_superpose:
        print("Running USalign ...")
        # If need superpose, tell USalign to output superposed structure
        if parsed_args.need_superpose:
            sup_prefix = os.path.join(dir_path, "%s_sup" % file_name_without_ext)
            parsed_args.superpose_output = sup_prefix
        else:
            parsed_args.superpose_output = None

        with open(log_file, 'a') as f:
            f.write("[Measure - USalign]\n")
            f.flush()
        usalign_output = usalign_main(parsed_args) or {}
        with open(log_file, 'a') as f:
            f.write("\nDONE!\n\n")
            f.flush()

        # Use superposed structure for all subsequent evaluations
        if parsed_args.need_superpose and "_superposed_structure" in usalign_output:
            eval_structure = usalign_output["_superposed_structure"]
            print("Using superposed structure: %s" % eval_structure)

    # --- Chain Comparison ---
    if parsed_args.chain_comparison:
        print("Running Chain Comparison ...")
        # Use eval_structure (possibly superposed) as predicted
        cc_args = argparse.Namespace(
            predicted_structure=eval_structure,
            target_structure=parsed_args.target_structure,
            output_file=log_file,
            verbose=parsed_args.verbose,
        )
        with open(log_file, 'a') as f:
            f.write("[Measure - Chain Comparison]\n")
            f.flush()
        chain_comparison_output = chain_comparison_main(cc_args) or {}
        with open(log_file, 'a') as f:
            f.write("\nDONE!\n\n")
            f.flush()

    # --- MolProbity ---
    if parsed_args.molprobity:
        print("Running MolProbity ...")
        mp_args = argparse.Namespace(
            predicted_structure=eval_structure,
            output_file=log_file,
            verbose=parsed_args.verbose,
        )
        with open(log_file, 'a') as f:
            f.write("[Measure - MolProbity]\n")
            f.flush()
        molprobity_output = molprobity_main(mp_args) or {}
        with open(log_file, 'a') as f:
            f.write("\nDONE!\n\n")
            f.flush()

    # --- Map-Model CC ---
    if parsed_args.map_cc:
        if not parsed_args.map_file:
            print("Warning: --map-cc requires --map-file, skipping.")
        else:
            print("Running Map-Model CC ...")
            mc_args = argparse.Namespace(
                predicted_structure=eval_structure,
                map_file=parsed_args.map_file,
                output_file=log_file,
                verbose=parsed_args.verbose,
            )
            with open(log_file, 'a') as f:
                f.write("[Measure - Map-Model CC]\n")
                f.flush()
            map_cc_output = map_model_cc_main(mc_args) or {}
            with open(log_file, 'a') as f:
                f.write("\nDONE!\n\n")
                f.flush()

    # --- Q-score ---
    if parsed_args.qscore:
        if not parsed_args.map_file:
            print("Warning: --qscore requires --map-file, skipping.")
        else:
            print("Running Q-score ...")
            qs_args = argparse.Namespace(
                predicted_structure=eval_structure,
                map_file=parsed_args.map_file,
                chimerax_bin=parsed_args.chimerax_bin,
                output_file=log_file,
                verbose=parsed_args.verbose,
            )
            with open(log_file, 'a') as f:
                f.write("[Measure - Q-score]\n")
                f.flush()
            qscore_output = qscore_main(qs_args) or {}
            with open(log_file, 'a') as f:
                f.write("\nDONE!\n\n")
                f.flush()

    # --- Write final output ---
    parsed_args.output_file = given_output_file
    all_results = {}

    print("\nEvaluation Results:")
    with open(parsed_args.output_file, 'a') as f:
        pred_name = os.path.basename(parsed_args.predicted_structure)
        target_name = os.path.basename(parsed_args.target_structure)
        f.write("# Prediction: %s  |  Target: %s #\n" % (pred_name, target_name))

        if parsed_args.usalign and usalign_output:
            f.write("########## USalign Results ##########\n")
            for key, value in usalign_output.items():
                if not key.startswith("_"):  # skip internal keys
                    f.write("%s: %s\n" % (key, value))
                    print("  - %s: %s" % (key, value))
            all_results.update({k: v for k, v in usalign_output.items() if not k.startswith("_")})

        if parsed_args.chain_comparison and chain_comparison_output:
            f.write("########## Chain Comparison Results ##########\n")
            for key, value in chain_comparison_output.items():
                f.write("%s: %s\n" % (key, value))
                print("  - %s: %s" % (key, value))
            all_results.update(chain_comparison_output)

        if parsed_args.molprobity and molprobity_output:
            f.write("########## MolProbity Results ##########\n")
            for key, value in molprobity_output.items():
                f.write("%s: %s\n" % (key, value))
                print("  - %s: %s" % (key, value))
            all_results.update(molprobity_output)

        if parsed_args.map_cc and map_cc_output:
            f.write("########## Map-Model CC Results ##########\n")
            for key, value in map_cc_output.items():
                f.write("%s: %s\n" % (key, value))
                print("  - %s: %s" % (key, value))
            all_results.update(map_cc_output)

        if parsed_args.qscore and qscore_output:
            f.write("########## Q-score Results ##########\n")
            for key, value in qscore_output.items():
                f.write("%s: %s\n" % (key, value))
                print("  - %s: %s" % (key, value))
            all_results.update(qscore_output)

        f.write("\n\n")
        f.flush()

    # Also write JSON for easy parsing
    json_file = os.path.join(dir_path, "%s.json" % file_name_without_ext)
    with open(json_file, 'w') as f:
        json.dump({
            "prediction": parsed_args.predicted_structure,
            "target": parsed_args.target_structure,
            "need_superpose": parsed_args.need_superpose,
            "results": all_results,
        }, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark evaluation for cryo-EM atomic model building"
    )
    parser = add_args(parser)
    parsed_args = parser.parse_args()
    main(parsed_args)
