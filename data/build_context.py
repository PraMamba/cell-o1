"""
Step 1: Convert raw .h5ad files into structured single-cell JSON with metadata and top genes.

Usage:
    python build_context.py \
        --input_dir path/to/raw_h5ad_folder \
        --output_dir path/to/save/json_output \
        --max_cells 20000
"""

import os
import json
import argparse
import anndata
import numpy as np
from tqdm import tqdm
import concurrent.futures

# ========== Constants ==========
MANDATORY_FIELDS = ["disease", "tissue", "sex", "development_stage", "self_reported_ethnicity"]

EXTRA_FIELDS_MAP = {
    "copd": ["ever_smoker", "tumor_stage"],
    "hiv": ["sampletype"],
    "type2_diabetes": ["donor_BMI"],
    "cystic_fibrosis": ["lung_condition", "smoking_status", "BMI"],
    "cortex": ["medical_conditions", "hemisphere"],
    "tumor": ["harm_tumor.type", "harm_tumor.site", "harm_sample.type", "harm_condition", "harm_cd45pos"],
    "kidney": ["diabetes_history", "hypertension", "BMI", "eGFR"],
    "fibroblast": ["Condition"],
    "obvarian": ["author_tumor_subsite"],
    "pbmc": ["activation"],
    "leukemic": ["Genotype"]
}

MEDICAL_CONDITION_MAP = {
    "epilepsy": "The patient has a history of epilepsy.",
    "tumor": "The patient has a diagnosed brain tumor.",
    "hydrocephalus": "The patient has hydrocephalus (fluid buildup in the brain).",
    "both": "The patient has both epilepsy and a brain tumor.",
    "other": "The patient has other neurological conditions."
}

FIBROBLAST_CONDITION_MAP = {
    "healthy": "The donor was healthy with no reported skin condition.",
    "dm – non ulcer": "The donor had diabetes mellitus without skin ulceration.",
    "keloid": "The donor had a keloid, which is an overgrowth of scar tissue.",
    "localised scleroderma": "The donor was diagnosed with localized scleroderma, a skin-related autoimmune disease.",
    "scar": "The donor had a typical scar from prior skin injury."
}

# ========== Utility Functions ==========
# (same as before: is_valid_value, get_top_genes, build_prompt...)

# Keep all previous helper function implementations unchanged...

# ========== Processing Logic ==========

def process_file(h5ad_path, root_input, root_output, max_cells=20000):
    # Create output subdir based on relative structure
    rel_path = os.path.relpath(os.path.dirname(h5ad_path), root_input)
    out_dir = os.path.join(root_output, rel_path)
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, os.path.splitext(os.path.basename(h5ad_path))[0] + ".json")
    process_h5ad(h5ad_path, output_path, max_cells)


def main():
    parser = argparse.ArgumentParser(description="Convert raw .h5ad files to single-cell JSON with metadata and top genes.")
    parser.add_argument("--input_dir", type=str, required=True, help="Root directory containing input .h5ad files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the processed JSON outputs")
    parser.add_argument("--max_cells", type=int, default=20000, help="Maximum number of cells per file")
    args = parser.parse_args()

    # Collect all .h5ad files
    h5ad_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file_name in files:
            if file_name.endswith(".h5ad"):
                h5ad_files.append(os.path.join(root, file_name))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(f"[INFO] Found {len(h5ad_files)} files. Starting parallel processing...")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(lambda path: process_file(path, args.input_dir, args.output_dir, args.max_cells), h5ad_files)


if __name__ == "__main__":
    main()
