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
from functools import partial


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
def is_valid_value(val):
    """
    Check if a value is valid (not None, 'NaN', 'unknown', etc.).
    Returns True if the value is valid, otherwise False.
    """
    if val is None:
        return False
    sval = str(val).strip().lower()
    if sval in ["nan", "unknown", "none", "", "na"]:
        return False
    return True


def get_top_genes(adata, cell_idx, top_n=50):
    """
    Retrieve top N expressed genes for a given cell index from an AnnData object.
    Uses a partition-based method (np.argpartition) to efficiently select the top N genes.
    Returns a list of gene names sorted in descending order of expression.
    """
    if "feature_name" not in adata.var.columns:
        raise KeyError("Missing 'feature_name' column in adata.var. Cannot extract gene names.")

    gene_names = adata.var["feature_name"].tolist()
    row_expr = adata.X[cell_idx, :]

    # Convert to a dense array if sparse
    if hasattr(row_expr, "toarray"):
        row_expr = row_expr.toarray().flatten()
    else:
        row_expr = row_expr.flatten()

    # Use np.argpartition to obtain indices of the top N values (unsorted)
    top_n_unsorted_idx = np.argpartition(row_expr, -top_n)[-top_n:]
    # Sort the top indices by expression in descending order
    sorted_top_idx = top_n_unsorted_idx[np.argsort(row_expr[top_n_unsorted_idx])[::-1]]
    
    # Retrieve gene names corresponding to these indices and clean ENSEMBL IDs
    top_N_genes = []
    for i in sorted_top_idx:
        gene_name = gene_names[i]
        # If gene name contains ENSEMBL ID (format: SYMBOL_ENSGxxxxxxxx), extract only the symbol part
        if "_ENSG" in gene_name:
            gene_name = gene_name.split("_ENSG")[0]
        top_N_genes.append(gene_name)
        
    return top_N_genes


def build_prompt(row_dict, top_genes):
    """
    Build an English prompt using the available context fields and top genes.
    'row_dict' is a dictionary of validated context fields.
    'top_genes' is a list of the top expressed genes.
    """
    parts = []
    sex = row_dict.get("sex")
    dev_stage = row_dict.get("development_stage")
    ethnicity = row_dict.get("self_reported_ethnicity")
    tissue = row_dict.get("tissue")
    disease = row_dict.get("disease")

    # Construct the introductory sentence
    intro = "The cell is from"
    if sex:
        intro += f" a {sex}"
    if dev_stage:
        intro += f" at the {dev_stage}"
    if ethnicity:
        intro += f" with {ethnicity} background"
    if tissue:
        intro += f", originating from the {tissue}"
    if disease:
        if disease.lower() == "normal":
            intro += ". The patient is healthy with no diagnosed disease."
        else:
            intro += f". The patient has been diagnosed with {disease}."
    parts.append(intro)

    # Extra context fields
    extras = []

    if "ever_smoker" in row_dict:
        val = row_dict["ever_smoker"].lower()
        if val in ["yes", "current smoker"]:
            extras.append("The patient is a smoker.")
        elif val in ["no", "non-smoker"]:
            extras.append("The patient is a non-smoker.")

    if "tumor_stage" in row_dict:
        val = row_dict["tumor_stage"].strip().lower()
        if val == "non-cancer":
            extras.append("There is no cancer present.")
        elif val == "early":
            extras.append("The patient has an early-stage tumor.")
        elif val == "advanced":
            extras.append("The patient has an advanced-stage tumor.")

    if "sampletype" in row_dict:
        st = row_dict["sampletype"].strip().upper()
        if st == "M3":
            extras.append("The sample was collected at month 3 post-treatment.")
        elif st == "M6":
            extras.append("The sample was collected at month 6 post-treatment.")
        elif st == "UV":
            extras.append("The sample was exposed to ultraviolet (UV) treatment.")
        elif st == "CONTROL":
            extras.append("The sample is from the control group.")

    if "donor_BMI" in row_dict:
        extras.append(f"The patient has a BMI of {row_dict['donor_BMI']}.")

    if "lung_condition" in row_dict:
        extras.append(f"The lung condition is described as {row_dict['lung_condition']}.")

    if "smoking_status" in row_dict:
        ss = row_dict["smoking_status"].strip().lower()
        if ss == "active":
            extras.append("The patient is an active smoker.")
        elif ss == "former":
            extras.append("The patient is a former smoker.")
        elif ss == "hist of marijuana use":
            extras.append("The patient has a history of marijuana use.")
        elif ss == "never":
            extras.append("The patient has never smoked.")

    if "BMI" in row_dict:
        extras.append(f"The patient's BMI is {row_dict['BMI']}.")
        
    if "medical_conditions" in row_dict:
        mc = row_dict["medical_conditions"].strip().lower()
        description = MEDICAL_CONDITION_MAP.get(mc)
        if description:
            extras.append(description)
            
    if "hemisphere" in row_dict:
        hemi = row_dict["hemisphere"].strip().lower()
        if hemi == "left":
            extras.append("The tissue sample was taken from the left hemisphere of the brain.")
        elif hemi == "right":
            extras.append("The tissue sample was taken from the right hemisphere of the brain.")

    # Tumor-specific field processing
    if "harm_tumor.type" in row_dict:
        tumor_type = row_dict["harm_tumor.type"].strip().lower()
        if tumor_type == "pbmc":
            extras.append("The sample is from peripheral blood mononuclear cells.")
        else:
            extras.append(f"The patient has been diagnosed with {tumor_type} cancer.")

    if "harm_tumor.site" in row_dict:
        site = row_dict["harm_tumor.site"].strip().lower()
        if site == "primary":
            extras.append("The tumor is located at the primary site.")
        elif site == "metastasis":
            extras.append("The tumor has metastasized to other parts of the body.")
        elif site == "normal":
            extras.append("This sample was collected from non-tumorous tissue.")

    if "harm_sample.type" in row_dict:
        stype = row_dict["harm_sample.type"].strip().lower()
        if stype == "tumor" and not any("non-tumorous tissue" in s for s in extras):
            extras.append("The sample is derived from tumor tissue.")
        elif stype == "normal" and not any("non-tumorous tissue" in s for s in extras):
            extras.append("The sample is derived from normal tissue.")
        elif stype == "blood":
            extras.append("The sample is a blood-derived specimen.")
        elif stype == "lymphnode":
            extras.append("The sample is derived from lymph node tissue.")

    if "harm_condition" in row_dict:
        cond = row_dict["harm_condition"].strip().replace("_", " ")
        tissue_type, status = cond.split(" ") if " " in cond else (cond, "")
        if status == "T":
            extras.append(f"The sample is from tumor tissue in the {tissue_type}.")
        elif status == "N":
            extras.append(f"The sample is from normal {tissue_type} tissue.")
        elif status == "M":
            extras.append(f"The sample is from metastatic {tissue_type} tumor.")
        else:
            extras.append(f"The overall condition is described as {cond}.")

    if "harm_cd45pos" in row_dict:
        cd45 = row_dict["harm_cd45pos"].strip().lower()
        if cd45 == "yes":
            extras.append("The cell is CD45-positive, suggesting an immune cell origin.")
        elif cd45 == "no":
            extras.append("The cell is CD45-negative, suggesting a non-immune cell lineage.")
        elif cd45 == "mixed":
            extras.append("The sample contains a mixture of CD45-positive and CD45-negative cells.")
            
    if "diabetes_history" in row_dict:
        val = row_dict["diabetes_history"].strip().lower()
        if val == "yes":
            extras.append("The patient has a history of diabetes.")
        elif val == "no":
            extras.append("The patient does not have diabetes.")

    if "hypertension" in row_dict:
        val = row_dict["hypertension"].strip().lower()
        if val == "yes":
            extras.append("The patient has a history of hypertension.")
        elif val == "no":
            extras.append("The patient does not have hypertension.")

    if "eGFR" in row_dict:
        val = row_dict["eGFR"].strip()
        if val != "unknown":
            extras.append(f"The patient's estimated glomerular filtration rate (eGFR) is in the range {val}.")
            
    if "Condition" in row_dict:
        val = row_dict["Condition"].strip().lower()
        if val in FIBROBLAST_CONDITION_MAP:
            extras.append(FIBROBLAST_CONDITION_MAP[val])
            
    if "author_tumor_subsite" in row_dict:
        subsite = row_dict["author_tumor_subsite"].strip().lower()
        extras.append(f"The tissue was collected from the {subsite}.")

    if "activation" in row_dict:
        act = row_dict["activation"].strip().lower()
        if act == "activated":
            extras.append("The sample was stimulated and represents activated immune cells.")
        elif act == "resting":
            extras.append("The sample represents resting (non-activated) immune cells.")
            
    if "Genotype" in row_dict:
        gt = row_dict["Genotype"].strip().upper()
        if gt == "FLT3-ITD,NPM1-MUT":
            extras.append("The patient carries FLT3-ITD and NPM1 mutations.")
        elif gt == "FLT3-WT,NPM1-MUT":
            extras.append("The patient carries a wild-type FLT3 and an NPM1 mutation.")
        elif gt == "APL":
            extras.append("The patient is diagnosed with acute promyelocytic leukemia (APL).")


    if extras:
        parts.append(" ".join(extras))

    # Append top-expressed genes information
    if top_genes:
        genes_str = ", ".join(top_genes)
        parts.append(f"Top expressed genes are: {genes_str}.")

    prompt_text = " ".join(parts)
    return prompt_text


def process_h5ad(h5ad_path, output_path, max_cells):
    """
    Process a single .h5ad file according to the rules:
      1) Filter to keep only cells with suspension_type 'cell'
      2) For each cell, gather top 50 genes, collect context fields, and build a prompt
      3) Save the output as JSON
    """
    print(f"[INFO] Processing {h5ad_path} ...")
    adata = anndata.read_h5ad(h5ad_path)

    # Skip if no valid 'cell' entries
    if "suspension_type" in adata.obs:
        mask_cell = adata.obs["suspension_type"] == "cell"
        if mask_cell.sum() == 0:
            print(f"[SKIP] {h5ad_path} has no 'cell' entries. Skipping.")
            return  # Skip this file entirely
        adata = adata[mask_cell, :]
    else:
        print(f"[SKIP] {h5ad_path} has no 'suspension_type' column. Skipping.")
        return  # Skip this file entirely
    
    # Randomly subsample cells if total exceeds max_cells
    if adata.n_obs > max_cells:
        sampled_indices = np.random.choice(adata.n_obs, size=max_cells, replace=False)
        adata = adata[sampled_indices, :]

    # Filter: keep only cells with 'suspension_type' equal to 'cell' (if available)
    if "suspension_type" in adata.obs:
        mask_cell = adata.obs["suspension_type"] == "cell"
        adata = adata[mask_cell, :]
    else:
        print("Warning: 'suspension_type' not found, no filtering applied.")

    # Identify the folder name to determine extra fields
    folder_name = os.path.basename(os.path.dirname(h5ad_path))
    extra_fields = EXTRA_FIELDS_MAP.get(folder_name, [])

    results = []
    for i in tqdm(range(adata.n_obs), desc=f"Cells in {folder_name}", unit="cell"):
        row = adata.obs.iloc[i]

        # Collect mandatory fields
        row_dict = {}
        for mf in MANDATORY_FIELDS:
            val = row.get(mf, None)
            if is_valid_value(val):
                row_dict[mf] = str(val)

        # Collect extra fields based on folder name
        for ef in extra_fields:
            val = row.get(ef, None)
            if is_valid_value(val):
                row_dict[ef] = str(val)

        # Retrieve the top expressed genes for this cell
        top_genes = get_top_genes(adata, i, top_n=50)

        # Optionally, get cell type if available
        cell_type_val = ""
        if "cell_type" in adata.obs:
            cval = row.get("cell_type", None)
            if is_valid_value(cval):
                cell_type_val = str(cval)

        # Build the prompt text
        prompt = build_prompt(row_dict, top_genes)

        # Create the record for this cell
        record = {
            "cell_index": adata.obs_names[i],
            "donor_id": adata.obs.donor_id.iloc[i],
            "top_expressed_genes": top_genes,
            "cell_type": cell_type_val,
            "context": prompt
        }
        if 'batch' in adata.obs.columns:
            record["batch_id"] = str(adata.obs.batch.iloc[i])
        # Include raw context fields (disease, tissue, etc.)
        for k, v in row_dict.items():
            record[k] = v

        results.append(record)

    # Save results to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[INFO] Finished processing {h5ad_path}. Output saved to {output_path}. # of cells: {len(results)}")


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

    from functools import partial
    partial_func = partial(process_file, root_input=args.input_dir, root_output=args.output_dir, max_cells=args.max_cells)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(partial_func, h5ad_files)


if __name__ == "__main__":
    main()
