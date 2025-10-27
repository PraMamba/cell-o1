"""
Cell Type Mapping and Standardization Module

This module provides standardized cell type names to handle synonyms,
terminology inconsistencies, and similar naming variations across datasets.

Usage:
    from celltype_mapping import standardize_celltype, get_dataset_mapping

    standardized = standardize_celltype("alpha-beta memory T cell", dataset="A013")
"""

from typing import Dict, List, Optional


# ============================================
# Cell Type Mapping Dictionaries
# ============================================

# A013 Dataset: Immune cells
CELLTYPE_MAPPING_A013 = [
    {"raw_name": "alpha-beta memory T cell", "standardized_name": "Alpha-beta memory T cell", "broad_category": "T cell"},
    {"raw_name": "alpha-beta T cell", "standardized_name": "Alpha-beta T cell", "broad_category": "T cell"},
    {"raw_name": "CD14-low", "standardized_name": "CD14-low monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD14-low, CD16-positive monocyte", "standardized_name": "CD14-low CD16+ monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD14-positive monocyte", "standardized_name": "CD14+ monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD16-negative", "standardized_name": "CD16-negative monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD16-negative classical monocyte", "standardized_name": "CD16-negative classical monocyte", "broad_category": "Monocyte"},
    {"raw_name": "CD16-negative, CD56-bright natural killer cell, human", "standardized_name": "CD16-negative CD56-bright NK cell", "broad_category": "NK cell"},
    {"raw_name": "CD56-dim natural killer cell", "standardized_name": "CD56^dim NK cell", "broad_category": "NK cell"},
    {"raw_name": "central memory CD4-positive", "standardized_name": "Central memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "central memory CD4-positive, alpha-beta T cell", "standardized_name": "Central memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "central memory CD8-positive, alpha-beta T cell", "standardized_name": "Central memory CD8+ T cell", "broad_category": "T cell"},
    {"raw_name": "conventional dendritic cell", "standardized_name": "Conventional dendritic cell", "broad_category": "Dendritic cell"},
    {"raw_name": "effector memory CD4-positive", "standardized_name": "Effector memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "effector memory CD4-positive, alpha-beta T cell", "standardized_name": "Effector memory CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "effector memory CD8-positive", "standardized_name": "Effector memory CD8+ T cell", "broad_category": "T cell"},
    {"raw_name": "effector memory CD8-positive, alpha-beta T cell", "standardized_name": "Effector memory CD8+ T cell", "broad_category": "T cell"},
    {"raw_name": "gamma-delta T cell", "standardized_name": "Gamma-delta T cell", "broad_category": "T cell"},
    {"raw_name": "hematopoietic precursor cell", "standardized_name": "Hematopoietic precursor cell", "broad_category": "Progenitor"},
    {"raw_name": "memory B cell", "standardized_name": "Memory B cell", "broad_category": "B cell"},
    {"raw_name": "mucosal invariant T cell", "standardized_name": "MAIT cell", "broad_category": "T cell"},
    {"raw_name": "naive B cell", "standardized_name": "Naive B cell", "broad_category": "B cell"},
    {"raw_name": "naive thymus-derived CD4-positive, alpha-beta T cell", "standardized_name": "Naive CD4+ T cell", "broad_category": "T cell"},
    {"raw_name": "naive thymus-derived CD8-positive, alpha-beta T cell", "standardized_name": "Naive CD8+ T cell", "broad_category": "T cell"},
    {"raw_name": "natural killer cell", "standardized_name": "NK cell", "broad_category": "NK cell"},
    {"raw_name": "plasmablast", "standardized_name": "Plasmablast", "broad_category": "B cell"},
    {"raw_name": "plasmacytoid dendritic cell", "standardized_name": "Plasmacytoid dendritic cell", "broad_category": "Dendritic cell"},
    {"raw_name": "regulatory T cell", "standardized_name": "Regulatory T cell", "broad_category": "T cell"},
    {"raw_name": "transitional stage B cell", "standardized_name": "Transitional B cell", "broad_category": "B cell"},
]

# D099 Dataset: Epithelial cells (Predicted column)
CELLTYPE_MAPPING_D099 = [
    {"raw_name": "basal cell", "standardized_name": "Basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epidermis", "standardized_name": "Epidermal basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epithelium of bronchus", "standardized_name": "Bronchial basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of epithelium of trachea", "standardized_name": "Tracheal basal cell", "broad_category": "Epithelial"},
    {"raw_name": "basal cell of prostate epithelium", "standardized_name": "Prostate basal cell", "broad_category": "Epithelial"},
    {"raw_name": "blood vessel endothelial cell", "standardized_name": "Vascular endothelial cell", "broad_category": "Endothelial"},
    {"raw_name": "brush cell", "standardized_name": "Brush cell", "broad_category": "Epithelial"},
    {"raw_name": "cell in vitro", "standardized_name": "Cultured cell", "broad_category": "Other"},
    {"raw_name": "ciliated cell", "standardized_name": "Ciliated epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "ciliated cell of the bronchus", "standardized_name": "Bronchial ciliated cell", "broad_category": "Epithelial"},
    {"raw_name": "club cell", "standardized_name": "Club cell", "broad_category": "Epithelial"},
    {"raw_name": "corneal epithelial cell", "standardized_name": "Corneal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "cultured cell", "standardized_name": "Cultured cell", "broad_category": "Other"},
    {"raw_name": "duct epithelial cell", "standardized_name": "Ductal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "endothelial cell of lymphatic vessel", "standardized_name": "Lymphatic endothelial cell", "broad_category": "Endothelial"},
    {"raw_name": "endothelial cell of umbilical vein", "standardized_name": "Umbilical vein endothelial cell (HUVEC)", "broad_category": "Endothelial"},
    {"raw_name": "epithelial cell", "standardized_name": "Epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of esophagus", "standardized_name": "Esophageal epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of lower respiratory tract", "standardized_name": "Lower respiratory epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of lung", "standardized_name": "Pulmonary epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of nephron", "standardized_name": "Nephron epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of proximal tubule", "standardized_name": "Proximal tubule epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of the bronchus", "standardized_name": "Bronchial epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "epithelial cell of urethra", "standardized_name": "Urethral epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "foveolar cell of stomach", "standardized_name": "Gastric foveolar cell", "broad_category": "Epithelial"},
    {"raw_name": "goblet cell", "standardized_name": "Goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "granulocyte", "standardized_name": "Granulocyte", "broad_category": "Immune"},
    {"raw_name": "kidney collecting duct principal cell", "standardized_name": "Collecting duct principal cell", "broad_category": "Epithelial"},
    {"raw_name": "kidney loop of Henle thin ascending limb epithelial cell", "standardized_name": "Loop of Henle thin ascending limb cell", "broad_category": "Epithelial"},
    {"raw_name": "kidney loop of Henle thin descending limb epithelial cell", "standardized_name": "Loop of Henle thin descending limb cell", "broad_category": "Epithelial"},
    {"raw_name": "lung ciliated cell", "standardized_name": "Pulmonary ciliated cell", "broad_category": "Epithelial"},
    {"raw_name": "lung goblet cell", "standardized_name": "Pulmonary goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "lung secretory cell", "standardized_name": "Pulmonary secretory cell", "broad_category": "Epithelial"},
    {"raw_name": "malignant cell", "standardized_name": "Malignant cell", "broad_category": "Other"},
    {"raw_name": "melanocyte", "standardized_name": "Melanocyte", "broad_category": "Pigment"},
    {"raw_name": "nasal mucosa goblet cell", "standardized_name": "Nasal goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "native cell", "standardized_name": "Native cell", "broad_category": "Other"},
    {"raw_name": "neutrophil", "standardized_name": "Neutrophil", "broad_category": "Immune"},
    {"raw_name": "respiratory basal cell", "standardized_name": "Respiratory basal cell", "broad_category": "Epithelial"},
    {"raw_name": "secretory cell", "standardized_name": "Secretory epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "squamous epithelial cell", "standardized_name": "Squamous epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "stratified epithelial cell", "standardized_name": "Stratified epithelial cell", "broad_category": "Epithelial"},
    {"raw_name": "tracheal goblet cell", "standardized_name": "Tracheal goblet cell", "broad_category": "Epithelial"},
    {"raw_name": "type II pneumocyte", "standardized_name": "Type II pneumocyte (AT2 cell)", "broad_category": "Epithelial"},
]

# D099 Dataset: State labels (GroundTruth column)
STATE_MAPPING_D099 = [
    {"raw_name": "Basal", "standardized_name": "Basal"},
    {"raw_name": "Ciliated", "standardized_name": "Ciliated"},
    {"raw_name": "Differentiating.Basal", "standardized_name": "Differentiating basal"},
    {"raw_name": "Proliferating.Basal", "standardized_name": "Proliferating basal"},
    {"raw_name": "Secretory", "standardized_name": "Secretory"},
    {"raw_name": "Suprabasal", "standardized_name": "Suprabasal"},
    {"raw_name": "Transitioning.Basal", "standardized_name": "Transitioning basal"},
]


def build_mapping_dict(mapping_list: List[Dict]) -> Dict[str, str]:
    """
    Build a dictionary for fast lookup from raw names to standardized names.
    Supports case-insensitive matching.
    """
    mapping_dict = {}
    for item in mapping_list:
        raw_lower = item["raw_name"].lower().strip()
        mapping_dict[raw_lower] = item["standardized_name"]
    return mapping_dict


# Build lookup dictionaries
_MAPPING_A013 = build_mapping_dict(CELLTYPE_MAPPING_A013)
_MAPPING_D099_CELLTYPE = build_mapping_dict(CELLTYPE_MAPPING_D099)
_MAPPING_D099_STATE = build_mapping_dict(STATE_MAPPING_D099)


def standardize_celltype(raw_name: str, dataset: str = "A013", mapping_type: str = "celltype") -> str:
    """
    Standardize a cell type name according to the dataset mapping.

    Args:
        raw_name: Original cell type name
        dataset: Dataset identifier ("A013" or "D099")
        mapping_type: Type of mapping ("celltype" or "state", only relevant for D099)

    Returns:
        Standardized cell type name. If no mapping found, returns the original name.
    """
    if not raw_name or not isinstance(raw_name, str):
        return raw_name

    raw_lower = raw_name.lower().strip()

    # Select appropriate mapping
    if dataset.upper() == "A013":
        mapping = _MAPPING_A013
    elif dataset.upper() == "D099":
        if mapping_type == "state":
            mapping = _MAPPING_D099_STATE
        else:
            mapping = _MAPPING_D099_CELLTYPE
    else:
        # Unknown dataset, return original
        return raw_name

    # Look up standardized name
    standardized = mapping.get(raw_lower)

    if standardized:
        return standardized
    else:
        # No mapping found, return original name
        return raw_name


def get_dataset_mapping(dataset: str = "A013", mapping_type: str = "celltype") -> List[Dict]:
    """
    Get the full mapping list for a dataset.

    Args:
        dataset: Dataset identifier ("A013" or "D099")
        mapping_type: Type of mapping ("celltype" or "state", only relevant for D099)

    Returns:
        List of mapping dictionaries
    """
    if dataset.upper() == "A013":
        return CELLTYPE_MAPPING_A013
    elif dataset.upper() == "D099":
        if mapping_type == "state":
            return STATE_MAPPING_D099
        else:
            return CELLTYPE_MAPPING_D099
    else:
        return []


def get_broad_category(raw_name: str, dataset: str = "A013") -> Optional[str]:
    """
    Get the broad category for a cell type.

    Args:
        raw_name: Original cell type name
        dataset: Dataset identifier ("A013" or "D099")

    Returns:
        Broad category name, or None if not found
    """
    mapping_list = get_dataset_mapping(dataset, "celltype")
    raw_lower = raw_name.lower().strip()

    for item in mapping_list:
        if item["raw_name"].lower() == raw_lower:
            return item.get("broad_category")

    return None


if __name__ == "__main__":
    # Test examples
    print("Testing cell type standardization:")
    print("-" * 60)

    # A013 examples
    test_cases_a013 = [
        "alpha-beta memory T cell",
        "CD14-positive monocyte",
        "natural killer cell",
        "mucosal invariant T cell"
    ]

    print("\nA013 Dataset:")
    for raw in test_cases_a013:
        std = standardize_celltype(raw, dataset="A013")
        cat = get_broad_category(raw, dataset="A013")
        print(f"  {raw:40} -> {std:40} [{cat}]")

    # D099 examples
    test_cases_d099 = [
        "epithelial cell of lung",
        "type II pneumocyte",
        "lung ciliated cell",
        "basal cell of epithelium of bronchus"
    ]

    print("\nD099 Dataset (Cell Types):")
    for raw in test_cases_d099:
        std = standardize_celltype(raw, dataset="D099", mapping_type="celltype")
        cat = get_broad_category(raw, dataset="D099")
        print(f"  {raw:40} -> {std:40} [{cat}]")

    # D099 state examples
    test_cases_state = [
        "Differentiating.Basal",
        "Proliferating.Basal",
        "Secretory"
    ]

    print("\nD099 Dataset (State Labels):")
    for raw in test_cases_state:
        std = standardize_celltype(raw, dataset="D099", mapping_type="state")
        print(f"  {raw:40} -> {std}")
