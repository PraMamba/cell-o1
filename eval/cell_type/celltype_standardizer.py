"""
Cell Type Standardization Module

This module provides functionality to standardize cell type names using a predefined mapping.
It handles both batch-level (multiple cell types separated by '|') and single-cell annotations.
"""

import os
import sys
import json
from typing import List, Dict, Union, Tuple

# Add the mapping file path to Python path
MAPPING_FILE_PATH = "/home/scbjtfy/RVQ-Alpha/data_process/metadata_standard/metadata_standard_mapping.py"


def load_celltype_mapping() -> Dict[str, str]:
    """
    Load the CELL_TYPE_MAPPING from the standardization file.
    
    Returns:
        Dictionary mapping original cell type names to standardized names
    """
    try:
        # Import the mapping dynamically
        import importlib.util
        spec = importlib.util.spec_from_file_location("metadata_standard_mapping", MAPPING_FILE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        cell_type_mapping = getattr(module, "CELL_TYPE_MAPPING", {})
        print(f"[INFO] Loaded {len(cell_type_mapping)} cell type mappings")
        return cell_type_mapping
    
    except Exception as e:
        print(f"[ERROR] Failed to load cell type mapping: {e}")
        return {}


class CellTypeStandardizer:
    """Class for standardizing cell type names using predefined mapping."""
    
    def __init__(self, mapping_file_path: str = MAPPING_FILE_PATH):
        """
        Initialize the standardizer with a mapping file.
        
        Args:
            mapping_file_path: Path to the Python file containing CELL_TYPE_MAPPING
        """
        self.mapping_file_path = mapping_file_path
        self.cell_type_mapping = load_celltype_mapping()
        self.unmapped_types = set()
        
    def standardize_single_celltype(self, cell_type: str) -> Tuple[str, bool]:
        """
        Standardize a single cell type name.
        
        Args:
            cell_type: Original cell type name
            
        Returns:
            Tuple of (standardized_name, is_mapped)
            - standardized_name: The mapped name if found, otherwise original name
            - is_mapped: True if mapping was found, False otherwise
        """
        cell_type = cell_type.strip()
        
        if not cell_type:
            return "", False
        
        # Try exact match (case-sensitive first)
        if cell_type in self.cell_type_mapping:
            return self.cell_type_mapping[cell_type], True
        
        # Try case-insensitive match
        for key, value in self.cell_type_mapping.items():
            if key.lower() == cell_type.lower():
                return value, True
        
        # No mapping found
        self.unmapped_types.add(cell_type)
        return cell_type, False
    
    def standardize_batch_celltype(self, batch_answer: str) -> Tuple[str, List[str]]:
        """
        Standardize a batch answer containing multiple cell types separated by '|'.
        
        Args:
            batch_answer: String containing multiple cell types separated by ' | '
            
        Returns:
            Tuple of (standardized_batch, unmapped_list)
            - standardized_batch: Standardized answer with ' | ' separator
            - unmapped_list: List of cell types that couldn't be mapped
        """
        if not batch_answer or not batch_answer.strip():
            return "", []
        
        # Split by '|' and process each cell type
        cell_types = [ct.strip() for ct in batch_answer.split('|')]
        standardized_types = []
        unmapped = []
        
        for cell_type in cell_types:
            standardized, is_mapped = self.standardize_single_celltype(cell_type)
            standardized_types.append(standardized)
            
            if not is_mapped:
                unmapped.append(cell_type)
        
        standardized_batch = ' | '.join(standardized_types)
        return standardized_batch, unmapped
    
    def get_unmapped_types(self) -> List[str]:
        """
        Get list of cell types that couldn't be mapped.
        
        Returns:
            Sorted list of unmapped cell type names
        """
        return sorted(list(self.unmapped_types))
    
    def reset_unmapped_tracker(self):
        """Reset the tracker for unmapped cell types."""
        self.unmapped_types = set()


def extract_dataset_id_from_path(file_path: str) -> str:
    """
    Extract dataset ID from file path.
    
    Args:
        file_path: Path to the input file
        
    Returns:
        Dataset ID (e.g., 'A013', 'D099') or 'unknown'
    """
    # Common patterns: /A013/, /D099/, etc.
    import re
    match = re.search(r'/([A-Z]\d{3,})/', file_path)
    if match:
        return match.group(1)
    return "unknown"


def save_unmapped_report(
    unmapped_data: List[Dict],
    output_dir: str,
    task_variant: str,
    timestamp: str
):
    """
    Save a report of unmapped cell types.
    
    Args:
        unmapped_data: List of dictionaries containing unmapped cell type information
        output_dir: Directory to save the report
        task_variant: Task variant name (e.g., 'batch_constrained')
        timestamp: Timestamp string for filename
    """
    if not unmapped_data:
        print(f"[INFO] No unmapped cell types found for {task_variant}")
        return
    
    unmapped_file = os.path.join(
        output_dir,
        f"{task_variant}_unmapped_celltypes_{timestamp}.json"
    )
    
    # Create summary
    unmapped_summary = {
        "task_variant": task_variant,
        "total_unmapped_instances": len(unmapped_data),
        "unique_unmapped_types": list(set([
            item["original_type"]
            for item in unmapped_data
        ])),
        "details": unmapped_data
    }
    
    with open(unmapped_file, 'w', encoding='utf-8') as f:
        json.dump(unmapped_summary, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Unmapped cell types report saved to: {unmapped_file}")
    print(f"[INFO] Total unmapped instances: {len(unmapped_data)}")
    print(f"[INFO] Unique unmapped types: {len(unmapped_summary['unique_unmapped_types'])}")


if __name__ == "__main__":
    # Test the standardizer
    standardizer = CellTypeStandardizer()
    
    # Test single cell type
    print("\n=== Testing Single Cell Type Standardization ===")
    test_types = [
        "CD8-positive, alpha-beta T cell",
        "natural killer cell",
        "B cell",
        "Unknown Type"  # This should not be mapped
    ]
    
    for test_type in test_types:
        standardized, is_mapped = standardizer.standardize_single_celltype(test_type)
        print(f"{test_type} -> {standardized} (mapped: {is_mapped})")
    
    # Test batch cell types
    print("\n=== Testing Batch Cell Type Standardization ===")
    batch_answer = "CD8-positive, alpha-beta T cell | natural killer cell | B cell | Unknown Type"
    standardized_batch, unmapped = standardizer.standardize_batch_celltype(batch_answer)
    print(f"Original: {batch_answer}")
    print(f"Standardized: {standardized_batch}")
    print(f"Unmapped: {unmapped}")
    
    # Show all unmapped types
    print("\n=== All Unmapped Types ===")
    print(standardizer.get_unmapped_types())

