#!/usr/bin/env python3
"""
Cell Type Prediction Performance Visualization
===============================================

Generate comprehensive visualizations for cell type prediction model performance
based on LLM judgment results.

Usage:
    python visualize_performance.py \
        --judged_results_path /path/to/celltype_judged_results.json \
        --output_dir /path/to/output \
        [--plots plot1 plot2 ...]
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Color mapping for semantic relations
SEMANTIC_COLORS = {
    'equivalent': '#2ecc71',      # Green
    'parent-child': '#3498db',    # Blue
    'same_major_lineage': '#f39c12',  # Orange/Yellow
    'different': '#e74c3c',       # Red
    'ambiguous': '#95a5a6',       # Gray
    'other': '#9b59b6'            # Purple
}

SEMANTIC_ORDER = ['equivalent', 'parent-child', 'same_major_lineage', 'different', 'ambiguous', 'other']


def load_judged_results(results_path: str) -> List[Dict]:
    """Load LLM judged results from JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} judged results from {results_path}")
    return data


def prepare_dataframe(judged_results: List[Dict]) -> pd.DataFrame:
    """
    Convert judged results to a pandas DataFrame for easier analysis.
    
    Args:
        judged_results: List of dictionaries containing judgment results
        
    Returns:
        DataFrame with columns: model_name, dataset_id, ground_truth, predicted_answer,
                                score, semantic_relation, group, etc.
    """
    rows = []
    
    for result in judged_results:
        # Extract basic fields with defaults
        model_name = result.get('model_name', 'unknown')
        dataset_id = result.get('dataset_id', 'unknown')
        ground_truth = result.get('ground_truth', '')
        predicted_answer = result.get('predicted_answer', '')
        group = result.get('group', '')
        
        # Extract LLM judgment
        llm_judgment = result.get('llm_judgment', {})
        if not isinstance(llm_judgment, dict):
            llm_judgment = {}
        
        score = llm_judgment.get('score', 0.0)
        semantic_relation = llm_judgment.get('semantic_relation', 'unknown')
        
        # Skip if essential fields are missing
        if not ground_truth:
            continue
        
        rows.append({
            'model_name': str(model_name),
            'dataset_id': str(dataset_id),
            'ground_truth': str(ground_truth),
            'predicted_answer': str(predicted_answer) if predicted_answer else '',
            'score': float(score) if score is not None else 0.0,
            'semantic_relation': str(semantic_relation),
            'group': str(group),
            'index': result.get('index', -1)
        })
    
    if len(rows) == 0:
        logging.warning("No valid rows found in data")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    logging.info(f"Created DataFrame with {len(df)} rows")
    logging.info(f"Unique models: {df['model_name'].nunique()}")
    logging.info(f"Unique datasets: {df['dataset_id'].nunique()}")
    logging.info(f"Unique cell types: {df['ground_truth'].nunique()}")
    
    return df


def plot_score_distribution(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (12, 6)):
    """
    Plot 1: Model Performance Distribution (Violin Plot / Box Plot)
    
    Shows score distribution for each model.
    """
    logging.info("Generating score distribution plot...")
    
    if df.empty or 'score' not in df.columns:
        logging.warning("No data for score distribution plot. Skipping.")
        return
    
    # Filter out invalid scores
    df_clean = df[df['score'].notna()].copy()
    if df_clean.empty:
        logging.warning("No valid scores found. Skipping.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Violin plot
    ax1 = axes[0]
    models = df_clean['model_name'].unique()
    if len(models) == 0:
        logging.warning("No models found. Skipping.")
        return
    
    data_to_plot = [df_clean[df_clean['model_name'] == model]['score'].values for model in models]
    
    parts = ax1.violinplot(data_to_plot, positions=range(len(models)), showmeans=True, showmedians=True)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_title('Score Distribution by Model (Violin Plot)', fontsize=14, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Color violins
    for pc in parts['bodies']:
        pc.set_facecolor('#3498db')
        pc.set_alpha(0.7)
    
    # Box plot
    ax2 = axes[1]
    sns.boxplot(data=df_clean, x='model_name', y='score', ax=ax2, palette='Set2')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_title('Score Distribution by Model (Box Plot)', fontsize=14, fontweight='bold')
    ax2.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '1_score_distribution.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_semantic_relation_breakdown(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (12, 8)):
    """
    Plot 2: Semantic Relation Breakdown (100% Stacked Bar Chart)
    
    Shows the proportion of each semantic relation type for each model.
    """
    logging.info("Generating semantic relation breakdown plot...")
    
    # Calculate percentages
    model_relation_counts = df.groupby(['model_name', 'semantic_relation']).size().reset_index(name='count')
    model_totals = df.groupby('model_name').size().reset_index(name='total')
    model_relation_counts = model_relation_counts.merge(model_totals, on='model_name')
    model_relation_counts['percentage'] = (model_relation_counts['count'] / model_relation_counts['total']) * 100
    
    # Pivot for plotting
    pivot_df = model_relation_counts.pivot(index='model_name', columns='semantic_relation', values='percentage').fillna(0)
    
    # Reorder columns according to SEMANTIC_ORDER
    existing_relations = [r for r in SEMANTIC_ORDER if r in pivot_df.columns]
    pivot_df = pivot_df[existing_relations]
    
    # Create color mapping
    colors = [SEMANTIC_COLORS.get(rel, '#95a5a6') for rel in existing_relations]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
    
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Semantic Relation Breakdown by Model', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Semantic Relation', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_semantic_relation_breakdown.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_performance_by_celltype(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (14, 10)):
    """
    Plot 3: Performance by Cell Type (Sorted Bar Chart)
    
    Shows average score for each cell type.
    """
    logging.info("Generating performance by cell type plot...")
    
    # Filter valid data
    df_clean = df[df['ground_truth'].notna() & df['score'].notna()].copy()
    if df_clean.empty:
        logging.warning("No valid data for cell type performance plot. Skipping.")
        return
    
    # Calculate mean scores by cell type
    celltype_scores = df_clean.groupby('ground_truth')['score'].agg(['mean', 'std', 'count']).reset_index()
    celltype_scores = celltype_scores.sort_values('mean', ascending=False)
    
    if len(celltype_scores) == 0:
        logging.warning("No cell types found. Skipping.")
        return
    
    # Limit to top 50 cell types if too many
    max_celltypes = 50
    if len(celltype_scores) > max_celltypes:
        logging.info(f"Too many cell types ({len(celltype_scores)}), showing top {max_celltypes}")
        celltype_scores = celltype_scores.head(max_celltypes)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Top plot: Mean scores
    ax1 = axes[0]
    # Handle NaN std values
    std_values = celltype_scores['std'].fillna(0)
    bars = ax1.barh(range(len(celltype_scores)), celltype_scores['mean'], 
                    xerr=std_values, capsize=3, color='#3498db', alpha=0.7)
    ax1.set_yticks(range(len(celltype_scores)))
    ax1.set_yticklabels(celltype_scores['ground_truth'], fontsize=9)
    ax1.set_xlabel('Mean Score', fontsize=12)
    ax1.set_title('Average Performance by Cell Type', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(celltype_scores.iterrows()):
        ax1.text(row['mean'] + 0.02, i, f"{row['mean']:.3f}", 
                va='center', fontsize=8)
    
    # Bottom plot: Sample counts
    ax2 = axes[1]
    ax2.barh(range(len(celltype_scores)), celltype_scores['count'], color='#95a5a6', alpha=0.7)
    ax2.set_yticks(range(len(celltype_scores)))
    ax2.set_yticklabels(celltype_scores['ground_truth'], fontsize=9)
    ax2.set_xlabel('Sample Count', fontsize=12)
    ax2.set_title('Sample Count by Cell Type', fontsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3_performance_by_celltype.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_model_celltype_heatmap(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (14, 12)):
    """
    Plot 4: Model vs. Cell Type Heatmap
    
    Shows average score for each model-cell type combination.
    """
    logging.info("Generating model-cell type heatmap...")
    
    # Filter valid data
    df_clean = df[df['ground_truth'].notna() & df['score'].notna()].copy()
    if df_clean.empty:
        logging.warning("No valid data for heatmap. Skipping.")
        return
    
    # Calculate mean scores
    heatmap_data = df_clean.groupby(['model_name', 'ground_truth'])['score'].mean().reset_index()
    
    if len(heatmap_data) == 0:
        logging.warning("No data after grouping. Skipping.")
        return
    
    pivot_df = heatmap_data.pivot(index='ground_truth', columns='model_name', values='score')
    
    if pivot_df.empty:
        logging.warning("Pivot table is empty. Skipping.")
        return
    
    # Sort by mean score across all models
    celltype_means = pivot_df.mean(axis=1).sort_values(ascending=False)
    pivot_df = pivot_df.loc[celltype_means.index]
    
    # Limit cell types if too many
    max_celltypes = 50
    if len(pivot_df) > max_celltypes:
        logging.info(f"Too many cell types ({len(pivot_df)}), showing top {max_celltypes}")
        pivot_df = pivot_df.head(max_celltypes)
    
    # Calculate figure size dynamically
    n_rows = len(pivot_df)
    n_cols = len(pivot_df.columns)
    adjusted_height = max(8, min(20, n_rows * 0.4))
    adjusted_width = max(10, min(20, n_cols * 1.2))
    figsize = (adjusted_width, adjusted_height)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                vmin=0, vmax=1, cbar_kws={'label': 'Mean Score'}, ax=ax,
                linewidths=0.5, linecolor='white', annot_kws={'fontsize': 8})
    
    ax.set_ylabel('Cell Type (Ground Truth)', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_title('Model Performance Heatmap by Cell Type', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4_model_celltype_heatmap.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def detect_cycles_in_graph(flow_counts: pd.DataFrame) -> bool:
    """
    Detect if there are cycles in the directed graph.
    A cycle exists if there's a path from A->B and B->A (direct or indirect).
    
    Args:
        flow_counts: DataFrame with columns 'ground_truth', 'predicted_answer'
        
    Returns:
        True if cycles detected, False otherwise
    """
    # Build adjacency list
    graph = defaultdict(set)
    for _, row in flow_counts.iterrows():
        source = row['ground_truth']
        target = row['predicted_answer']
        graph[source].add(target)
    
    # Check for direct cycles (A->B and B->A)
    for source, targets in graph.items():
        for target in targets:
            if target in graph and source in graph[target]:
                return True
    
    # Check for indirect cycles using DFS
    visited = set()
    rec_stack = set()
    
    def has_cycle(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if node not in visited:
            if has_cycle(node):
                return True
    
    return False


def plot_sankey_diagram(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (16, 12)):
    """
    Plot 5: Raw Mapping Sankey Diagram
    
    Shows flow from ground_truth to predicted_answer with semantic relation coloring.
    Uses Plotly by default (supports cyclic graphs), falls back to HoloViews if no cycles detected.
    """
    logging.info("Generating Sankey diagram...")
    
    # Count occurrences
    flow_counts = df.groupby(['ground_truth', 'predicted_answer', 'semantic_relation']).size().reset_index(name='count')
    flow_counts = flow_counts.sort_values('count', ascending=False)
    
    # Check for cycles - Plotly supports cycles, HoloViews doesn't
    has_cycles = detect_cycles_in_graph(flow_counts)
    
    if has_cycles:
        logging.info("Cycles detected in graph. Using Plotly (supports cyclic graphs).")
        use_plotly = True
    else:
        logging.info("No cycles detected. Can use HoloViews or Plotly.")
        use_plotly = False
    
    # Try Plotly first (supports cycles) or HoloViews (if no cycles)
    if use_plotly:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            # Get unique labels
            all_gt = sorted(df['ground_truth'].unique())
            all_pred = sorted(df['predicted_answer'].unique())
            
            # Create label list (GT first, then predictions)
            all_labels = all_gt + [p for p in all_pred if p not in all_gt]
            label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
            
            # Create source, target, value, and color lists
            source = []
            target = []
            value = []
            color = []
            
            for _, row in flow_counts.iterrows():
                gt_idx = label_to_idx[row['ground_truth']]
                pred_idx = label_to_idx[row['predicted_answer']]
                
                source.append(gt_idx)
                target.append(pred_idx)
                value.append(row['count'])
                
                # Color based on semantic relation
                rel = row['semantic_relation']
                color.append(SEMANTIC_COLORS.get(rel, '#95a5a6'))
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=all_labels,
                    color="lightblue"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=color
                )
            )])
            
            fig.update_layout(
                title_text="Ground Truth to Predicted Answer Mapping (Sankey Diagram)",
                font_size=12,
                width=figsize[0] * 100,
                height=figsize[1] * 100
            )
            
            output_path = os.path.join(output_dir, '5_sankey_diagram.html')
            plot(fig, filename=output_path, auto_open=False)
            logging.info(f"Saved: {output_path}")
            return
            
        except ImportError:
            logging.warning("Plotly not installed. Trying HoloViews...")
            use_plotly = False
    
    # Try HoloViews (only if no cycles)
    if not use_plotly:
        try:
            import holoviews as hv
            
            # Set Bokeh backend
            hv.extension('bokeh')
            
            # Prepare data for HoloViews Sankey
            # HoloViews Sankey expects: source, target, value columns
            sankey_data = pd.DataFrame({
                'source': flow_counts['ground_truth'].values,
                'target': flow_counts['predicted_answer'].values,
                'value': flow_counts['count'].values,
                'semantic_relation': flow_counts['semantic_relation'].values
            })
            
            # Create Sankey diagram
            sankey = hv.Sankey(
                sankey_data,
                kdims=['source', 'target'],
                vdims=['value', 'semantic_relation'],
                label='Ground Truth → Predicted Answer Mapping'
            )
            
            # Configure visualization options
            sankey = sankey.opts(
                width=figsize[0] * 100,
                height=figsize[1] * 100,
                node_width=30,
                node_padding=0.1,
                edge_color='semantic_relation',
                cmap='Category20',
                title='Ground Truth to Predicted Answer Mapping (Sankey Diagram)',
                toolbar='above',
                tools=['hover', 'pan', 'box_zoom', 'wheel_zoom', 'reset', 'save'],
                show_legend=True,
                edge_line_width=2,
                node_color='lightblue'
            )
            
            # Save as HTML
            output_path = os.path.join(output_dir, '5_sankey_diagram.html')
            hv.save(sankey, output_path, fmt='html')
            logging.info(f"Saved: {output_path}")
            return
            
        except ImportError:
            logging.warning("HoloViews not installed. Trying Plotly...")
            # Fallback to Plotly
            try:
                import plotly.graph_objects as go
                from plotly.offline import plot
                
                # Get unique labels
                all_gt = sorted(df['ground_truth'].unique())
                all_pred = sorted(df['predicted_answer'].unique())
                
                # Create label list (GT first, then predictions)
                all_labels = all_gt + [p for p in all_pred if p not in all_gt]
                label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
                
                # Create source, target, value, and color lists
                source = []
                target = []
                value = []
                color = []
                
                for _, row in flow_counts.iterrows():
                    gt_idx = label_to_idx[row['ground_truth']]
                    pred_idx = label_to_idx[row['predicted_answer']]
                    
                    source.append(gt_idx)
                    target.append(pred_idx)
                    value.append(row['count'])
                    
                    # Color based on semantic relation
                    rel = row['semantic_relation']
                    color.append(SEMANTIC_COLORS.get(rel, '#95a5a6'))
                
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_labels,
                        color="lightblue"
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=color
                    )
                )])
                
                fig.update_layout(
                    title_text="Ground Truth to Predicted Answer Mapping (Sankey Diagram)",
                    font_size=12,
                    width=figsize[0] * 100,
                    height=figsize[1] * 100
                )
                
                output_path = os.path.join(output_dir, '5_sankey_diagram.html')
                plot(fig, filename=output_path, auto_open=False)
                logging.info(f"Saved: {output_path}")
                return
                
            except ImportError:
                logging.warning("Neither HoloViews nor Plotly installed. Using simplified matplotlib version.")
                plot_sankey_simplified(df, output_dir, figsize)
                return
        except RecursionError as e:
            # HoloViews doesn't support cyclic graphs
            logging.warning(f"HoloViews error (likely cyclic graph): {e}")
            logging.warning("Falling back to Plotly which supports cyclic graphs...")
            try:
                import plotly.graph_objects as go
                from plotly.offline import plot
                
                # Get unique labels
                all_gt = sorted(df['ground_truth'].unique())
                all_pred = sorted(df['predicted_answer'].unique())
                
                # Create label list (GT first, then predictions)
                all_labels = all_gt + [p for p in all_pred if p not in all_gt]
                label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
                
                # Create source, target, value, and color lists
                source = []
                target = []
                value = []
                color = []
                
                for _, row in flow_counts.iterrows():
                    gt_idx = label_to_idx[row['ground_truth']]
                    pred_idx = label_to_idx[row['predicted_answer']]
                    
                    source.append(gt_idx)
                    target.append(pred_idx)
                    value.append(row['count'])
                    
                    # Color based on semantic relation
                    rel = row['semantic_relation']
                    color.append(SEMANTIC_COLORS.get(rel, '#95a5a6'))
                
                # Create Sankey diagram
                fig = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=all_labels,
                        color="lightblue"
                    ),
                    link=dict(
                        source=source,
                        target=target,
                        value=value,
                        color=color
                    )
                )])
                
                fig.update_layout(
                    title_text="Ground Truth to Predicted Answer Mapping (Sankey Diagram)",
                    font_size=12,
                    width=figsize[0] * 100,
                    height=figsize[1] * 100
                )
                
                output_path = os.path.join(output_dir, '5_sankey_diagram.html')
                plot(fig, filename=output_path, auto_open=False)
                logging.info(f"Saved: {output_path}")
                return
                
            except ImportError:
                logging.warning("Plotly not installed. Using simplified matplotlib version.")
                plot_sankey_simplified(df, output_dir, figsize)
        except Exception as e:
            logging.error(f"Error generating Sankey diagram: {e}", exc_info=True)
            logging.warning("Falling back to simplified matplotlib version.")
            plot_sankey_simplified(df, output_dir, figsize)


def plot_sankey_simplified(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (16, 12)):
    """Simplified Sankey diagram using matplotlib (fallback)."""
    logging.info("Generating simplified Sankey diagram (matplotlib fallback)...")
    
    # Get top flows
    flow_counts = df.groupby(['ground_truth', 'predicted_answer', 'semantic_relation']).size().reset_index(name='count')
    flow_counts = flow_counts.sort_values('count', ascending=False).head(50)  # Top 50 flows
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # This is a simplified representation - showing top flows as a network-style plot
    y_pos = 0
    for _, row in flow_counts.iterrows():
        rel = row['semantic_relation']
        color = SEMANTIC_COLORS.get(rel, '#95a5a6')
        
        ax.text(0.1, 1 - y_pos * 0.02, f"{row['ground_truth']} → {row['predicted_answer']}", 
                fontsize=8, color=color, weight='bold' if rel == 'different' else 'normal')
        ax.text(0.8, 1 - y_pos * 0.02, f"Count: {row['count']}", fontsize=8)
        y_pos += 1
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Top Ground Truth → Predicted Answer Flows (Simplified)', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '5_sankey_diagram_simplified.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_top_different_errors(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (12, 10)):
    """
    Plot 6: Top "Different" Errors Treemap
    
    Shows the most frequent wrong predictions (semantic_relation == "different").
    """
    logging.info("Generating top different errors plot...")
    
    # Filter for "different" errors
    different_errors = df[df['semantic_relation'] == 'different'].copy()
    
    if len(different_errors) == 0:
        logging.warning("No 'different' errors found. Skipping this plot.")
        return
    
    # Count error pairs
    error_counts = different_errors.groupby(['ground_truth', 'predicted_answer']).size().reset_index(name='count')
    error_counts = error_counts.sort_values('count', ascending=False).head(30)  # Top 30
    
    # Create labels
    error_counts['label'] = error_counts.apply(
        lambda x: f"GT: {x['ground_truth']}\n→ Pred: {x['predicted_answer']}", axis=1
    )
    
    # Plot as horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(error_counts)), error_counts['count'], color='#e74c3c', alpha=0.7)
    ax.set_yticks(range(len(error_counts)))
    ax.set_yticklabels(error_counts['label'], fontsize=9)
    ax.set_xlabel('Error Count', fontsize=12)
    ax.set_title('Top "Different" Errors (Most Frequent Wrong Predictions)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(error_counts['count']):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '6_top_different_errors.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_top_ambiguous_answers(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (10, 8)):
    """
    Plot 7: Top "Ambiguous" Answers Bar Chart
    
    Shows what models say when they give ambiguous answers.
    """
    logging.info("Generating top ambiguous answers plot...")
    
    # Filter for "ambiguous" predictions
    ambiguous_preds = df[df['semantic_relation'] == 'ambiguous'].copy()
    
    if len(ambiguous_preds) == 0:
        logging.warning("No 'ambiguous' answers found. Skipping this plot.")
        return
    
    # Count ambiguous answers
    ambig_counts = ambiguous_preds['predicted_answer'].value_counts().head(20)
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(range(len(ambig_counts)), ambig_counts.values, color='#95a5a6', alpha=0.7)
    ax.set_yticks(range(len(ambig_counts)))
    ax.set_yticklabels(ambig_counts.index, fontsize=10)
    ax.set_xlabel('Count', fontsize=12)
    ax.set_title('Top "Ambiguous" Answers (What models say when uncertain)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, count in enumerate(ambig_counts.values):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '7_top_ambiguous_answers.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def plot_performance_by_dataset(df: pd.DataFrame, output_dir: str, figsize: Tuple[int, int] = (14, 8)):
    """
    Plot 8: Performance by Dataset/Tissue
    
    Shows model performance across different datasets.
    """
    logging.info("Generating performance by dataset plot...")
    
    # Filter valid data
    df_clean = df[df['dataset_id'].notna() & df['score'].notna()].copy()
    if df_clean.empty:
        logging.warning("No valid data for dataset performance plot. Skipping.")
        return
    
    # Get unique datasets and models
    datasets = sorted(df_clean['dataset_id'].unique())
    models = df_clean['model_name'].unique()
    
    if len(datasets) == 0 or len(models) == 0:
        logging.warning("No datasets or models found. Skipping.")
        return
    
    # Grouped box plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grouped box plot
    data_to_plot = []
    labels = []
    positions = []
    
    pos = 0
    for dataset in datasets:
        for model in models:
            subset = df_clean[(df_clean['dataset_id'] == dataset) & (df_clean['model_name'] == model)]
            if len(subset) > 0:
                scores = subset['score'].values
                if len(scores) > 0:
                    data_to_plot.append(scores)
                    labels.append(f"{dataset}\n{model}")
                    positions.append(pos)
                    pos += 1
    
    if len(data_to_plot) == 0:
        logging.warning("No data to plot. Skipping this plot.")
        return
    
    bp = ax.boxplot(data_to_plot, positions=positions, patch_artist=True, widths=0.6)
    
    # Color boxes by model
    model_colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    model_to_color = {model: color for model, color in zip(models, model_colors)}
    
    for patch, label in zip(bp['boxes'], labels):
        label_parts = label.split('\n')
        if len(label_parts) >= 2:
            model_name = label_parts[1]
            patch.set_facecolor(model_to_color.get(model_name, 'lightblue'))
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Dataset × Model', fontsize=12)
    ax.set_title('Model Performance by Dataset', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, alpha=0.7, label=model) 
                      for model, color in model_to_color.items()]
    ax.legend(handles=legend_elements, title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '8_performance_by_dataset.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive performance visualizations')
    parser.add_argument('--judged_results_path', type=str, required=True,
                       help='Path to JSON file with LLM judged results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualization files')
    parser.add_argument('--plots', type=str, nargs='+', 
                       default=['all'],
                       choices=['all', 'score_dist', 'semantic_breakdown', 'celltype_perf', 
                               'heatmap', 'sankey', 'different_errors', 'ambiguous_answers', 
                               'dataset_perf'],
                       help='Which plots to generate (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    logging.info(f"Loading data from {args.judged_results_path}")
    judged_results = load_judged_results(args.judged_results_path)
    
    # Prepare DataFrame
    df = prepare_dataframe(judged_results)
    
    # Generate plots
    plots_to_generate = args.plots if 'all' not in args.plots else [
        'score_dist', 'semantic_breakdown', 'celltype_perf', 'heatmap', 
        'sankey', 'different_errors', 'ambiguous_answers', 'dataset_perf'
    ]
    
    plot_functions = {
        'score_dist': plot_score_distribution,
        'semantic_breakdown': plot_semantic_relation_breakdown,
        'celltype_perf': plot_performance_by_celltype,
        'heatmap': plot_model_celltype_heatmap,
        'sankey': plot_sankey_diagram,
        'different_errors': plot_top_different_errors,
        'ambiguous_answers': plot_top_ambiguous_answers,
        'dataset_perf': plot_performance_by_dataset
    }
    
    for plot_name in plots_to_generate:
        if plot_name in plot_functions:
            try:
                plot_functions[plot_name](df, args.output_dir)
            except Exception as e:
                logging.error(f"Error generating {plot_name}: {e}", exc_info=True)
    
    logging.info(f"All visualizations completed. Output saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

