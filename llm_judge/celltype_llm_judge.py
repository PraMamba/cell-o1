#!/usr/bin/env python3
"""
Cell Type LLM Judge - DeepSeek API Based Evaluation
====================================================

Evaluates cell type predictions using LLM as a judge with semantic understanding.
Uses DeepSeek API for evaluation with detailed semantic relation scoring.
"""

import json
import argparse
import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime
import os
from collections import Counter
import random
import time
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_logging(output_dir: str, log_filename: str = "celltype_llm_judge.log"):
    """Set up logging to both console and file."""
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, log_filename)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.handlers = []
    logging.root.addHandler(console_handler)
    logging.root.addHandler(file_handler)
    logging.root.setLevel(logging.INFO)
    
    logging.info(f"LLM Judge logging initialized. Log file: {log_file_path}")
    return log_file_path


# Judge Prompt Template
ENGLISH_JUDGE_PROMPT = """## Role & Goal

You are an expert bioinformatician specializing in immunology, single-cell biology, and cell naming conventions (e.g., Cell Ontology - CL).

Your core task is to receive a `ground_truth` cell type and a `predicted_answer` cell type. You must judge the semantic relationship between them based on their **biological lineage, differentiation state, function, known markers, and consistent usage in scientific literature and ontologies (like CL)**.

You must strictly follow the classification criteria below and output your judgment in the specified JSON format.

---

## Evaluation Criteria & Scoring Rubric

You must select one of the 6 categories and provide a precise score (0.0-1.0) according to the rules.

### 1. equivalent
* **Criteria**: The terms are known synonyms, common abbreviations, have only minor differences (case, hyphenation, pluralization), or map to the same Cell Ontology ID.
* **Examples**:
    * `ground_truth: "Natural killer cell"` | `predicted_answer: "NK cell"`
    * `ground_truth: "CD4+ T cell"` | `predicted_answer: "CD4-positive T-cell"`
* **Score**: 1.0

### 2. parent-child
* **Criteria**: One term is a biological parent (a broader category) or a child (a more granular subtype) of the other.
* **Examples**:
    * (Child predicting Parent) `ground_truth: "Classical monocyte"` | `predicted_answer: "Monocyte"`
    * (Parent predicting Child) `ground_truth: "T cell"` | `predicted_answer: "CD8+ naive T cell"`
* **Score**: 0.7 - 0.9.
    * If the prediction is a **direct child** of the ground truth (e.g., T cell -> CD8+ T cell), lean towards 0.8-0.9 (correct but more specific).
    * If the prediction is a **parent** of the ground truth (e.g., Classical monocyte -> Monocyte), lean towards 0.7-0.8 (correct but not specific enough).

### 3. same_major_lineage
* **Criteria**: Both terms share a close common ancestor (e.g., both are lymphocytes or both are myeloid cells), but they are **distinct** branches or types with different functions and identities.
* **Examples**:
    * `ground_truth: "NK cell"` | `predicted_answer: "CD8+ T cell"` (Both are lymphocytes, but different sub-lineages)
    * `ground_truth: "B cell"` | `predicted_answer: "Plasma cell"` (Plasma cells differentiate from B cells, but are a terminally differentiated state with a distinct function)
    * `ground_truth: "Neutrophil"` | `predicted_answer: "Macrophage"` (Both are myeloid, but different branches and functions)
* **Score**: 0.5 - 0.7.

### 4. partially_related
* **Criteria**: The terms are not on the same direct lineage but share significant functional similarity (e.g., both are APCs), tissue location, or high-level concepts, leading to potential confusion. They are **definitively different** cell identities.
* **Examples**:
    * `ground_truth: "Macrophage"` | `predicted_answer: "Dendritic cell"` (Similar function (APC), but different lineages and primary identities)
    * `ground_truth: "Fibroblast"` | `predicted_answer: "Pericyte"` (Both are mesenchymal, related in function/location, but distinct identities)
* **Score**: 0.3 - 0.5.

### 5. ambiguous
* **Criteria**: The predicted term is too broad, vague, or non-specific (e.g., "Immune cell," "Stromal cell") to be meaningfully compared to the ground truth.
* **Examples**:
    * `ground_truth: "CD4+ T cell"` | `predicted_answer: "Immune cell"`
    * `ground_truth: "Monocyte"` | `predicted_answer: "Myeloid"` (Refers to a lineage, not a cell type)
* **Score**: 0.1 - 0.3.

### 6. different
* **Criteria**: The two terms are biologically unrelated in lineage, function, and ontology.
* **Examples**:
    * `ground_truth: "T cell"` | `predicted_answer: "Fibroblast"`
    * `ground_truth: "Epithelial cell"` | `predicted_answer: "Erythrocyte"`
* **Score**: 0.0 - 0.1 (Usually 0.0).

---

## 📌 Output Format

You MUST return a single, valid JSON object and nothing else. Do not include any explanatory text outside the JSON structure.

{{
  "semantic_relation": "...", // Must be one of [equivalent, parent-child, same_major_lineage, partially_related, ambiguous, different]
  "score": ..., // A float between 0.0 and 1.0
  "explanation": "A brief justification based on lineage, differentiation, or ontology (CL). E.g., 'Prediction is the parent lineage of the ground truth, a parent-child relationship.'"
}}

---
## Data to Evaluate:

Ground Truth: {ground_truth_cell_type}
Predicted Answer: {predicted_cell_type}
"""


class CellTypeJudgment(BaseModel):
    """Cell type judgment schema."""
    semantic_relation: str = Field(description="Semantic relationship category")
    score: float = Field(description="Score between 0.0 and 1.0")
    explanation: str = Field(description="Brief explanation")


class CellTypeLLMJudge:
    """Cell Type LLM Judge using DeepSeek API."""
    
    def __init__(self, llm_model: str = "deepseek-chat", 
                 api_key: str = "", 
                 base_url: str = "https://api.deepseek.com",
                 max_concurrent: int = 5, 
                 delay_between_batches: float = 1.0):
        
        self.api_key = api_key
        self.model = llm_model
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.delay_between_batches = delay_between_batches
        
        # Initialize AsyncOpenAI client for DeepSeek
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=60.0,
            max_retries=3
        )
        
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logging.info(f"Initialized CellTypeLLMJudge with model: {self.model}")
        logging.info(f"Base URL: {self.base_url}")
        logging.info(f"Max concurrent: {self.max_concurrent}")
    
    def load_predictions(self, predictions_path: str) -> List[Dict]:
        """Load prediction results from JSON file."""
        with open(predictions_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} predictions from {predictions_path}")
        return data
    
    def sample_data(self, data: List[Dict], max_samples: int = None, random_seed: int = 42) -> List[Dict]:
        """Sample data if needed."""
        if max_samples is None or max_samples < 0 or len(data) <= max_samples:
            return data

        random.seed(random_seed)
        sampled = random.sample(data, max_samples)
        logging.info(f"Randomly sampled {len(sampled)} from {len(data)} samples")
        return sampled
    
    async def judge_single_prediction(self, ground_truth: str, predicted_answer: str) -> Dict:
        """Judge a single prediction using DeepSeek API."""
        async with self.semaphore:
            try:
                # Format prompt
                prompt = ENGLISH_JUDGE_PROMPT.format(
                    ground_truth_cell_type=ground_truth,
                    predicted_cell_type=predicted_answer
                )
                
                # Call DeepSeek API
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert bioinformatician specializing in cell type identification."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500
                )
                
                # Parse response
                content = response.choices[0].message.content.strip()
                
                # Try to parse JSON
                try:
                    result = json.loads(content)
                    
                    # Validate semantic_relation
                    valid_relations = ["equivalent", "parent-child", "same_major_lineage", 
                                      "partially_related", "ambiguous", "different"]
                    if result.get("semantic_relation") not in valid_relations:
                        logging.warning(f"Invalid semantic_relation: {result.get('semantic_relation')}")
                        result["semantic_relation"] = "different"
                    
                    # Validate score
                    score = float(result.get("score", 0.0))
                    if not (0.0 <= score <= 1.0):
                        score = 0.0
                    result["score"] = score
                    
                    return result
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON parse error: {e}, content: {content[:200]}")
                    return {
                        "semantic_relation": "different",
                        "score": 0.0,
                        "explanation": f"JSON parse error: {str(e)[:100]}"
                    }
                    
            except Exception as e:
                logging.error(f"API call failed: {e}")
                return {
                    "semantic_relation": "different",
                    "score": 0.0,
                    "explanation": f"API call failed: {str(e)[:100]}"
                }
    
    async def judge_predictions(self, predictions: List[Dict], batch_size: int = 50) -> List[Dict]:
        """Judge all predictions in batches."""
        judged_results = []
        total_batches = (len(predictions) + batch_size - 1) // batch_size
        
        for i in range(0, len(predictions), batch_size):
            batch = predictions[i:i + batch_size]
            logging.info(f"Processing batch {i//batch_size + 1}/{total_batches} ({len(batch)} samples)")
            
            # Create judgment tasks for this batch
            tasks = []
            task_info = []
            
            for sample_idx, sample in enumerate(batch):
                ground_truth = sample.get("ground_truth", "")
                predicted_answer = sample.get("predicted_answer", "")
                
                # Skip if missing data
                if not ground_truth or not predicted_answer:
                    logging.warning(f"Skipping sample {sample.get('index', i + sample_idx)}: missing ground_truth or predicted_answer")
                    continue
                
                task = self.judge_single_prediction(ground_truth, predicted_answer)
                tasks.append(task)
                task_info.append({
                    "sample_idx": sample_idx,
                    "sample": sample
                })
            
            # Execute all tasks for this batch
            start_time = time.time()
            batch_judgments = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time
            
            # Assign judgments back to samples
            for task_idx, (judgment, info) in enumerate(zip(batch_judgments, task_info)):
                sample = info["sample"]
                
                if isinstance(judgment, Exception):
                    logging.error(f"Judgment failed for sample {sample.get('index')}: {judgment}")
                    judgment = {
                        "semantic_relation": "different",
                        "score": 0.0,
                        "explanation": "Judgment failed"
                    }
                
                # Create judged result
                judged_sample = {
                    **sample,  # Keep original fields
                    "llm_judgment": judgment,
                    "judgment_timestamp": datetime.now().isoformat()
                }
                
                judged_results.append(judged_sample)
            
            # Log progress
            avg_time_per_call = elapsed / len(tasks) if tasks else 0
            logging.info(f"  Batch completed in {elapsed:.1f}s "
                        f"({len(tasks)} judgments, avg {avg_time_per_call:.2f}s/judgment)")
            
            # Rate limiting between batches
            if i + batch_size < len(predictions) and self.delay_between_batches > 0:
                await asyncio.sleep(self.delay_between_batches)
        
        return judged_results
    
    def analyze_judgments(self, judged_results: List[Dict]) -> Dict:
        """Analyze judgment results and compute statistics."""
        # Collect all judgments
        semantic_relations = []
        scores = []
        
        for sample in judged_results:
            if "llm_judgment" in sample:
                judgment = sample["llm_judgment"]
                semantic_relations.append(judgment.get("semantic_relation", "different"))
                scores.append(judgment.get("score", 0.0))
        
        total_samples = len(judged_results)
        
        if total_samples == 0:
            return {}
        
        # Count semantic relations
        relation_counts = Counter(semantic_relations)
        
        # Calculate statistics
        avg_score = sum(scores) / total_samples if scores else 0.0
        
        # Count by score ranges
        exact_match = sum(1 for s in scores if s >= 0.95)  # score >= 0.95
        good_match = sum(1 for s in scores if s >= 0.7)    # score >= 0.7
        partial_match = sum(1 for s in scores if s >= 0.5) # score >= 0.5
        poor_match = sum(1 for s in scores if s < 0.5)     # score < 0.5
        
        analysis = {
            "total_samples": total_samples,
            "semantic_relation_distribution": {
                "equivalent": relation_counts.get("equivalent", 0),
                "parent-child": relation_counts.get("parent-child", 0),
                "same_major_lineage": relation_counts.get("same_major_lineage", 0),
                "partially_related": relation_counts.get("partially_related", 0),
                "ambiguous": relation_counts.get("ambiguous", 0),
                "different": relation_counts.get("different", 0)
            },
            "semantic_relation_rates": {
                "equivalent_rate": relation_counts.get("equivalent", 0) / total_samples,
                "parent-child_rate": relation_counts.get("parent-child", 0) / total_samples,
                "same_major_lineage_rate": relation_counts.get("same_major_lineage", 0) / total_samples,
                "partially_related_rate": relation_counts.get("partially_related", 0) / total_samples,
                "ambiguous_rate": relation_counts.get("ambiguous", 0) / total_samples,
                "different_rate": relation_counts.get("different", 0) / total_samples
            },
            "score_statistics": {
                "average_score": avg_score,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "exact_match_count": exact_match,
                "exact_match_rate": exact_match / total_samples,
                "good_match_count": good_match,
                "good_match_rate": good_match / total_samples,
                "partial_match_count": partial_match,
                "partial_match_rate": partial_match / total_samples,
                "poor_match_count": poor_match,
                "poor_match_rate": poor_match / total_samples
            }
        }
        
        return analysis


async def main():
    parser = argparse.ArgumentParser(description="Cell Type LLM Judge using DeepSeek API")
    parser.add_argument("--predictions_path", type=str, required=True,
                       help="Path to predictions JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    
    # Sampling options
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to evaluate (random sampling)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    # Performance options
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--max_concurrent", type=int, default=5, help="Max concurrent API calls")
    parser.add_argument("--delay_between_batches", type=float, default=1.0, 
                       help="Delay between batches (seconds)")
    
    # API options
    parser.add_argument("--llm_model", type=str, default="deepseek-chat", help="LLM model name")
    parser.add_argument("--llm_api_key", type=str, default=None, help="DeepSeek API key")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com", 
                       help="DeepSeek API base URL")
    
    args = parser.parse_args()
    
    # Setup API key
    api_key = args.llm_api_key or os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        logging.error("API key not found. Please set DEEPSEEK_API_KEY environment variable or use --llm_api_key")
        return
    
    # Setup logging
    setup_logging(args.output_dir)
    
    logging.info("=" * 80)
    logging.info("CELL TYPE LLM JUDGE - DEEPSEEK API")
    logging.info("=" * 80)
    
    # Initialize judge
    judge = CellTypeLLMJudge(
        llm_model=args.llm_model,
        api_key=api_key,
        base_url=args.base_url,
        max_concurrent=args.max_concurrent,
        delay_between_batches=args.delay_between_batches
    )
    
    # Load predictions
    logging.info("Loading predictions...")
    predictions = judge.load_predictions(args.predictions_path)
    
    # Sample data if needed
    if args.max_samples:
        predictions = judge.sample_data(predictions, args.max_samples, args.random_seed)
    
    logging.info(f"Will evaluate {len(predictions)} predictions")
    logging.info(f"Estimated time: ~{len(predictions) * 2 / args.max_concurrent / 60:.1f} minutes")
    
    # Judge predictions
    logging.info("Starting LLM judgment...")
    start_time = time.time()
    judged_results = await judge.judge_predictions(predictions, args.batch_size)
    total_time = time.time() - start_time
    
    logging.info(f"Judgment completed in {total_time:.1f} seconds")
    
    # Analyze results
    logging.info("Analyzing results...")
    analysis = judge.analyze_judgments(judged_results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save judged results
    judged_file = os.path.join(args.output_dir, "celltype_judged_results.json")
    with open(judged_file, "w", encoding='utf-8') as f:
        json.dump(judged_results, f, indent=2, ensure_ascii=False)
    
    # Save analysis
    analysis_file = os.path.join(args.output_dir, "celltype_judgment_analysis.json")
    with open(analysis_file, "w", encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_processing_time": total_time,
            "predictions_file": args.predictions_path,
            "config": {
                "max_samples": args.max_samples,
                "batch_size": args.batch_size,
                "max_concurrent": args.max_concurrent,
                "llm_model": args.llm_model,
                "random_seed": args.random_seed
            },
            "analysis": analysis
        }, f, indent=2, ensure_ascii=False)
    
    # Print summary
    logging.info("\n" + "="*70)
    logging.info("CELL TYPE LLM JUDGMENT COMPLETED")
    logging.info("="*70)
    logging.info(f"Total time: {total_time:.1f} seconds")
    logging.info(f"Average time per judgment: {total_time/len(judged_results):.2f} seconds")
    
    if analysis:
        logging.info("\n=== Semantic Relation Distribution ===")
        dist = analysis['semantic_relation_distribution']
        rates = analysis['semantic_relation_rates']
        for relation in ["equivalent", "parent-child", "same_major_lineage", 
                        "partially_related", "ambiguous", "different"]:
            count = dist.get(relation, 0)
            rate = rates.get(f"{relation}_rate", 0.0)
            logging.info(f"  {relation}: {count} ({rate:.3f})")
        
        logging.info("\n=== Score Statistics ===")
        stats = analysis['score_statistics']
        logging.info(f"  Average score: {stats['average_score']:.3f}")
        logging.info(f"  Min score: {stats['min_score']:.3f}")
        logging.info(f"  Max score: {stats['max_score']:.3f}")
        logging.info(f"  Exact match (≥0.95): {stats['exact_match_count']} ({stats['exact_match_rate']:.3f})")
        logging.info(f"  Good match (≥0.7): {stats['good_match_count']} ({stats['good_match_rate']:.3f})")
        logging.info(f"  Partial match (≥0.5): {stats['partial_match_count']} ({stats['partial_match_rate']:.3f})")
        logging.info(f"  Poor match (<0.5): {stats['poor_match_count']} ({stats['poor_match_rate']:.3f})")
    
    logging.info(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())

