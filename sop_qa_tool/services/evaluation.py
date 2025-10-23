"""
RAGAS-based evaluation framework for SOP Q&A system.

This module provides comprehensive evaluation capabilities including:
- RAGAS metrics (faithfulness, relevancy, context precision/recall)
- Performance benchmarking (latency, throughput)
- Golden dataset management
- Automated evaluation reporting
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import pandas as pd
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
    answer_similarity
)

from ..models.sop_models import EvaluationResult, BenchmarkResult, GoldenDatasetItem
from ..services.rag_chain import RAGChain
from ..config.settings import get_settings

logger = logging.getLogger(__name__)


class EvaluationFramework:
    """
    Comprehensive evaluation framework for SOP Q&A system using RAGAS metrics.
    """
    
    def __init__(self, rag_chain: RAGChain, data_path: Optional[Path] = None):
        """
        Initialize evaluation framework.
        
        Args:
            rag_chain: RAG chain instance to evaluate
            data_path: Path to evaluation data directory
        """
        self.rag_chain = rag_chain
        self.settings = get_settings()
        self.data_path = data_path or Path("data/evaluation")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # RAGAS metrics configuration
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
            answer_similarity
        ]
        
        # Target thresholds from design document
        self.target_thresholds = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.7,
            "answer_correctness": 0.75,
            "answer_similarity": 0.7
        }
    
    async def evaluate_rag_pipeline(
        self, 
        golden_dataset: List[GoldenDatasetItem],
        save_results: bool = True
    ) -> EvaluationResult:
        """
        Evaluate RAG pipeline using RAGAS metrics on golden dataset.
        
        Args:
            golden_dataset: List of golden dataset items
            save_results: Whether to save results to disk
            
        Returns:
            EvaluationResult with metrics and analysis
        """
        logger.info(f"Starting RAG evaluation with {len(golden_dataset)} samples")
        
        # Prepare evaluation data
        eval_data = await self._prepare_evaluation_data(golden_dataset)
        
        # Convert to RAGAS dataset format
        dataset = Dataset.from_dict({
            "question": eval_data["questions"],
            "answer": eval_data["generated_answers"],
            "contexts": eval_data["contexts"],
            "ground_truth": eval_data["ground_truths"]
        })
        
        # Run RAGAS evaluation
        start_time = time.time()
        try:
            results = evaluate(dataset, metrics=self.metrics)
            evaluation_time = time.time() - start_time
            
            # Process results
            evaluation_result = self._process_ragas_results(
                results, 
                golden_dataset,
                evaluation_time
            )
            
            if save_results:
                await self._save_evaluation_results(evaluation_result)
            
            logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            raise
    
    async def benchmark_performance(
        self, 
        test_queries: List[str],
        concurrent_users: List[int] = [1, 5, 10],
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark system performance for latency and throughput.
        
        Args:
            test_queries: List of test queries
            concurrent_users: List of concurrent user counts to test
            iterations: Number of iterations per test
            
        Returns:
            BenchmarkResult with performance metrics
        """
        logger.info(f"Starting performance benchmark with {len(test_queries)} queries")
        
        benchmark_results = {
            "latency_metrics": {},
            "throughput_metrics": {},
            "memory_usage": {},
            "error_rates": {}
        }
        
        for user_count in concurrent_users:
            logger.info(f"Testing with {user_count} concurrent users")
            
            # Run latency tests
            latencies = await self._measure_latency(test_queries, user_count, iterations)
            benchmark_results["latency_metrics"][user_count] = {
                "mean": np.mean(latencies),
                "p50": np.percentile(latencies, 50),
                "p95": np.percentile(latencies, 95),
                "p99": np.percentile(latencies, 99),
                "std": np.std(latencies)
            }
            
            # Run throughput tests
            throughput = await self._measure_throughput(test_queries, user_count, iterations)
            benchmark_results["throughput_metrics"][user_count] = throughput
        
        # Create benchmark result
        result = BenchmarkResult(
            timestamp=datetime.now(),
            test_queries_count=len(test_queries),
            concurrent_users_tested=concurrent_users,
            iterations_per_test=iterations,
            results=benchmark_results
        )
        
        # Save results
        await self._save_benchmark_results(result)
        
        return result
    
    async def _prepare_evaluation_data(
        self, 
        golden_dataset: List[GoldenDatasetItem]
    ) -> Dict[str, List]:
        """
        Prepare evaluation data by running RAG pipeline on golden dataset.
        """
        questions = []
        generated_answers = []
        contexts = []
        ground_truths = []
        
        for item in golden_dataset:
            try:
                # Get RAG response
                rag_result = await self.rag_chain.answer_question(
                    item.question,
                    filters=item.filters or {}
                )
                
                questions.append(item.question)
                generated_answers.append(rag_result.answer)
                contexts.append([ctx.text for ctx in rag_result.context])
                ground_truths.append(item.expected_answer)
                
            except Exception as e:
                logger.warning(f"Failed to process question '{item.question}': {e}")
                continue
        
        return {
            "questions": questions,
            "generated_answers": generated_answers,
            "contexts": contexts,
            "ground_truths": ground_truths
        }
    
    def _process_ragas_results(
        self, 
        ragas_results: Dict,
        golden_dataset: List[GoldenDatasetItem],
        evaluation_time: float
    ) -> EvaluationResult:
        """
        Process RAGAS results into structured evaluation result.
        """
        metrics = {}
        passed_thresholds = {}
        
        for metric_name, threshold in self.target_thresholds.items():
            if metric_name in ragas_results:
                score = ragas_results[metric_name]
                metrics[metric_name] = {
                    "score": float(score),
                    "threshold": threshold,
                    "passed": score >= threshold
                }
                passed_thresholds[metric_name] = score >= threshold
        
        # Calculate overall pass rate
        overall_pass_rate = sum(passed_thresholds.values()) / len(passed_thresholds)
        
        return EvaluationResult(
            timestamp=datetime.now(),
            dataset_size=len(golden_dataset),
            evaluation_time_seconds=evaluation_time,
            metrics=metrics,
            overall_pass_rate=overall_pass_rate,
            raw_results=dict(ragas_results)
        )
    
    async def _measure_latency(
        self, 
        queries: List[str], 
        concurrent_users: int, 
        iterations: int
    ) -> List[float]:
        """
        Measure response latency under concurrent load.
        """
        latencies = []
        
        async def single_query(query: str) -> float:
            start_time = time.time()
            try:
                await self.rag_chain.answer_question(query)
                return time.time() - start_time
            except Exception as e:
                logger.warning(f"Query failed during latency test: {e}")
                return float('inf')
        
        for _ in range(iterations):
            # Create concurrent tasks
            tasks = []
            for i in range(concurrent_users):
                query = queries[i % len(queries)]
                tasks.append(single_query(query))
            
            # Execute concurrently and collect latencies
            batch_latencies = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and infinite values
            valid_latencies = [
                lat for lat in batch_latencies 
                if isinstance(lat, float) and lat != float('inf')
            ]
            latencies.extend(valid_latencies)
        
        return latencies
    
    async def _measure_throughput(
        self, 
        queries: List[str], 
        concurrent_users: int, 
        iterations: int
    ) -> Dict[str, float]:
        """
        Measure system throughput (queries per second).
        """
        total_queries = concurrent_users * iterations
        start_time = time.time()
        
        # Run all queries
        await self._measure_latency(queries, concurrent_users, iterations)
        
        total_time = time.time() - start_time
        throughput = total_queries / total_time if total_time > 0 else 0
        
        return {
            "queries_per_second": throughput,
            "total_queries": total_queries,
            "total_time_seconds": total_time
        }
    
    async def _save_evaluation_results(self, result: EvaluationResult):
        """Save evaluation results to disk."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filepath = self.data_path / f"evaluation_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {filepath}")
    
    async def _save_benchmark_results(self, result: BenchmarkResult):
        """Save benchmark results to disk."""
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        filepath = self.data_path / f"benchmark_results_{timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump(result.dict(), f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {filepath}")
    
    def load_golden_dataset(self, filepath: Optional[Path] = None) -> List[GoldenDatasetItem]:
        """
        Load golden dataset from JSON file.
        
        Args:
            filepath: Path to golden dataset file
            
        Returns:
            List of golden dataset items
        """
        if filepath is None:
            filepath = self.data_path / "golden_dataset.json"
        
        if not filepath.exists():
            logger.warning(f"Golden dataset not found at {filepath}")
            return []
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return [GoldenDatasetItem(**item) for item in data]
    
    def save_golden_dataset(
        self, 
        dataset: List[GoldenDatasetItem], 
        filepath: Optional[Path] = None
    ):
        """
        Save golden dataset to JSON file.
        
        Args:
            dataset: List of golden dataset items
            filepath: Path to save dataset
        """
        if filepath is None:
            filepath = self.data_path / "golden_dataset.json"
        
        data = [item.dict() for item in dataset]
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Golden dataset saved to {filepath} ({len(dataset)} items)")