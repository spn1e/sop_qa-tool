"""
Tests for the evaluation framework.

This module tests the RAGAS evaluation pipeline, performance benchmarking,
and golden dataset management functionality.
"""

import pytest
import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import tempfile

from sop_qa_tool.services.evaluation import EvaluationFramework
from sop_qa_tool.models.sop_models import (
    GoldenDatasetItem, EvaluationResult, BenchmarkResult,
    EvaluationMetric
)


class TestEvaluationFramework:
    """Test cases for EvaluationFramework."""
    
    @pytest.fixture
    def mock_rag_chain(self):
        """Mock RAG chain for testing."""
        rag_chain = Mock()
        rag_chain.answer_question = AsyncMock()
        return rag_chain
    
    @pytest.fixture
    def temp_data_path(self):
        """Temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def evaluation_framework(self, mock_rag_chain, temp_data_path):
        """EvaluationFramework instance for testing."""
        return EvaluationFramework(mock_rag_chain, temp_data_path)
    
    @pytest.fixture
    def sample_golden_dataset(self):
        """Sample golden dataset for testing."""
        return [
            GoldenDatasetItem(
                question="What are the safety requirements?",
                expected_answer="Safety requirements include wearing PPE and following procedures.",
                category="safety",
                difficulty="medium",
                filters={"roles": ["Operator"]},
                source_documents=["SOP-001"]
            ),
            GoldenDatasetItem(
                question="How do you calibrate the equipment?",
                expected_answer="Calibration involves checking accuracy and adjusting as needed.",
                category="calibration",
                difficulty="hard",
                filters={"equipment": ["calibrator"]},
                source_documents=["SOP-002"]
            )
        ]
    
    def test_initialization(self, evaluation_framework, temp_data_path):
        """Test EvaluationFramework initialization."""
        assert evaluation_framework.data_path == temp_data_path
        assert len(evaluation_framework.metrics) == 6  # All RAGAS metrics
        assert "faithfulness" in evaluation_framework.target_thresholds
        assert evaluation_framework.target_thresholds["faithfulness"] == 0.8
    
    def test_save_and_load_golden_dataset(self, evaluation_framework, sample_golden_dataset):
        """Test saving and loading golden dataset."""
        # Save dataset
        evaluation_framework.save_golden_dataset(sample_golden_dataset)
        
        # Verify file exists
        dataset_file = evaluation_framework.data_path / "golden_dataset.json"
        assert dataset_file.exists()
        
        # Load dataset
        loaded_dataset = evaluation_framework.load_golden_dataset()
        
        # Verify content
        assert len(loaded_dataset) == len(sample_golden_dataset)
        assert loaded_dataset[0].question == sample_golden_dataset[0].question
        assert loaded_dataset[0].category == sample_golden_dataset[0].category
    
    def test_load_nonexistent_golden_dataset(self, evaluation_framework):
        """Test loading golden dataset when file doesn't exist."""
        dataset = evaluation_framework.load_golden_dataset()
        assert dataset == []
    
    @pytest.mark.asyncio
    async def test_prepare_evaluation_data(self, evaluation_framework, sample_golden_dataset):
        """Test preparation of evaluation data."""
        # Mock RAG chain responses
        mock_context = Mock()
        mock_context.text = "Sample context text"
        
        mock_result = Mock()
        mock_result.answer = "Generated answer"
        mock_result.context = [mock_context]
        
        evaluation_framework.rag_chain.answer_question.return_value = mock_result
        
        # Prepare data
        eval_data = await evaluation_framework._prepare_evaluation_data(sample_golden_dataset)
        
        # Verify structure
        assert "questions" in eval_data
        assert "generated_answers" in eval_data
        assert "contexts" in eval_data
        assert "ground_truths" in eval_data
        
        # Verify content
        assert len(eval_data["questions"]) == 2
        assert eval_data["questions"][0] == sample_golden_dataset[0].question
        assert eval_data["generated_answers"][0] == "Generated answer"
        assert eval_data["contexts"][0] == ["Sample context text"]
        assert eval_data["ground_truths"][0] == sample_golden_dataset[0].expected_answer
    
    def test_process_ragas_results(self, evaluation_framework, sample_golden_dataset):
        """Test processing of RAGAS results."""
        # Mock RAGAS results
        ragas_results = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.75,
            "context_precision": 0.65,
            "context_recall": 0.72
        }
        
        evaluation_time = 10.5
        
        # Process results
        result = evaluation_framework._process_ragas_results(
            ragas_results, sample_golden_dataset, evaluation_time
        )
        
        # Verify result structure
        assert isinstance(result, EvaluationResult)
        assert result.dataset_size == len(sample_golden_dataset)
        assert result.evaluation_time_seconds == evaluation_time
        
        # Verify metrics
        assert "faithfulness" in result.metrics
        assert result.metrics["faithfulness"]["score"] == 0.85
        assert result.metrics["faithfulness"]["passed"] == True  # 0.85 > 0.8
        
        assert "context_precision" in result.metrics
        assert result.metrics["context_precision"]["passed"] == False  # 0.65 < 0.7
        
        # Verify overall pass rate
        passed_count = sum(1 for m in result.metrics.values() if m["passed"])
        expected_pass_rate = passed_count / len(result.metrics)
        assert result.overall_pass_rate == expected_pass_rate
    
    @pytest.mark.asyncio
    async def test_measure_latency(self, evaluation_framework):
        """Test latency measurement."""
        # Mock RAG chain with controlled latency
        async def mock_answer_question(query):
            await asyncio.sleep(0.1)  # 100ms latency
            return Mock()
        
        evaluation_framework.rag_chain.answer_question = mock_answer_question
        
        # Measure latency
        queries = ["test query 1", "test query 2"]
        latencies = await evaluation_framework._measure_latency(
            queries, concurrent_users=2, iterations=2
        )
        
        # Verify results
        assert len(latencies) == 4  # 2 users * 2 iterations
        assert all(0.08 <= lat <= 0.15 for lat in latencies)  # Around 100ms ± tolerance
    
    @pytest.mark.asyncio
    async def test_measure_throughput(self, evaluation_framework):
        """Test throughput measurement."""
        # Mock RAG chain
        evaluation_framework.rag_chain.answer_question = AsyncMock()
        
        # Mock _measure_latency to return quickly
        evaluation_framework._measure_latency = AsyncMock(return_value=[0.1, 0.1, 0.1, 0.1])
        
        # Measure throughput
        queries = ["test query"]
        throughput = await evaluation_framework._measure_throughput(
            queries, concurrent_users=2, iterations=2
        )
        
        # Verify structure
        assert "queries_per_second" in throughput
        assert "total_queries" in throughput
        assert "total_time_seconds" in throughput
        
        # Verify values
        assert throughput["total_queries"] == 4  # 2 users * 2 iterations
        assert throughput["queries_per_second"] > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_performance(self, evaluation_framework):
        """Test performance benchmarking."""
        # Mock latency measurement
        evaluation_framework._measure_latency = AsyncMock(
            return_value=[0.1, 0.12, 0.15, 0.11]
        )
        
        # Mock throughput measurement
        evaluation_framework._measure_throughput = AsyncMock(
            return_value={
                "queries_per_second": 8.5,
                "total_queries": 4,
                "total_time_seconds": 0.47
            }
        )
        
        # Run benchmark
        test_queries = ["query1", "query2"]
        result = await evaluation_framework.benchmark_performance(
            test_queries, concurrent_users=[1, 2], iterations=2
        )
        
        # Verify result structure
        assert isinstance(result, BenchmarkResult)
        assert result.test_queries_count == 2
        assert result.concurrent_users_tested == [1, 2]
        assert result.iterations_per_test == 2
        
        # Verify results structure
        assert "latency_metrics" in result.results
        assert "throughput_metrics" in result.results
        
        # Verify latency metrics for each user count
        for user_count in [1, 2]:
            assert user_count in result.results["latency_metrics"]
            latency_data = result.results["latency_metrics"][user_count]
            assert "mean" in latency_data
            assert "p50" in latency_data
            assert "p95" in latency_data
            assert "p99" in latency_data
            assert "std" in latency_data
    
    @pytest.mark.asyncio
    async def test_save_evaluation_results(self, evaluation_framework):
        """Test saving evaluation results."""
        # Create mock evaluation result
        result = EvaluationResult(
            timestamp=datetime.now(),
            dataset_size=10,
            evaluation_time_seconds=15.5,
            metrics={
                "faithfulness": EvaluationMetric(score=0.85, threshold=0.8, passed=True)
            },
            overall_pass_rate=1.0,
            raw_results={"faithfulness": 0.85}
        )
        
        # Save results
        await evaluation_framework._save_evaluation_results(result)
        
        # Verify file was created
        result_files = list(evaluation_framework.data_path.glob("evaluation_results_*.json"))
        assert len(result_files) == 1
        
        # Verify content
        with open(result_files[0]) as f:
            saved_data = json.load(f)
        
        assert saved_data["dataset_size"] == 10
        assert saved_data["evaluation_time_seconds"] == 15.5
        assert saved_data["overall_pass_rate"] == 1.0
    
    @pytest.mark.asyncio
    async def test_save_benchmark_results(self, evaluation_framework):
        """Test saving benchmark results."""
        # Create mock benchmark result
        result = BenchmarkResult(
            timestamp=datetime.now(),
            test_queries_count=5,
            concurrent_users_tested=[1, 2],
            iterations_per_test=3,
            results={"latency_metrics": {}, "throughput_metrics": {}}
        )
        
        # Save results
        await evaluation_framework._save_benchmark_results(result)
        
        # Verify file was created
        result_files = list(evaluation_framework.data_path.glob("benchmark_results_*.json"))
        assert len(result_files) == 1
        
        # Verify content
        with open(result_files[0]) as f:
            saved_data = json.load(f)
        
        assert saved_data["test_queries_count"] == 5
        assert saved_data["concurrent_users_tested"] == [1, 2]
        assert saved_data["iterations_per_test"] == 3


class TestEvaluationIntegration:
    """Integration tests for evaluation framework."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline with mocked components."""
        # This test would require actual RAG chain and RAGAS
        # For now, we'll skip it and implement when integration testing is set up
        pytest.skip("Integration test - requires full system setup")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_performance_under_load(self):
        """Test evaluation framework performance under load."""
        # This test would stress test the evaluation framework
        pytest.skip("Performance test - requires extended runtime")


if __name__ == "__main__":
    pytest.main([__file__])
