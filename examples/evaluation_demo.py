#!/usr/bin/env python3
"""
Evaluation Framework Demo

This script demonstrates the evaluation framework capabilities including:
- Loading and managing golden datasets
- Running RAGAS evaluation
- Performance benchmarking
- Results analysis and reporting

Run this demo to see the evaluation framework in action.
"""

import asyncio
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sop_qa_tool.services.evaluation import EvaluationFramework
from sop_qa_tool.services.rag_chain import RAGChain
from sop_qa_tool.models.sop_models import GoldenDatasetItem
from sop_qa_tool.config.settings import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demo_golden_dataset_management():
    """Demonstrate golden dataset creation and management."""
    print("\n" + "="*60)
    print("GOLDEN DATASET MANAGEMENT DEMO")
    print("="*60)
    
    # Initialize evaluation framework
    rag_chain = RAGChain()
    eval_framework = EvaluationFramework(rag_chain)
    
    # Create sample golden dataset
    sample_dataset = [
        GoldenDatasetItem(
            question="What safety equipment is required in the chemical storage area?",
            expected_answer="Safety equipment includes safety goggles, chemical-resistant gloves, apron, and closed-toe shoes.",
            category="safety",
            difficulty="medium",
            filters={"area": ["chemical_storage"]},
            source_documents=["SAFETY-002"],
            metadata={"area_type": "hazardous"}
        ),
        GoldenDatasetItem(
            question="How often should conveyor belts be cleaned?",
            expected_answer="Conveyor belts should be cleaned at the end of each shift and during product changeovers.",
            category="maintenance",
            difficulty="easy",
            filters={"equipment": ["conveyor_belt"]},
            source_documents=["SOP-CLEAN-001"],
            metadata={"frequency": "shift_end"}
        ),
        GoldenDatasetItem(
            question="What is the maximum batch size for chocolate mixing?",
            expected_answer="The maximum batch size is 500 kg to ensure proper mixing consistency.",
            category="procedure",
            difficulty="easy",
            filters={"equipment": ["chocolate_mixer"]},
            source_documents=["SOP-MIX-002"],
            metadata={"unit": "kg", "limit": 500}
        )
    ]
    
    print(f"Created sample dataset with {len(sample_dataset)} items")
    
    # Save dataset
    eval_framework.save_golden_dataset(sample_dataset)
    print("✓ Golden dataset saved")
    
    # Load dataset
    loaded_dataset = eval_framework.load_golden_dataset()
    print(f"✓ Loaded {len(loaded_dataset)} items from saved dataset")
    
    # Analyze dataset
    categories = {}
    difficulties = {}
    
    for item in loaded_dataset:
        categories[item.category] = categories.get(item.category, 0) + 1
        difficulties[item.difficulty] = difficulties.get(item.difficulty, 0) + 1
    
    print("\nDataset Analysis:")
    print(f"Categories: {dict(categories)}")
    print(f"Difficulties: {dict(difficulties)}")
    
    return loaded_dataset


async def demo_ragas_evaluation(golden_dataset):
    """Demonstrate RAGAS evaluation (mocked for demo)."""
    print("\n" + "="*60)
    print("RAGAS EVALUATION DEMO")
    print("="*60)
    
    # Note: This is a demo with mocked results since we need actual documents
    # and a working RAG chain for real RAGAS evaluation
    
    print("Note: This demo shows the evaluation framework structure.")
    print("For actual RAGAS evaluation, you need:")
    print("1. Indexed documents in the vector store")
    print("2. Working RAG chain")
    print("3. RAGAS library properly configured")
    
    # Initialize framework
    rag_chain = RAGChain()
    eval_framework = EvaluationFramework(rag_chain)
    
    print(f"\n✓ Evaluation framework initialized")
    print(f"✓ Target thresholds: {eval_framework.target_thresholds}")
    print(f"✓ RAGAS metrics configured: {len(eval_framework.metrics)} metrics")
    
    # Show what the evaluation would do
    print(f"\nEvaluation would process {len(golden_dataset)} questions:")
    for i, item in enumerate(golden_dataset, 1):
        print(f"{i}. [{item.category}] {item.question[:50]}...")
    
    print("\nEvaluation steps:")
    print("1. Generate answers using RAG chain")
    print("2. Extract context from retrieval")
    print("3. Run RAGAS metrics (faithfulness, relevancy, etc.)")
    print("4. Compare against thresholds")
    print("5. Generate evaluation report")


async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking (mocked for demo)."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING DEMO")
    print("="*60)
    
    # Initialize framework
    rag_chain = RAGChain()
    eval_framework = EvaluationFramework(rag_chain)
    
    # Sample test queries
    test_queries = [
        "What are the safety requirements?",
        "How do you calibrate equipment?",
        "What is the cleaning procedure?"
    ]
    
    print(f"Test queries prepared: {len(test_queries)}")
    print("Concurrent user levels to test: [1, 3, 5, 10]")
    print("Iterations per test: 5")
    
    print("\nBenchmarking would measure:")
    print("- Response latency (mean, P50, P95, P99)")
    print("- System throughput (queries per second)")
    print("- Memory usage patterns")
    print("- Error rates under load")
    
    # Show target performance requirements
    settings = get_settings()
    target_latency = 3.0 if settings.mode == 'aws' else 6.0
    
    print(f"\nPerformance targets ({settings.mode} mode):")
    print(f"- Target latency: {target_latency}s (P95)")
    print(f"- Memory limit: 1.5GB for 50MB document set")
    print(f"- Error rate: <1% under normal load")


async def demo_results_analysis():
    """Demonstrate results analysis and reporting."""
    print("\n" + "="*60)
    print("RESULTS ANALYSIS DEMO")
    print("="*60)
    
    # Mock evaluation results for demonstration
    mock_metrics = {
        "faithfulness": {"score": 0.85, "threshold": 0.8, "passed": True},
        "answer_relevancy": {"score": 0.75, "threshold": 0.8, "passed": False},
        "context_precision": {"score": 0.72, "threshold": 0.7, "passed": True},
        "context_recall": {"score": 0.68, "threshold": 0.7, "passed": False}
    }
    
    print("Sample RAGAS Results:")
    print("-" * 40)
    for metric, data in mock_metrics.items():
        status = "PASS" if data["passed"] else "FAIL"
        gap = data["score"] - data["threshold"]
        print(f"{metric:20s}: {data['score']:.3f} "
              f"(threshold: {data['threshold']:.3f}) "
              f"[{status}] Gap: {gap:+.3f}")
    
    # Calculate pass rate
    passed = sum(1 for m in mock_metrics.values() if m["passed"])
    total = len(mock_metrics)
    pass_rate = passed / total
    
    print(f"\nOverall Pass Rate: {pass_rate:.1%} ({passed}/{total})")
    
    # Mock performance results
    print("\nSample Performance Results:")
    print("-" * 30)
    mock_perf = {
        1: {"mean": 1.2, "p95": 1.8, "qps": 0.8},
        3: {"mean": 1.5, "p95": 2.3, "qps": 2.1},
        5: {"mean": 2.1, "p95": 3.2, "qps": 2.8}
    }
    
    for users, metrics in mock_perf.items():
        print(f"{users} users: Mean={metrics['mean']:.1f}s, "
              f"P95={metrics['p95']:.1f}s, QPS={metrics['qps']:.1f}")
    
    # Generate sample recommendations
    print("\nSample Recommendations:")
    print("-" * 25)
    recommendations = [
        "Improve answer relevancy by optimizing retrieval parameters",
        "Enhance context recall by increasing top-k retrieval values",
        "Optimize performance for concurrent loads above 3 users",
        "Expand golden dataset to include more edge cases"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")


async def demo_automated_reporting():
    """Demonstrate automated reporting capabilities."""
    print("\n" + "="*60)
    print("AUTOMATED REPORTING DEMO")
    print("="*60)
    
    print("The evaluation framework generates multiple report formats:")
    print("\n1. JSON Report (machine-readable)")
    print("   - Complete evaluation results")
    print("   - Raw RAGAS metrics")
    print("   - Detailed performance data")
    print("   - Structured recommendations")
    
    print("\n2. Text Summary (human-readable)")
    print("   - Executive summary")
    print("   - Key findings")
    print("   - Actionable recommendations")
    print("   - Performance analysis")
    
    print("\n3. CSV Metrics (analysis-ready)")
    print("   - Metrics scores and thresholds")
    print("   - Performance benchmarks")
    print("   - Trend analysis data")
    
    print("\n4. Jupyter Notebook (interactive)")
    print("   - Visualizations and charts")
    print("   - Detailed analysis")
    print("   - Customizable reporting")
    
    print("\nReport locations:")
    print("- data/evaluation/results/")
    print("- notebooks/evaluation_analysis.ipynb")
    print("- Automated timestamping for version control")


async def main():
    """Run the complete evaluation framework demo."""
    print("SOP Q&A System - Evaluation Framework Demo")
    print("=" * 60)
    
    try:
        # Demo golden dataset management
        golden_dataset = await demo_golden_dataset_management()
        
        # Demo RAGAS evaluation
        await demo_ragas_evaluation(golden_dataset)
        
        # Demo performance benchmarking
        await demo_performance_benchmarking()
        
        # Demo results analysis
        await demo_results_analysis()
        
        # Demo automated reporting
        await demo_automated_reporting()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nNext steps:")
        print("1. Set up your document index and RAG chain")
        print("2. Create or expand your golden dataset")
        print("3. Run: python scripts/run_evaluation.py")
        print("4. Analyze results in notebooks/evaluation_analysis.ipynb")
        print("5. Implement recommendations to improve performance")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)