#!/usr/bin/env python3
"""
Automated evaluation script for SOP Q&A system.

This script runs comprehensive evaluation including RAGAS metrics and performance
benchmarking, then generates reports and visualizations.

Usage:
    python scripts/run_evaluation.py [--mode aws|local] [--output-dir path]
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from sop_qa_tool.services.evaluation import EvaluationFramework
from sop_qa_tool.services.rag_chain import RAGChain
from sop_qa_tool.models.sop_models import EvaluationReport
from sop_qa_tool.config.settings import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """Automated evaluation runner with reporting."""
    
    def __init__(self, output_dir: Path = None):
        """
        Initialize evaluation runner.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.settings = get_settings()
        self.output_dir = output_dir or Path("data/evaluation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize services
        self.rag_chain = RAGChain()
        self.eval_framework = EvaluationFramework(self.rag_chain)
        
        logger.info(f"Initialized evaluation runner in {self.settings.mode} mode")
    
    async def run_full_evaluation(self) -> EvaluationReport:
        """
        Run complete evaluation including RAGAS metrics and performance benchmarks.
        
        Returns:
            EvaluationReport with all results
        """
        logger.info("Starting full evaluation suite")
        
        # Load golden dataset
        golden_dataset = self.eval_framework.load_golden_dataset()
        if not golden_dataset:
            raise ValueError("No golden dataset found. Please create golden dataset first.")
        
        logger.info(f"Loaded {len(golden_dataset)} items from golden dataset")
        
        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation...")
        evaluation_result = await self.eval_framework.evaluate_rag_pipeline(
            golden_dataset, save_results=True
        )
        
        # Prepare test queries for benchmarking
        test_queries = [item.question for item in golden_dataset[:10]]  # Use first 10
        
        # Run performance benchmarking
        logger.info("Running performance benchmarking...")
        benchmark_result = await self.eval_framework.benchmark_performance(
            test_queries=test_queries,
            concurrent_users=[1, 3, 5, 10],
            iterations=5
        )
        
        # Generate comprehensive report
        report = self._generate_report(evaluation_result, benchmark_result)
        
        # Save report
        await self._save_report(report)
        
        logger.info("Full evaluation completed successfully")
        return report
    
    def _generate_report(self, evaluation_result, benchmark_result) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        
        # Calculate summary statistics
        passed_metrics = sum(1 for m in evaluation_result.metrics.values() if m['passed'])
        total_metrics = len(evaluation_result.metrics)
        
        # Performance analysis
        target_latency = 3.0 if self.settings.mode == 'aws' else 6.0
        performance_issues = []
        
        for users in benchmark_result.concurrent_users_tested:
            p95_latency = benchmark_result.results['latency_metrics'][users]['p95']
            if p95_latency > target_latency:
                performance_issues.append(f"{users} users: P95 latency {p95_latency:.3f}s > {target_latency}s")
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            evaluation_result, benchmark_result, performance_issues
        )
        
        # Create summary
        summary = {
            "evaluation_date": evaluation_result.timestamp.isoformat(),
            "mode": self.settings.mode,
            "dataset_size": evaluation_result.dataset_size,
            "metrics_passed": f"{passed_metrics}/{total_metrics}",
            "overall_pass_rate": f"{evaluation_result.overall_pass_rate:.1%}",
            "evaluation_time": f"{evaluation_result.evaluation_time_seconds:.2f}s",
            "performance_issues_count": len(performance_issues),
            "recommendations_count": len(recommendations)
        }
        
        return EvaluationReport(
            evaluation_result=evaluation_result,
            benchmark_result=benchmark_result,
            summary=summary,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, eval_result, bench_result, perf_issues) -> list:
        """Generate actionable recommendations based on results."""
        recommendations = []
        
        # RAGAS-based recommendations
        for metric_name, metric_data in eval_result.metrics.items():
            if not metric_data['passed']:
                gap = metric_data['threshold'] - metric_data['score']
                
                if metric_name == 'faithfulness':
                    recommendations.append({
                        "category": "Quality",
                        "priority": "High",
                        "issue": f"Faithfulness below threshold (gap: {gap:.3f})",
                        "recommendation": "Improve citation accuracy and reduce hallucinations by refining prompt engineering and implementing stricter source verification",
                        "expected_impact": "Increase user trust and answer reliability"
                    })
                
                elif metric_name == 'answer_relevancy':
                    recommendations.append({
                        "category": "Relevance",
                        "priority": "High", 
                        "issue": f"Answer relevancy below threshold (gap: {gap:.3f})",
                        "recommendation": "Optimize retrieval parameters and improve query understanding to return more relevant context",
                        "expected_impact": "Better user satisfaction and more targeted answers"
                    })
                
                elif metric_name == 'context_precision':
                    recommendations.append({
                        "category": "Retrieval",
                        "priority": "Medium",
                        "issue": f"Context precision below threshold (gap: {gap:.3f})",
                        "recommendation": "Fine-tune chunk size, overlap parameters, and retrieval scoring to improve precision",
                        "expected_impact": "Reduced noise in retrieved context"
                    })
                
                elif metric_name == 'context_recall':
                    recommendations.append({
                        "category": "Retrieval",
                        "priority": "Medium",
                        "issue": f"Context recall below threshold (gap: {gap:.3f})",
                        "recommendation": "Increase top-k retrieval values and implement hybrid search (vector + keyword)",
                        "expected_impact": "Better coverage of relevant information"
                    })
        
        # Performance-based recommendations
        if perf_issues:
            recommendations.append({
                "category": "Performance",
                "priority": "High",
                "issue": f"Latency issues under load: {'; '.join(perf_issues)}",
                "recommendation": "Implement caching, optimize embedding generation, and consider async processing for better scalability",
                "expected_impact": "Improved response times under concurrent load"
            })
        
        # Dataset recommendations
        if eval_result.dataset_size < 20:
            recommendations.append({
                "category": "Testing",
                "priority": "Medium",
                "issue": f"Small golden dataset ({eval_result.dataset_size} items)",
                "recommendation": "Expand golden dataset to 30-50 questions covering more edge cases and domain-specific scenarios",
                "expected_impact": "More robust and reliable evaluation results"
            })
        
        return recommendations
    
    async def _save_report(self, report: EvaluationReport):
        """Save comprehensive report to multiple formats."""
        timestamp = report.evaluation_result.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_path = self.output_dir / f"evaluation_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report.dict(), f, indent=2, default=str)
        
        # Save human-readable summary
        summary_path = self.output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            self._write_text_summary(f, report)
        
        # Save CSV metrics
        csv_path = self.output_dir / f"metrics_summary_{timestamp}.csv"
        self._save_metrics_csv(report, csv_path)
        
        logger.info(f"Report saved to:")
        logger.info(f"  JSON: {json_path}")
        logger.info(f"  Summary: {summary_path}")
        logger.info(f"  CSV: {csv_path}")
    
    def _write_text_summary(self, f, report: EvaluationReport):
        """Write human-readable summary to file."""
        f.write("SOP Q&A System Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Executive summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 20 + "\n")
        for key, value in report.summary.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        # RAGAS metrics
        f.write("RAGAS METRICS RESULTS\n")
        f.write("-" * 25 + "\n")
        for metric_name, metric_data in report.evaluation_result.metrics.items():
            status = "PASS" if metric_data['passed'] else "FAIL"
            gap = metric_data['score'] - metric_data['threshold']
            f.write(f"{metric_name:20s}: {metric_data['score']:.3f} "
                   f"(threshold: {metric_data['threshold']:.3f}) "
                   f"[{status}] Gap: {gap:+.3f}\n")
        f.write("\n")
        
        # Performance results
        f.write("PERFORMANCE RESULTS\n")
        f.write("-" * 20 + "\n")
        for users in report.benchmark_result.concurrent_users_tested:
            latency_data = report.benchmark_result.results['latency_metrics'][users]
            throughput_data = report.benchmark_result.results['throughput_metrics'][users]
            f.write(f"{users:2d} users: Mean={latency_data['mean']:.3f}s, "
                   f"P95={latency_data['p95']:.3f}s, "
                   f"QPS={throughput_data['queries_per_second']:.2f}\n")
        f.write("\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        for i, rec in enumerate(report.recommendations, 1):
            f.write(f"{i}. [{rec['priority']}] {rec['category']}: {rec['issue']}\n")
            f.write(f"   Recommendation: {rec['recommendation']}\n")
            f.write(f"   Expected Impact: {rec['expected_impact']}\n\n")
    
    def _save_metrics_csv(self, report: EvaluationReport, csv_path: Path):
        """Save metrics in CSV format for analysis."""
        import pandas as pd
        
        # Prepare metrics data
        metrics_data = []
        for metric_name, metric_data in report.evaluation_result.metrics.items():
            metrics_data.append({
                'metric': metric_name,
                'score': metric_data['score'],
                'threshold': metric_data['threshold'],
                'passed': metric_data['passed'],
                'gap': metric_data['score'] - metric_data['threshold']
            })
        
        df = pd.DataFrame(metrics_data)
        df.to_csv(csv_path, index=False)


async def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(description="Run SOP Q&A system evaluation")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation with fewer iterations")
    
    args = parser.parse_args()
    
    try:
        runner = EvaluationRunner(output_dir=args.output_dir)
        report = await runner.run_full_evaluation()
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Overall Pass Rate: {report.summary['overall_pass_rate']}")
        print(f"Recommendations: {report.summary['recommendations_count']}")
        print(f"Results saved to: {runner.output_dir}")
        
        # Print top recommendations
        if report.recommendations:
            print("\nTOP RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations[:3], 1):
                print(f"{i}. [{rec['priority']}] {rec['issue']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)