#!/usr/bin/env python3
"""
Integration test runner for the SOP Q&A Tool.

This script runs comprehensive integration tests including end-to-end tests,
performance tests, mode switching tests, error scenario tests, and acceptance tests.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


class IntegrationTestRunner:
    """Runner for integration tests with various options and reporting."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> int:
        """Run a command and return the exit code."""
        if cwd is None:
            cwd = self.project_root
            
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=False,
                capture_output=False
            )
            return result.returncode
        except Exception as e:
            print(f"Error running command: {e}")
            return 1
    
    def run_end_to_end_tests(self) -> int:
        """Run end-to-end integration tests."""
        print("\n" + "="*60)
        print("RUNNING END-TO-END INTEGRATION TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_integration_e2e.py"),
            "-v",
            "--tb=short",
            "-m", "not slow"
        ]
        
        return self.run_command(cmd)
    
    def run_performance_tests(self) -> int:
        """Run performance tests."""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_performance.py"),
            "-v",
            "--tb=short",
            "-m", "performance"
        ]
        
        return self.run_command(cmd)
    
    def run_mode_switching_tests(self) -> int:
        """Run mode switching tests."""
        print("\n" + "="*60)
        print("RUNNING MODE SWITCHING TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_mode_switching.py"),
            "-v",
            "--tb=short"
        ]
        
        return self.run_command(cmd)
    
    def run_error_scenario_tests(self) -> int:
        """Run error scenario tests."""
        print("\n" + "="*60)
        print("RUNNING ERROR SCENARIO TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_error_scenarios.py"),
            "-v",
            "--tb=short"
        ]
        
        return self.run_command(cmd)
    
    def run_acceptance_tests(self) -> int:
        """Run acceptance tests for all requirements."""
        print("\n" + "="*60)
        print("RUNNING ACCEPTANCE TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_acceptance_requirements.py"),
            "-v",
            "--tb=short",
            "-m", "not slow"
        ]
        
        return self.run_command(cmd)
    
    def run_all_integration_tests(self, include_slow: bool = False) -> int:
        """Run all integration tests."""
        print("\n" + "="*80)
        print("RUNNING ALL INTEGRATION TESTS")
        print("="*80)
        
        test_files = [
            "test_integration_e2e.py",
            "test_performance.py",
            "test_mode_switching.py",
            "test_error_scenarios.py",
            "test_acceptance_requirements.py"
        ]
        
        cmd = [
            sys.executable, "-m", "pytest"
        ]
        
        # Add test files
        for test_file in test_files:
            cmd.append(str(self.test_dir / test_file))
        
        # Add options
        cmd.extend([
            "-v",
            "--tb=short",
            "--maxfail=5",  # Stop after 5 failures
            "--durations=10"  # Show 10 slowest tests
        ])
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        return self.run_command(cmd)
    
    def run_quick_smoke_tests(self) -> int:
        """Run a quick subset of tests for smoke testing."""
        print("\n" + "="*60)
        print("RUNNING QUICK SMOKE TESTS")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir / "test_integration_e2e.py::TestEndToEndIntegration::test_health_check_integration"),
            str(self.test_dir / "test_mode_switching.py::TestModeSwitching::test_local_mode_functionality"),
            str(self.test_dir / "test_error_scenarios.py::TestErrorScenarios::test_invalid_file_type_rejection"),
            "-v",
            "--tb=line"
        ]
        
        return self.run_command(cmd)
    
    def generate_test_report(self) -> int:
        """Generate comprehensive test report with coverage."""
        print("\n" + "="*60)
        print("GENERATING TEST REPORT")
        print("="*60)
        
        cmd = [
            sys.executable, "-m", "pytest",
            str(self.test_dir),
            "--cov=sop_qa_tool",
            "--cov-report=html:test_reports/coverage_html",
            "--cov-report=xml:test_reports/coverage.xml",
            "--cov-report=term-missing",
            "--junit-xml=test_reports/junit.xml",
            "-v",
            "--tb=short",
            "-m", "not slow"
        ]
        
        # Create reports directory
        reports_dir = self.project_root / "test_reports"
        reports_dir.mkdir(exist_ok=True)
        
        return self.run_command(cmd)
    
    def validate_test_environment(self) -> bool:
        """Validate that the test environment is properly set up."""
        print("Validating test environment...")
        
        # Check if required packages are installed
        required_packages = [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "fastapi",
            "streamlit"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"Missing required packages: {', '.join(missing_packages)}")
            print("Please install them with: pip install -r requirements.txt")
            return False
        
        # Check if test data directory can be created
        test_data_dir = self.project_root / "test_data"
        try:
            test_data_dir.mkdir(exist_ok=True)
            (test_data_dir / "test_file.txt").write_text("test")
            (test_data_dir / "test_file.txt").unlink()
            test_data_dir.rmdir()
        except Exception as e:
            print(f"Cannot create test data directory: {e}")
            return False
        
        print("Test environment validation passed.")
        return True


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run integration tests for the SOP Q&A Tool"
    )
    
    parser.add_argument(
        "test_type",
        choices=[
            "all",
            "e2e",
            "performance",
            "mode-switching",
            "error-scenarios",
            "acceptance",
            "smoke",
            "report"
        ],
        help="Type of tests to run"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow-running tests"
    )
    
    parser.add_argument(
        "--validate-env",
        action="store_true",
        help="Validate test environment before running tests"
    )
    
    args = parser.parse_args()
    
    runner = IntegrationTestRunner()
    
    # Validate environment if requested
    if args.validate_env:
        if not runner.validate_test_environment():
            sys.exit(1)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    # Run the specified tests
    start_time = time.time()
    
    if args.test_type == "all":
        exit_code = runner.run_all_integration_tests(args.include_slow)
    elif args.test_type == "e2e":
        exit_code = runner.run_end_to_end_tests()
    elif args.test_type == "performance":
        exit_code = runner.run_performance_tests()
    elif args.test_type == "mode-switching":
        exit_code = runner.run_mode_switching_tests()
    elif args.test_type == "error-scenarios":
        exit_code = runner.run_error_scenario_tests()
    elif args.test_type == "acceptance":
        exit_code = runner.run_acceptance_tests()
    elif args.test_type == "smoke":
        exit_code = runner.run_quick_smoke_tests()
    elif args.test_type == "report":
        exit_code = runner.generate_test_report()
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n" + "="*60)
    print(f"TEST EXECUTION COMPLETED")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Exit code: {exit_code}")
    print("="*60)
    
    if exit_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above for details.")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()