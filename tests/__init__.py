"""
Test package for the SOP Q&A Tool.

This package contains unit tests, integration tests, and test utilities
for validating the functionality of the SOP Q&A system.

Test Categories:
- Unit tests: Individual component testing (test_*.py)
- Integration tests: End-to-end pipeline testing (test_integration_*.py)
- Performance tests: Memory usage and response time validation (test_performance.py)
- Mode switching tests: AWS/local compatibility (test_mode_switching.py)
- Error scenario tests: Network failures and resource limits (test_error_scenarios.py)
- Acceptance tests: Requirements validation (test_acceptance_requirements.py)

Usage:
    # Run all integration tests
    python -m pytest tests/test_integration_*.py tests/test_performance.py tests/test_mode_switching.py tests/test_error_scenarios.py tests/test_acceptance_requirements.py
    
    # Run specific test categories
    python -m pytest tests/test_performance.py -m performance
    python -m pytest tests/test_acceptance_requirements.py
    
    # Use the test runner scripts
    python scripts/run_integration_tests.py all
    .\\scripts\\run-integration-tests.ps1 smoke
"""
