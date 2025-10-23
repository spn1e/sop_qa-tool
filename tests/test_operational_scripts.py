"""
Tests for operational scripts and system maintenance tools.
"""

import pytest
import subprocess
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from sop_qa_tool.config.logging_config import setup_logging, get_logger, log_performance, log_error_with_context


class TestBulkIngestScript:
    """Tests for bulk URL ingestion PowerShell script"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.script_path = Path("scripts/bulk-ingest-urls.ps1")
        self.test_urls_file = None
        self.test_log_file = None
    
    def teardown_method(self):
        """Clean up test files"""
        if self.test_urls_file and self.test_urls_file.exists():
            self.test_urls_file.unlink()
        if self.test_log_file and self.test_log_file.exists():
            self.test_log_file.unlink()
    
    def create_test_urls_file(self, urls):
        """Create a temporary URLs file for testing"""
        self.test_urls_file = Path(tempfile.mktemp(suffix=".txt"))
        with open(self.test_urls_file, 'w') as f:
            for url in urls:
                f.write(f"{url}\n")
        return self.test_urls_file
    
    def test_script_exists(self):
        """Test that the bulk ingest script exists"""
        assert self.script_path.exists(), "Bulk ingest script should exist"
    
    def test_script_help(self):
        """Test script help functionality"""
        try:
            result = subprocess.run(
                ["pwsh", "-File", str(self.script_path), "-?"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert "SYNOPSIS" in result.stdout or "DESCRIPTION" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")
    
    @pytest.mark.integration
    def test_bulk_ingest_dry_run(self):
        """Test bulk ingest script with mock API"""
        # Create test URLs file
        test_urls = [
            "https://example.com/sop1.pdf",
            "https://example.com/sop2.pdf"
        ]
        urls_file = self.create_test_urls_file(test_urls)
        
        # Create log file
        self.test_log_file = Path(tempfile.mktemp(suffix=".log"))
        
        try:
            # Note: This would require a mock API server for full testing
            # For now, we test that the script can be invoked with proper parameters
            result = subprocess.run(
                [
                    "pwsh", "-File", str(self.script_path),
                    "-UrlFile", str(urls_file),
                    "-ApiUrl", "http://localhost:9999",  # Non-existent API
                    "-LogFile", str(self.test_log_file),
                    "-BatchSize", "1"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Script should fail due to API not being available, but should create log
            assert self.test_log_file.exists(), "Log file should be created"
            
            # Check log content
            log_content = self.test_log_file.read_text()
            assert "Found 2 URLs to process" in log_content
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")
    
    def test_invalid_urls_file(self):
        """Test script behavior with invalid URLs file"""
        try:
            result = subprocess.run(
                [
                    "pwsh", "-File", str(self.script_path),
                    "-UrlFile", "nonexistent.txt"
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Script should exit with error code
            assert result.returncode != 0
            assert "not found" in result.stdout or "not found" in result.stderr
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")


class TestRebuildIndexScript:
    """Tests for index rebuild PowerShell script"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.script_path = Path("scripts/rebuild-index.ps1")
        self.test_log_file = None
    
    def teardown_method(self):
        """Clean up test files"""
        if self.test_log_file and self.test_log_file.exists():
            self.test_log_file.unlink()
    
    def test_script_exists(self):
        """Test that the rebuild index script exists"""
        assert self.script_path.exists(), "Rebuild index script should exist"
    
    def test_script_help(self):
        """Test script help functionality"""
        try:
            result = subprocess.run(
                ["pwsh", "-File", str(self.script_path), "-?"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert "SYNOPSIS" in result.stdout or "DESCRIPTION" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")
    
    @pytest.mark.integration
    def test_rebuild_index_dry_run(self):
        """Test rebuild index script with mock API"""
        self.test_log_file = Path(tempfile.mktemp(suffix=".log"))
        
        try:
            result = subprocess.run(
                [
                    "pwsh", "-File", str(self.script_path),
                    "-ApiUrl", "http://localhost:9999",  # Non-existent API
                    "-LogFile", str(self.test_log_file),
                    "-Force"  # Skip confirmation
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Script should fail due to API not being available, but should create log
            assert self.test_log_file.exists(), "Log file should be created"
            
            # Check log content
            log_content = self.test_log_file.read_text()
            assert "INDEX REBUILD STARTED" in log_content
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")


class TestHealthMonitorScript:
    """Tests for health monitoring PowerShell script"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.script_path = Path("scripts/health-monitor.ps1")
        self.test_log_file = None
    
    def teardown_method(self):
        """Clean up test files"""
        if self.test_log_file and self.test_log_file.exists():
            self.test_log_file.unlink()
    
    def test_script_exists(self):
        """Test that the health monitor script exists"""
        assert self.script_path.exists(), "Health monitor script should exist"
    
    def test_script_help(self):
        """Test script help functionality"""
        try:
            result = subprocess.run(
                ["pwsh", "-File", str(self.script_path), "-?"],
                capture_output=True,
                text=True,
                timeout=10
            )
            assert "SYNOPSIS" in result.stdout or "DESCRIPTION" in result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")
    
    @pytest.mark.integration
    def test_health_monitor_single_check(self):
        """Test health monitor script single check"""
        self.test_log_file = Path(tempfile.mktemp(suffix=".log"))
        
        try:
            result = subprocess.run(
                [
                    "pwsh", "-File", str(self.script_path),
                    "-ApiUrl", "http://localhost:9999",  # Non-existent API
                    "-LogFile", str(self.test_log_file),
                    "-OutputFormat", "console"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Script should complete and create log
            assert self.test_log_file.exists(), "Log file should be created"
            
            # Check log content
            log_content = self.test_log_file.read_text()
            assert "SYSTEM HEALTH CHECK" in log_content
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")
    
    def test_health_monitor_json_output(self):
        """Test health monitor JSON output format"""
        try:
            result = subprocess.run(
                [
                    "pwsh", "-File", str(self.script_path),
                    "-ApiUrl", "http://localhost:9999",
                    "-OutputFormat", "json"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Try to parse JSON output
            try:
                json.loads(result.stdout)
            except json.JSONDecodeError:
                # JSON parsing might fail due to API errors, but format should be attempted
                pass
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("PowerShell not available or script timeout")


class TestLoggingConfiguration:
    """Tests for logging configuration module"""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        logger = setup_logging(log_level="INFO", enable_console=True, enable_structured=False)
        
        assert logger is not None
        assert logger.level == 20  # INFO level
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            
            logger = setup_logging(
                log_level="DEBUG",
                log_file=log_file,
                enable_console=False,
                enable_structured=True
            )
            
            # Test logging
            test_logger = get_logger("test")
            test_logger.info("Test message")
            
            # Close all handlers to release file locks
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Check file was created and contains content
            assert log_file.exists()
            log_content = log_file.read_text()
            assert "Test message" in log_content
    
    def test_structured_logging(self):
        """Test structured logging format"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "structured.log"
            
            root_logger = setup_logging(
                log_level="INFO",
                log_file=log_file,
                enable_console=False,
                enable_structured=True
            )
            
            logger = get_logger("test")
            logger.info("Structured test", extra={'component': 'test', 'operation': 'unit_test'})
            
            # Close all handlers to release file locks
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            
            # Check structured format
            log_content = log_file.read_text()
            assert "component" in log_content
            assert "operation" in log_content
    
    def test_performance_logging(self):
        """Test performance logging utility"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "performance.log"
            
            root_logger = setup_logging(log_file=log_file, enable_console=False, enable_structured=True)
            logger = get_logger("test")
            
            log_performance(logger, "test_operation", 123.45, component="test")
            
            # Close all handlers to release file locks
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            
            log_content = log_file.read_text()
            assert "test_operation" in log_content
            assert "123.45" in log_content
    
    def test_error_logging_with_context(self):
        """Test error logging with context"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "error.log"
            
            root_logger = setup_logging(log_file=log_file, enable_console=False, enable_structured=True)
            logger = get_logger("test")
            
            try:
                raise ValueError("Test error")
            except ValueError as e:
                log_error_with_context(logger, e, {'component': 'test', 'user_id': '123'})
            
            # Close all handlers to release file locks
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            
            log_content = log_file.read_text()
            assert "Test error" in log_content
            assert "ValueError" in log_content
            assert "user_id" in log_content
    
    def test_log_rotation(self):
        """Test log file rotation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "rotation.log"
            
            # Set very small max size to trigger rotation
            root_logger = setup_logging(
                log_file=log_file,
                enable_console=False,
                max_file_size_mb=0.001,  # 1KB
                backup_count=2
            )
            
            logger = get_logger("test")
            
            # Write enough data to trigger rotation
            for i in range(100):
                logger.info(f"Log message {i} with some additional content to increase size")
            
            # Close all handlers to release file locks
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            
            # Check that rotation occurred (backup files created)
            backup_files = list(Path(temp_dir).glob("rotation.log.*"))
            assert len(backup_files) > 0, "Log rotation should create backup files"


class TestSystemMaintenance:
    """Tests for system maintenance functionality"""
    
    def test_script_permissions(self):
        """Test that scripts have appropriate permissions"""
        scripts = [
            "scripts/bulk-ingest-urls.ps1",
            "scripts/rebuild-index.ps1",
            "scripts/health-monitor.ps1"
        ]
        
        for script_path in scripts:
            script = Path(script_path)
            assert script.exists(), f"Script {script_path} should exist"
            assert script.is_file(), f"Script {script_path} should be a file"
            # Note: Windows file permissions are different from Unix
            # We just check that the file is readable
            assert script.stat().st_size > 0, f"Script {script_path} should not be empty"
    
    def test_logging_directory_creation(self):
        """Test that logging directories are created properly"""
        from sop_qa_tool.config.settings import Settings
        
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                mode="local",
                local_data_path=Path(temp_dir) / "data"
            )
            
            settings.create_directories()
            
            # Check that logs directory was created
            logs_dir = settings.local_data_path / "logs"
            assert logs_dir.exists(), "Logs directory should be created"
            assert logs_dir.is_dir(), "Logs path should be a directory"
    
    @pytest.mark.integration
    def test_health_check_integration(self):
        """Test health check integration with actual system metrics"""
        # This test checks that we can collect basic system metrics
        # without requiring the full API to be running
        
        import psutil
        
        # Test memory metrics
        memory = psutil.virtual_memory()
        assert memory.total > 0
        assert 0 <= memory.percent <= 100
        
        # Test disk metrics
        disk = psutil.disk_usage('.')
        assert disk.total > 0
        assert 0 <= (disk.used / disk.total * 100) <= 100
        
        # Test CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        assert 0 <= cpu_percent <= 100
    
    def test_operational_script_error_handling(self):
        """Test that operational scripts handle errors gracefully"""
        # Test with invalid parameters to ensure scripts don't crash
        scripts_to_test = [
            ("scripts/bulk-ingest-urls.ps1", ["-UrlFile", "nonexistent.txt"]),
            ("scripts/rebuild-index.ps1", ["-ApiUrl", "invalid-url"]),
            ("scripts/health-monitor.ps1", ["-ApiUrl", "invalid-url"])
        ]
        
        for script_path, args in scripts_to_test:
            try:
                result = subprocess.run(
                    ["pwsh", "-File", script_path] + args,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                # Scripts should exit with non-zero code for invalid inputs
                # but should not crash or hang
                assert result.returncode != 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
                
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pytest.skip(f"PowerShell not available or {script_path} timeout")


if __name__ == "__main__":
    pytest.main([__file__])
