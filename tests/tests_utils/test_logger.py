"""Tests for logger utility."""
import logging
import pytest

from rosclient.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for setup_logger function."""

    def test_logger_creation(self):
        """Test that logger is created successfully."""
        logger = setup_logger("test_logger")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_logger_has_handler(self):
        """Test that logger has a handler."""
        logger = setup_logger("test_logger_handler")
        assert len(logger.handlers) > 0
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_logger_level(self):
        """Test that logger level is set correctly."""
        logger = setup_logger("test_logger_level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logger_no_propagate(self):
        """Test that logger doesn't propagate to root."""
        logger = setup_logger("test_logger_propagate")
        assert logger.propagate is False

    def test_logger_reuse(self):
        """Test that calling setup_logger twice returns same logger."""
        logger1 = setup_logger("test_logger_reuse")
        logger2 = setup_logger("test_logger_reuse")
        assert logger1 is logger2

    def test_logger_formatter(self):
        """Test that logger has correct formatter."""
        logger = setup_logger("test_logger_formatter")
        handler = logger.handlers[0]
        formatter = handler.formatter
        assert isinstance(formatter, logging.Formatter)
        assert "%(asctime)s" in formatter._fmt
        assert "%(levelname)s" in formatter._fmt
        assert "%(name)s" in formatter._fmt

    def test_different_loggers(self):
        """Test that different names create different loggers."""
        logger1 = setup_logger("logger1")
        logger2 = setup_logger("logger2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name

