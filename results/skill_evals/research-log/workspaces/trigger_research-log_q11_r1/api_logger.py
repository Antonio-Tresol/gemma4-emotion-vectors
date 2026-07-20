import logging
import json
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Any, Dict
from datetime import datetime
import sys


class APILogFormatter(logging.Formatter):
    """Custom formatter for API requests/responses with structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        if record.name.startswith("api."):
            return self._format_structured(record)
        return super().format(record)

    def _format_structured(self, record: logging.LogRecord) -> str:
        """Format as JSON for structured logging."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "request_data"):
            log_data["request"] = record.request_data
        if hasattr(record, "response_data"):
            log_data["response"] = record.response_data
        if hasattr(record, "duration"):
            log_data["duration_ms"] = record.duration
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


class APILogger:
    """Logging wrapper for API clients with automatic log rotation."""

    def __init__(
        self,
        name: str = "api",
        log_dir: str = "./logs",
        level: int = logging.INFO,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console: bool = True,
    ):
        """
        Initialize API logger with file rotation.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level (default: INFO)
            max_bytes: Max size per log file before rotation (default: 10MB)
            backup_count: Number of backup files to keep (default: 5)
            console: Whether to also log to console (default: True)
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        self._add_file_handler(max_bytes, backup_count)
        if console:
            self._add_console_handler(level)

    def _add_file_handler(self, max_bytes: int, backup_count: int) -> None:
        """Add rotating file handler."""
        log_file = self.log_dir / f"{self.name}.log"
        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setFormatter(APILogFormatter())
        self.logger.addHandler(handler)

    def _add_console_handler(self, level: int) -> None:
        """Add console handler for real-time output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def log_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
    ) -> None:
        """Log API request."""
        request_data = {
            "method": method,
            "url": url,
            "headers": self._sanitize_headers(headers),
            "body": self._truncate(body),
        }
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "(api request)",
            0,
            f"{method} {url}",
            (),
            None,
        )
        record.request_data = request_data
        self.logger.handle(record)

    def log_response(
        self,
        status_code: int,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """Log API response."""
        response_data = {
            "status_code": status_code,
            "headers": self._sanitize_headers(headers),
            "body": self._truncate(body),
        }
        message = f"Response: {status_code}"
        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "(api response)",
            0,
            message,
            (),
            None,
        )
        record.response_data = response_data
        if duration_ms is not None:
            record.duration = duration_ms
        self.logger.handle(record)

    def log_request_response(
        self,
        method: str,
        url: str,
        status_code: int,
        duration_ms: Optional[float] = None,
        request_body: Optional[Any] = None,
        response_body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log both request and response atomically."""
        message = f"{method} {url} -> {status_code}"
        if duration_ms:
            message += f" ({duration_ms:.1f}ms)"

        log_data = {
            "method": method,
            "url": url,
            "status_code": status_code,
            "request": {"body": self._truncate(request_body)},
            "response": {
                "body": self._truncate(response_body),
                "headers": self._sanitize_headers(headers),
            },
        }
        if duration_ms:
            log_data["duration_ms"] = duration_ms

        record = self.logger.makeRecord(
            self.logger.name,
            logging.INFO,
            "(api call)",
            0,
            message,
            (),
            None,
        )
        record.request_data = log_data
        self.logger.handle(record)

    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log API error with context."""
        self.logger.error(message, exc_info=error, extra={"context": context})

    @staticmethod
    def _sanitize_headers(headers: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """Remove sensitive headers before logging."""
        if not headers:
            return None
        sensitive_keys = {
            "authorization",
            "x-api-key",
            "x-auth-token",
            "password",
            "token",
            "secret",
        }
        return {
            k: "***" if k.lower() in sensitive_keys else v
            for k, v in headers.items()
        }

    @staticmethod
    def _truncate(data: Any, max_length: int = 500) -> Any:
        """Truncate large payloads for logging."""
        if data is None:
            return None
        if isinstance(data, str):
            if len(data) > max_length:
                return data[:max_length] + f"... [{len(data)} chars total]"
            return data
        if isinstance(data, dict):
            return {k: APILogger._truncate(v, max_length) for k, v in data.items()}
        if isinstance(data, list):
            if len(data) > 10:
                return data[:10] + [f"... [{len(data)} items total]"]
            return [APILogger._truncate(item, max_length) for item in data]
        return data

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False) -> None:
        """Log error message."""
        self.logger.error(message, exc_info=exc_info)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)


class APIClientWrapper:
    """Context manager for timing and logging API calls."""

    def __init__(self, logger: APILogger, method: str, url: str):
        self.logger = logger
        self.method = method
        self.url = url
        self.start_time = None
        self.duration_ms = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_request(self.method, self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            self.logger.log_error(
                f"API call failed: {self.method} {self.url}",
                error=exc_val,
            )
            return False

        return True

    def log_response(
        self,
        status_code: int,
        body: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log response within context."""
        self.logger.log_response(
            status_code,
            headers=headers,
            body=body,
            duration_ms=self.duration_ms,
        )
