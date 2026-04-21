"""Application entrypoint for Streamlit."""

from src.config.settings import LOG_FORMAT, LOG_LEVEL
from src.logging_config import configure_logging, get_logger
from src.ui.app import run_app


def main() -> None:
    """Initialize logging and launch the Streamlit application.

    This is the primary entry point for the Jira MCP client.  It bootstraps
    structured logging using the configured log level and format, then
    delegates execution to the Streamlit UI layer.

    The function is idempotent with respect to logging configuration — if
    ``configure_logging`` has already been called, subsequent calls are
    no-ops unless ``force=True`` is passed explicitly.

    Raises:
        SystemExit: Propagated from ``run_app`` if Streamlit exits with a
            non-zero status code.
    """
    configure_logging(LOG_LEVEL, LOG_FORMAT)
    logger = get_logger(__name__)
    logger.info("Starting Jira MCP client")
    run_app()


if __name__ == "__main__":
    main()
