"""Application entrypoint for Streamlit."""

from src.config.settings import LOG_FORMAT, LOG_LEVEL
from src.logging_config import configure_logging, get_logger
from src.ui.app import run_app


def main() -> None:
    configure_logging(LOG_LEVEL, LOG_FORMAT)
    logger = get_logger(__name__)
    logger.info("Starting Jira MCP client")
    run_app()


if __name__ == "__main__":
    main()
