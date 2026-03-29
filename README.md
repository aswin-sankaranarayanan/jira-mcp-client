# Jira MCP Streamlit Client

Streamlit-based chat client that answers Jira queries using:

- MCP tools and prompts from <http://localhost:8000/sse>
- LangGraph workflow orchestration
- Local Ollama model llama3.2:latest

## Implemented Workflows

1. Get Sprint Insights
2. Get Issue Details

Routing is automatic based on user intent.

## Architecture

- Centralized MCP client: src/client/mcp_client.py
- Centralized structured logging: src/logging_config.py
- LangGraph router: src/graph/router.py
- Workflow nodes:
  - src/graph/workflows/sprint_insights.py
  - src/graph/workflows/issue_details.py
- Streamlit UI: src/ui/app.py

All tool and prompt calls flow through the centralized MCP client.

## MCP Server Assumption

MCP server is expected at:

<http://localhost:8000/sse>

Expected tools:

- get_active_sprint_issues
- get_issue_details

Expected prompts:

- format_issue_details
- format_sprint_progress

## Setup

1. Install dependencies:

- `pip install -r requirements.txt`

1. Ensure Ollama is running and model is available:

- `ollama pull llama3.2:latest`

1. Ensure MCP server is running at the configured endpoint.

## Logging

- Logs are emitted through the standard library logger and default to JSON for production-friendly ingestion.
- Configure log output with environment variables:
  - `LOG_LEVEL` defaults to `INFO`
  - `LOG_FORMAT` supports `json` or `text` and defaults to `json`
- Runtime configuration such as `MCP_SERVER_URL`, `OLLAMA_MODEL`, `APP_TITLE`, and `APP_SUBTITLE` can also be provided through environment variables.

## Run

streamlit run src/main.py

## Usage Examples

- Show sprint progress for board Platform Team
- Give me issue details for PROJ-42

## Notes

- Board name is extracted from chat request for sprint workflow.
- If board name or issue key is missing, the app asks for clarification.
- UI is configured so page body is locked and message panel is scrollable.
