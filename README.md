# Jira MCP Streamlit Client

Streamlit-based chat client that answers Jira queries using:

- MCP tools and prompts from <http://localhost:8000/sse>
- LangGraph workflow orchestration
- Local Ollama model llama3.2:latest

## Implemented Workflows

1. **Get Sprint Insights** — summarises active sprint progress for a named board.
2. **Get Issue Details** — fetches and formats details for a specific Jira issue key.
3. **Get Team Status** — interactive multi-turn collection of per-assignee status updates, followed by a synthesised scrum master report.

Routing is automatic based on user intent.

## Architecture

- Centralized MCP client: src/client/mcp_client.py
- Centralized structured logging: src/logging_config.py
- LangGraph router: src/graph/router.py
- Workflow nodes:
  - src/graph/workflows/sprint_insights.py
  - src/graph/workflows/issue_details.py
  - src/graph/workflows/team_status.py
- Streamlit UI: src/ui/app.py

All tool and prompt calls flow through the centralized MCP client.

### Team Status Workflow

The team status workflow is a multi-turn interactive collection cycle:

1. Fetches all active sprint issues for the board and groups them by assignee.
2. Presents each assignee's issues in turn, prompting them to provide a status update.
3. After all assignees have responded, generates a synthesised scrum master report via the LLM.

Session state (`team_status_session`) is persisted in Streamlit across turns so the collection loop survives multiple user interactions.

## MCP Server Assumption

MCP server is expected at:

<http://localhost:8000/sse>

Expected tools:

- `get_active_sprint_issues`
- `get_issue_details`

Expected prompts:

- `format_issue_details`
- `format_sprint_progress`

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure Ollama is running and the model is available:

   ```bash
   ollama pull llama3.2:latest
   ```

3. Ensure the MCP server is running at the configured endpoint.

## Configuration

Runtime behaviour can be tuned via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MCP_SERVER_URL` | `http://localhost:8000/sse` | MCP server endpoint |
| `OLLAMA_MODEL` | `llama3.2:latest` | Ollama model to use |
| `APP_TITLE` | `Jira Copilot` | Streamlit page title |
| `APP_SUBTITLE` | `LangGraph + MCP powered Jira assistant` | Subtitle shown in the UI |
| `STREAMING_ENABLED` | `true` | Enable token-by-token streaming |
| `LOG_LEVEL` | `INFO` | Log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `json` | Log output format (`json` or `text`) |

## Run

```bash
streamlit run src/main.py
```

## Usage Examples

- `Show sprint progress for board Platform Team`
- `Give me issue details for PROJ-42`
- `Get team status for Platform Team` — starts the interactive status collection cycle

## Notes

- Board name is extracted from the chat request for sprint and team status workflows; if it cannot be parsed, the LLM is asked to infer it.
- If board name or issue key is missing, the app asks for clarification.
- UI is configured so the page body is locked and the message panel is scrollable.
- The team status collection loop persists across Streamlit re-runs via `st.session_state.team_status_session`.
