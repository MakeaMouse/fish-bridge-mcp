# Graph Schema

## Node types (`NodeType`)

| Enum | String | Description |
|---|---|---|
| `CONCEPT` | `"concept"` | Abstract idea, pattern, or principle |
| `TASK` | `"task"` | Work item or action |
| `DECISION` | `"decision"` | Architectural or design choice |
| `ERROR` | `"error"` | Bug, failure, or test failure |
| `QUESTION` | `"question"` | Open question or unknown |
| `SKILL` | `"skill"` | Library, tool, or technique |
| `FILE` | `"file"` | Source file or document |

## Node status (`NodeStatus`)

| Enum | String | Terminal? |
|---|---|---|
| `ACTIVE` | `"active"` | No |
| `IN_PROGRESS` | `"in_progress"` | No |
| `PROPOSED` | `"proposed"` | No |
| `PENDING` | `"pending"` | No |
| `BLOCKED` | `"blocked"` | No |
| `ADOPTED` | `"adopted"` | **Yes** |
| `RESOLVED` | `"resolved"` | **Yes** |
| `FIXED` | `"fixed"` | **Yes** |
| `DONE` | `"done"` | **Yes** |
| `REJECTED` | `"rejected"` | **Yes** |
| `DEFERRED` | `"deferred"` | — |
| `CONFLICTED` | `"conflicted"` | — (needs review) |
| `SUPERSEDED` | `"superseded"` | — |
| `UNCONFIRMED` | `"unconfirmed"` | — |

Nodes progress through statuses as conversation turns are ingested. Conflict detection
fires when an incoming turn tries to move a **terminal** node back to an **open** status.

## Edge relations (`EdgeRelation`)

| Enum | String | Direction |
|---|---|---|
| `RELATES_TO` | `"relates-to"` | bidirectional |
| `DEPENDS_ON` | `"depends-on"` | A needs B |
| `LEADS_TO` | `"leads-to"` | A produces B |
| `CONTRADICTS` | `"contradicts"` | A conflicts with B |
| `USES` | `"uses"` | A uses B |
| `BLOCKS` | `"blocks"` | A blocks B |
| `SUPERSEDES` | `"supersedes"` | A replaces B |
| `REPLACED_BY` | `"replaced-by"` | A is replaced by B |
| `CREATED_BY` | `"created-by"` | A created by B |
| `REFERENCES` | `"references"` | A references B |
| `DOCUMENTS` | `"documents"` | A documents B |
| `TESTED_BY` | `"tested-by"` | error → test file |
| `CONFIGURES` | `"configures"` | A configures B |
| `IMPORTS` | `"imports"` | A imports B |
| `IMPLEMENTS` | `"implements"` | A realises B |
| `RESOLVES` | `"resolves"` | A resolves B |

## GraphNode fields

```python
GraphNode(
    id: str            # UUID v4 (auto-assigned)
    type: NodeType
    label: str         # short identifier (≤60 chars)
    summary: str       # 1–2 sentence description
    status: NodeStatus
    confidence: float  # 0.0–1.0
    subtype: str | None        # e.g. "library", "test_failure"
    source_url: str | None     # origin reference (file path, URL)
    metadata: dict             # arbitrary JSON-serialisable data
    created_at: datetime
    updated_at: datetime
    status_history: list[StatusEntry]  # audit trail
    embedding: list[float] | None      # cached sentence embedding
)
```

## Status history

Every status change is appended to `status_history`:

```json
{"status": "adopted", "timestamp": "2025-05-01T12:00:00Z", "note": "accepted by team"}
```
