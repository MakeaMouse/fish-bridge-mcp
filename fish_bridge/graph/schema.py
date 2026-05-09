"""Pydantic graph schema for fish_bridge: nodes, edges, session graph."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    QUESTION = "question"
    DECISION = "decision"
    CONCEPT  = "concept"
    SKILL    = "skill"
    FILE     = "file"
    ERROR    = "error"
    TASK     = "task"


class NodeStatus(str, Enum):
    # question lifecycle
    ACTIVE       = "active"        # in focus, present in active_thread
    RESOLVED     = "resolved"      # question answered / error fixed
    DEFERRED     = "deferred"      # explicitly parked for later
    # decision lifecycle
    PROPOSED     = "proposed"      # under consideration
    ADOPTED      = "adopted"       # decision confirmed and in use
    REJECTED     = "rejected"      # decision explicitly rejected
    SUPERSEDED   = "superseded"    # replaced by a newer decision
    # error lifecycle
    FIXED        = "fixed"         # error resolved
    # task lifecycle
    PENDING      = "pending"       # not yet started
    IN_PROGRESS  = "in_progress"   # actively being worked on
    DONE         = "done"          # completed
    BLOCKED      = "blocked"       # blocked by something
    CANCELLED    = "cancelled"     # explicitly abandoned (not just deferred)
    # quality
    UNCONFIRMED  = "unconfirmed"   # failed grounding check, needs review
    CONFLICTED   = "conflicted"    # status reversal detected, requires user resolution


class EdgeRelation(str, Enum):
    RESOLVES     = "resolves"
    DEPENDS_ON   = "depends-on"
    LEADS_TO     = "leads-to"
    CONTRADICTS  = "contradicts"
    USES         = "uses"
    BLOCKS       = "blocks"
    SUPERSEDES   = "supersedes"
    REPLACED_BY  = "replaced-by"   # old decision → newer decision that supersedes it
    CREATED_BY   = "created-by"
    REFERENCES   = "references"
    DOCUMENTS    = "documents"
    TESTED_BY    = "tested-by"
    CONFIGURES   = "configures"
    IMPORTS      = "imports"
    IMPLEMENTS   = "implements"
    RELATES_TO   = "relates-to"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StatusHistoryEntry(BaseModel):
    status:    NodeStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note:      str | None = None
    turn:      int | None = None  # turn number that caused this status change


class GraphNode(BaseModel):
    id:             str      = Field(default_factory=lambda: str(uuid4()))
    type:           NodeType
    label:          str
    summary:        str      = ""
    status:         NodeStatus = NodeStatus.ACTIVE
    confidence:     float    = 1.0        # 0.0–1.0; <0.5 → unconfirmed
    subtype:        str | None = None      # e.g. "test_failure", "endpoint", "library"
    source_url:     str | None = None
    metadata:       dict[str, Any] = Field(default_factory=dict)
    status_history: list[StatusHistoryEntry] = Field(default_factory=list)
    embedding:      list[float] | None = None  # stored as JSON text in SQLite
    created_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}

    def touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def push_status(self, new_status: NodeStatus, note: str | None = None, turn: int | None = None) -> None:
        """Append status change to history and update current status."""
        self.status_history.append(
            StatusHistoryEntry(status=self.status, note=note, turn=turn)
        )
        self.status = new_status
        self.touch()


class GraphEdge(BaseModel):
    id:         str      = Field(default_factory=lambda: str(uuid4()))
    from_id:    str
    to_id:      str
    relation:   EdgeRelation
    weight:     float    = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"use_enum_values": True}


class SessionGraph(BaseModel):
    """In-memory representation of a full session graph (serializable to JSON)."""
    session_id:          str
    fish_bridge_version: str = "0.1.0"
    created_at:          datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at:          datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    nodes:               list[GraphNode] = Field(default_factory=list)
    edges:               list[GraphEdge] = Field(default_factory=list)

    model_config = {"use_enum_values": True}


# ---------------------------------------------------------------------------
# RawTurn — normalised ingestor output
# ---------------------------------------------------------------------------

class RawTurn(BaseModel):
    """Normalised user/assistant exchange produced by any ingestor."""
    session_id:    str
    turn_number:   int
    role_user:     str
    role_assistant: str
    source:        str = "unknown"    # e.g. "copilot_jsonl", "paste", "file"
    timestamp:     datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
