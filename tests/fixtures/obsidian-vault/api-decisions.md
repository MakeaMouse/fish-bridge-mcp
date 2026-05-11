---
tags: [api, rest, design]
status: active
---
# REST API Design Decisions

Architectural decisions for our REST API layer.

## Decision: Use JWT for authentication
Rationale: Stateless, self-contained, works across microservices. No session state on server.
Alternatives considered: Session cookies (stateful), API keys (no expiry built-in).

## Decision: JSON:API response envelope
Rationale: Consistent pagination, error format, and relationship links.

Related: [[python-architecture]], [[authentication-patterns]]
