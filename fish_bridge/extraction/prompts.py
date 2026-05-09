"""Extraction prompt templates for fish_bridge."""
from __future__ import annotations


EXTRACTION_SYSTEM = (
    "You are a knowledge graph extractor for AI coding assistant sessions.\n"
    "Analyze the provided AI coding assistant exchange and extract all semantic entities "
    "as a structured graph.\n\n"
    "NODE TYPES:\n"
    "- question: something asked, uncertain, or needing resolution\n"
    "- decision: architectural/technical/process choice made or discussed\n"
    "- concept: key idea, term, pattern, or abstraction introduced\n"
    "- skill: tool, technique, API, library, or pattern being used or learned\n"
    "- file: source code file, function, class, module, or symbol referenced\n"
    "- error: bug, exception, test failure, or problem encountered\n"
    "- task: concrete work item identified, planned, or tracked\n\n"
    "EDGE RELATIONS:\n"
    "resolves, depends-on, leads-to, contradicts, uses, blocks, supersedes,\n"
    "created-by, references, documents, tested-by, configures, imports, implements, relates-to\n\n"
    "STATUS VALUES per node type:\n"
    "- question: active | resolved | deferred\n"
    "- decision: proposed | adopted | rejected | superseded\n"
    "- concept/skill: active\n"
    "- file: active\n"
    "- error: active | fixed | deferred\n"
    "- task: pending | in_progress | done | blocked | deferred\n\n"
    "RULES:\n"
    "1. Extract only entities clearly grounded in the exchange text.\n"
    "2. Do not invent entities that are not mentioned or strongly implied.\n"
    "3. Assign confidence 0.0-1.0 based on how explicitly the entity appears.\n"
    "4. Labels must be concise (8 words max). Summaries 2 sentences max.\n"
    "5. For edges, use the exact label strings you assigned to the nodes.\n"
    "6. MANDATORY EDGE RULES:\n"
    "   a. Every extracted node MUST have at least 2 edges. Isolated nodes are not useful.\n"
    "   b. 'relates-to' is the LAST RESORT. Use specific relations first:\n"
    "      task → blocks question | task → depends-on skill | task → implements decision\n"
    "      error → created-by file | error → resolves ← task | decision → supersedes decision\n"
    "      skill → used-by ← task | concept → documents ← decision | file → implements decision\n"
    "   c. If a node has only 1 natural edge, add a second via the most specific applicable relation.\n"
    "   d. 'relates-to' must be < 30% of all edges. If you exceed that, replace with specific types.\n"
    "7. SPECULATIVE ENTITY RULE: If an entity is discussed hypothetically ('might use', \n"
    "   'future version', 'not yet released', 'we could'), set confidence ≤ 0.5 and \n"
    "   add metadata: {\"speculative\": true}. Model names not yet in production (e.g. \n"
    "   version numbers that seem futuristic) must also be marked speculative.\n"
)

EXTRACTION_USER_TEMPLATE = (
    "EXCHANGE:\n"
    "User: {user_message}\n\n"
    "Assistant: {assistant_message}\n\n"
    "Return ONLY valid JSON in this exact shape:\n"
    '{{"nodes": [...], "edges": [...]}}\n\n'
    "Node shape:\n"
    '{{"type":"<type>","label":"<label>","summary":"<summary>",'
    '"status":"<status>","confidence":<0.0-1.0>,'
    '"subtype":"<optional>","source_url":"<optional>","metadata":{{}}}}\n\n'
    "Edge shape:\n"
    '{{"from_label":"<label>","to_label":"<label>","relation":"<relation>","weight":1.0}}\n\n'
    "EDGE CHECKLIST before returning:\\n"
    "- Count edges per node. Any node with < 2 edges? Add more using specific relations.\\n"
    "- Count 'relates-to' edges. More than 30% of total? Replace with specific types.\\n"
    "Do not include any text outside the JSON object."
)

# JSON Schema for structured output enforcement (tool_use / json_schema backends)
EXTRACTION_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["nodes", "edges"],
    "additionalProperties": False,
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "label", "summary", "status", "confidence"],
                "additionalProperties": True,
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["question", "decision", "concept", "skill", "file", "error", "task"],
                    },
                    "label":      {"type": "string"},
                    "summary":    {"type": "string"},
                    "status":     {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "subtype":    {"type": "string"},
                    "source_url": {"type": "string"},
                    "metadata":   {"type": "object"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from_label", "to_label", "relation"],
                "additionalProperties": True,
                "properties": {
                    "from_label": {"type": "string"},
                    "to_label":   {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": [
                            "resolves", "depends-on", "leads-to", "contradicts",
                            "uses", "blocks", "supersedes", "created-by", "references",
                            "documents", "tested-by", "configures", "imports",
                            "implements", "relates-to",
                        ],
                    },
                    "weight": {"type": "number"},
                },
            },
        },
    },
}
