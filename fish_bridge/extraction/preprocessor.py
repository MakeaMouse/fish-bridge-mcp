"""Content-zone pre-processor for extraction pipeline step [0.5].

Detects structured content zones in a chat turn BEFORE passing to the LLM:
  A2 — fenced code blocks  → extract language + symbols (regex, no tree-sitter)
  A3 — stack traces        → extract exception_type, file, line
  A4 — terminal/CLI output → classify tool, exit_code, test signals
  A5 — JSON/YAML blobs     → flatten to schema key list
  A6 — file path refs      → extract path + line
  A7 — URLs                → collect source_url candidates

Output: a StructuredHints object attached to the turn before extraction.
The hints are injected into the LLM prompt as additional context to guide
node type and metadata placement.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CodeBlock:
    language:  str
    content:   str
    symbols:   list[str] = field(default_factory=list)


@dataclass
class StackTrace:
    exception_type: str
    message:        str
    file:           str | None = None
    line:           int | None = None
    http_status:    int | None = None


@dataclass
class CliOutput:
    tool:       str        # "pytest" | "jest" | "npm" | "aws" | "generic"
    exit_code:  int | None = None
    test_name:  str | None = None
    passed:     int | None = None
    failed:     int | None = None
    signal_text: str       = ""   # short extracted summary


@dataclass
class SchemaBlob:
    keys:        list[str]
    data_source: str = "unknown"   # "json" | "yaml"


@dataclass
class FileRef:
    path:     str
    line:     int | None = None
    language: str | None = None


@dataclass
class StructuredHints:
    code_blocks:  list[CodeBlock]  = field(default_factory=list)
    stack_traces: list[StackTrace] = field(default_factory=list)
    cli_outputs:  list[CliOutput]  = field(default_factory=list)
    schema_blobs: list[SchemaBlob] = field(default_factory=list)
    file_refs:    list[FileRef]    = field(default_factory=list)
    urls:         list[str]        = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([
            self.code_blocks, self.stack_traces, self.cli_outputs,
            self.schema_blobs, self.file_refs, self.urls,
        ])

    def to_prompt_section(self) -> str:
        """Render hints as a structured section to prepend to the extraction prompt."""
        if self.is_empty():
            return ""
        parts = ["## Structured Content Detected\n"]

        for cb in self.code_blocks:
            syms = ", ".join(cb.symbols[:10]) if cb.symbols else "(no symbols detected)"
            parts.append(
                f"- CODE_BLOCK language={cb.language} symbols=[{syms}]"
                f" ({len(cb.content)} chars)"
            )

        for st in self.stack_traces:
            loc = f" at {st.file}:{st.line}" if st.file else ""
            http = f" HTTP {st.http_status}" if st.http_status else ""
            parts.append(
                f"- STACK_TRACE exception={st.exception_type}{http}{loc}"
                f" msg=\"{st.message[:80]}\""
            )

        for co in self.cli_outputs:
            ec = f" exit={co.exit_code}" if co.exit_code is not None else ""
            tc = ""
            if co.passed is not None or co.failed is not None:
                tc = f" passed={co.passed} failed={co.failed}"
            parts.append(f"- CLI_OUTPUT tool={co.tool}{ec}{tc} \"{co.signal_text[:80]}\"")

        for sb in self.schema_blobs:
            parts.append(f"- SCHEMA_BLOB source={sb.data_source} keys=[{', '.join(sb.keys[:15])}]")

        for fr in self.file_refs:
            lang = f" ({fr.language})" if fr.language else ""
            line = f":{fr.line}" if fr.line else ""
            parts.append(f"- FILE_REF {fr.path}{line}{lang}")

        if self.urls:
            parts.append(f"- URLS {', '.join(self.urls[:5])}")

        parts.append("")  # trailing newline
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Fenced code blocks: ```lang\n...\n```
_FENCED_BLOCK = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

# Stack trace patterns (Python, JavaScript, Java, Go)
_PYTHON_TRACEBACK = re.compile(
    r'(?:Traceback \(most recent call last\):.*?)'
    r'(\w+(?:\.\w+)*Error|\w+Exception|KeyboardInterrupt):\s*(.+)',
    re.DOTALL
)
_PYTHON_FILE_LINE = re.compile(r'File "([^"]+)", line (\d+)')
_JS_ERROR = re.compile(r'(TypeError|ReferenceError|SyntaxError|Error):\s*(.+?)(?:\n|$)')
_JS_STACK_FRAME = re.compile(r'at .+?\((.+?):(\d+):\d+\)')
_HTTP_STATUS = re.compile(r'\b(4\d\d|5\d\d)\b')

# Terminal / CLI signals
_PYTEST_SUMMARY = re.compile(
    r'=+ ([\d,]+ passed|[\d,]+ failed|[\d,]+ error)', re.IGNORECASE
)
_JEST_SUMMARY = re.compile(r'Tests:\s+(\d+) passed.*?(\d+) failed', re.IGNORECASE)
_NPM_ERR = re.compile(r'npm ERR!', re.IGNORECASE)
_AWS_ARN = re.compile(r'arn:aws:[a-z0-9-]+:[a-z0-9-]*:\d+:[A-Za-z0-9:/_-]+')
_EXIT_CODE = re.compile(r'exited? (?:with )?(?:code )?(\d+)', re.IGNORECASE)

# File path detection (unix/windows style, with optional :line)
_FILE_PATH = re.compile(
    r'(?:^|[\s`"\'])((\.{1,2}/|/|[A-Za-z]:[/\\])[A-Za-z0-9._/\\-]+\.[A-Za-z]{1,6})'
    r'(?::(\d+))?',
    re.MULTILINE
)

# URL detection
_URL = re.compile(r'https?://[^\s\'"<>)]+')

# Extension → language mapping
_EXT_LANG = {
    "py": "python", "js": "javascript", "ts": "typescript",
    "jsx": "javascriptreact", "tsx": "typescriptreact",
    "go": "go", "rs": "rust", "java": "java", "rb": "ruby",
    "sh": "bash", "bash": "bash", "zsh": "bash", "fish": "fish",
    "tf": "hcl", "yaml": "yaml", "yml": "yaml",
    "json": "json", "toml": "toml", "sql": "sql",
    "md": "markdown", "html": "html", "css": "css",
    "c": "c", "cpp": "cpp", "cs": "csharp",
}

# Symbol extraction patterns per language (function/class/const defs)
_SYMBOL_PATTERNS: dict[str, re.Pattern] = {
    "python": re.compile(r'^(?:async )?def\s+(\w+)|^class\s+(\w+)', re.MULTILINE),
    "javascript": re.compile(
        r'(?:function\s+(\w+)|const\s+(\w+)\s*=.*?(?:=>|\()|class\s+(\w+))', re.MULTILINE
    ),
    "typescript": re.compile(
        r'(?:function\s+(\w+)|const\s+(\w+)\s*=.*?(?:=>|\()|class\s+(\w+)|interface\s+(\w+))',
        re.MULTILINE
    ),
    "go": re.compile(r'^func\s+(?:\(\w+ \*?\w+\)\s+)?(\w+)', re.MULTILINE),
    "rust": re.compile(r'^(?:pub )?fn\s+(\w+)|^(?:pub )?struct\s+(\w+)', re.MULTILINE),
    "java": re.compile(r'(?:public|private|protected)?\s*(?:static\s+)?[\w<>[\]]+\s+(\w+)\s*\(', re.MULTILINE),
}


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_symbols(code: str, language: str) -> list[str]:
    """Extract function/class names from code using language-specific patterns."""
    pattern = _SYMBOL_PATTERNS.get(language)
    if not pattern:
        return []
    symbols = []
    for m in pattern.finditer(code):
        name = next((g for g in m.groups() if g), None)
        if name and name not in symbols:
            symbols.append(name)
    return symbols[:20]  # cap to 20 per block


def _extract_code_blocks(text: str) -> list[CodeBlock]:
    blocks = []
    for m in _FENCED_BLOCK.finditer(text):
        lang_raw = m.group(1).strip().lower() or "text"
        content  = m.group(2)
        # Normalize lang alias
        lang = _EXT_LANG.get(lang_raw, lang_raw)
        symbols = _extract_symbols(content, lang)
        blocks.append(CodeBlock(language=lang, content=content, symbols=symbols))
    return blocks


def _extract_stack_traces(text: str) -> list[StackTrace]:
    traces = []

    # Python tracebacks
    for m in _PYTHON_TRACEBACK.finditer(text):
        exc_type = m.group(1)
        message  = m.group(2).strip()[:200]
        file_match = None
        line_num = None
        for fm in _PYTHON_FILE_LINE.finditer(text):
            file_match = fm.group(1)
            line_num   = int(fm.group(2))
        # HTTP status code in surrounding text
        http_m = _HTTP_STATUS.search(message)
        http_status = int(http_m.group(1)) if http_m else None
        traces.append(StackTrace(
            exception_type=exc_type,
            message=message,
            file=file_match,
            line=line_num,
            http_status=http_status,
        ))

    # JavaScript errors
    if not traces:
        for m in _JS_ERROR.finditer(text):
            exc_type = m.group(1)
            message  = m.group(2).strip()[:200]
            file_m   = _JS_STACK_FRAME.search(text)
            file_ref = file_m.group(1) if file_m else None
            line_num = int(file_m.group(2)) if file_m else None
            traces.append(StackTrace(
                exception_type=exc_type,
                message=message,
                file=file_ref,
                line=line_num,
            ))

    return traces


def _classify_cli(text: str) -> list[CliOutput]:
    outputs = []

    # Pytest
    for m in _PYTEST_SUMMARY.finditer(text):
        signal = m.group(0).strip()
        # Count passed/failed
        p = re.search(r'(\d+) passed', signal)
        f = re.search(r'(\d+) failed', signal)
        ec_m = _EXIT_CODE.search(text)
        outputs.append(CliOutput(
            tool="pytest",
            exit_code=int(ec_m.group(1)) if ec_m else None,
            passed=int(p.group(1)) if p else None,
            failed=int(f.group(1)) if f else None,
            signal_text=signal,
        ))

    # Jest
    for m in _JEST_SUMMARY.finditer(text):
        passed = int(m.group(1))
        failed = int(m.group(2))
        outputs.append(CliOutput(
            tool="jest",
            passed=passed,
            failed=failed,
            signal_text=f"{passed} passed, {failed} failed",
        ))

    # npm errors
    if _NPM_ERR.search(text) and not outputs:
        outputs.append(CliOutput(tool="npm", signal_text="npm error detected"))

    # AWS CLI (presence of ARN usually means successful resource creation/describe)
    arns = _AWS_ARN.findall(text)
    if arns:
        outputs.append(CliOutput(
            tool="aws",
            signal_text=f"AWS ARNs found: {arns[0]}{'...' if len(arns)>1 else ''}",
        ))

    return outputs


def _extract_schema_blobs(text: str) -> list[SchemaBlob]:
    """Detect and flatten JSON/YAML blobs to their key lists."""
    blobs = []

    # JSON objects/arrays embedded in backticks or as fenced blocks
    for m in re.finditer(r'```json\n(.*?)```', text, re.DOTALL):
        try:
            obj = json.loads(m.group(1))
            keys = _flatten_keys(obj)
            if len(keys) >= 3:
                blobs.append(SchemaBlob(keys=keys, data_source="json"))
        except (json.JSONDecodeError, ValueError):
            pass

    # Inline JSON (heuristic: starts with { or [, at least 50 chars)
    for m in re.finditer(r'(?<!\w)(\{[^`]{50,}?\})', text, re.DOTALL):
        snippet = m.group(1)
        try:
            obj = json.loads(snippet)
            keys = _flatten_keys(obj)
            if len(keys) >= 3:
                blobs.append(SchemaBlob(keys=keys, data_source="json"))
        except (json.JSONDecodeError, ValueError):
            pass

    return blobs[:3]  # cap at 3 blobs per turn


def _flatten_keys(obj: Any, prefix: str = "", depth: int = 0) -> list[str]:
    """Recursively collect keys from a JSON object (max depth 3)."""
    if depth > 3:
        return []
    keys = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            full = f"{prefix}.{k}" if prefix else k
            keys.append(full)
            keys.extend(_flatten_keys(v, full, depth + 1))
    elif isinstance(obj, list) and obj:
        keys.extend(_flatten_keys(obj[0], prefix, depth + 1))
    return keys[:30]


def _extract_file_refs(text: str) -> list[FileRef]:
    """Find file path references in text."""
    seen: set[str] = set()
    refs = []
    for m in _FILE_PATH.finditer(text):
        path_str = m.group(1)
        line_str = m.group(3)
        # Filter out obviously non-file matches
        if len(path_str) < 4 or path_str.endswith("/"):
            continue
        if path_str in seen:
            continue
        seen.add(path_str)
        ext = PurePosixPath(path_str).suffix.lstrip(".")
        lang = _EXT_LANG.get(ext)
        refs.append(FileRef(
            path=path_str,
            line=int(line_str) if line_str else None,
            language=lang,
        ))
    return refs[:10]


def _extract_urls(text: str) -> list[str]:
    seen: set[str] = set()
    urls = []
    for m in _URL.finditer(text):
        url = m.group(0).rstrip(".,;)")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls[:10]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(user_message: str, assistant_message: str) -> StructuredHints:
    """Run the content-zone pre-processor over a turn pair.

    Returns a StructuredHints object. If nothing interesting is detected,
    hints.is_empty() is True and to_prompt_section() returns "".
    """
    combined = f"{user_message}\n\n{assistant_message}"

    code_blocks  = _extract_code_blocks(combined)
    stack_traces = _extract_stack_traces(combined)
    cli_outputs  = _classify_cli(combined)
    schema_blobs = _extract_schema_blobs(combined)
    file_refs    = _extract_file_refs(combined)
    urls         = _extract_urls(combined)

    return StructuredHints(
        code_blocks=code_blocks,
        stack_traces=stack_traces,
        cli_outputs=cli_outputs,
        schema_blobs=schema_blobs,
        file_refs=file_refs,
        urls=urls,
    )
