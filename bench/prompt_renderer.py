"""Shared prompt-template loading and placeholder rendering."""

import re
from pathlib import Path
from typing import Any, Mapping


PROMPT_ROOT = Path(__file__).parent / "prompts"
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def render_template(template: str, values: Mapping[str, Any]) -> str:
    """Replace all named placeholders and reject unknown placeholders."""

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in values:
            raise ValueError(f"unknown prompt placeholder: {key}")
        return str(values[key])

    return _PLACEHOLDER_PATTERN.sub(replace, template).strip()


def render_prompt_file(task_code: str, values: Mapping[str, Any], *, variant: str | None = None) -> str:
    """Load and render a UTF-8 prompt template for a task."""
    template_name = f"{task_code}.{variant}" if variant else task_code
    template_path = PROMPT_ROOT / f"{template_name}.txt"
    try:
        template = template_path.read_text(encoding="utf-8-sig")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"prompt template not found for {task_code}: {template_path}") from exc
    return render_template(template, values)
