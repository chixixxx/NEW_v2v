from __future__ import annotations

from pathlib import Path


def write_markdown_report(
    path: Path,
    *,
    title: str,
    summary_lines: list[str],
    table_rows: list[dict[str, object]],
    extra_tables: list[tuple[str, list[dict[str, object]]]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.extend(summary_lines)
    lines.append("")
    if table_rows:
        fields = list(table_rows[0].keys())
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in table_rows:
            lines.append("| " + " | ".join(_format_value(row[field]) for field in fields) + " |")
    for heading, rows in extra_tables or []:
        lines.extend(["", f"## {heading}", ""])
        if not rows:
            lines.append("无数据。")
            continue
        fields = list(rows[0].keys())
        lines.append("| " + " | ".join(fields) + " |")
        lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(_format_value(row[field]) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
