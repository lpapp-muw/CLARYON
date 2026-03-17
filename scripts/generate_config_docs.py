#!/usr/bin/env python3
"""Auto-generate config_reference.md from Pydantic schema."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claryon.config_schema import ClaryonConfig


def main() -> None:
    schema = ClaryonConfig.model_json_schema()
    defs = schema.get("$defs", {})

    lines = ["# CLARYON Configuration Reference", "", "Auto-generated from Pydantic schema.", ""]

    for name, defn in sorted(defs.items()):
        lines.append(f"## {name}")
        lines.append("")
        desc = defn.get("description", "")
        if desc:
            lines.append(desc)
            lines.append("")
        props = defn.get("properties", {})
        if props:
            lines.append("| Field | Type | Default | Description |")
            lines.append("|-------|------|---------|-------------|")
            for field, info in props.items():
                ftype = info.get("type", info.get("anyOf", ""))
                default = info.get("default", "")
                fdesc = info.get("description", info.get("title", ""))
                lines.append(f"| `{field}` | `{ftype}` | `{default}` | {fdesc} |")
            lines.append("")

    output = Path(__file__).resolve().parent.parent / "docs" / "config_reference_generated.md"
    output.write_text("\n".join(lines))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
