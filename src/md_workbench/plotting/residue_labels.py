from __future__ import annotations

import re


_RESIDUE_LABEL_RE = re.compile(
    r"^chain(?P<chain>\d+)_(?P<resname>[A-Za-z0-9]+?)(?P<resseq>-?\d+[A-Za-z]?)$"
)


def compact_residue_label(label: str) -> str:
    text = str(label).strip()
    match = _RESIDUE_LABEL_RE.match(text)
    if not match:
        return text
    chain_alias = f"c{int(match.group('chain'))}"
    resname = match.group("resname").upper()
    resseq = match.group("resseq")
    return f"{chain_alias}:{resname}{resseq}"


def compact_replica_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("replica_"):
        suffix = text.split("_", 1)[1]
        return f"R{suffix}"
    return text
