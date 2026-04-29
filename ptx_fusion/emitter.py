# ptx_fusion/emitter.py
"""
PTX Emitter — serializes PTXKernel IR back to valid .ptx text.
Performs:
  - Coalescing duplicate .reg declarations
  - Correct block ordering (topo-sort)
  - Valid PTX header/footer generation
"""

from typing import List
from .parser import PTXKernel, RegDecl, Instruction
from .merger import topo_sort_blocks
from collections import defaultdict

def coalesce_reg_decls(decls: List[RegDecl]) -> List[RegDecl]:
    """
    Merge duplicate base names.
    e.g. %f_kA_ count=8 + %f_kA_ count=4  → keep max count.
    Since we rename, there should be no collisions; just deduplicate.
    """
    seen = {}
    for d in decls:
        key = (d.reg_type, d.name)
        if key not in seen or seen[key].count < d.count:
            seen[key] = d
    return list(seen.values())

def emit_instruction(inst: Instruction, indent: str = "\t") -> str:
    pred_str = f"{inst.pred} " if inst.pred else ""
    ops_str  = ", ".join(inst.operands)
    if ops_str:
        line = f"{pred_str}{inst.opcode} {ops_str};"
    else:
        line = f"{pred_str}{inst.opcode};"
    if inst.comment:
        line += f"  // {inst.comment}"
    return f"{indent}{line}"

def emit_ptx(kernel: PTXKernel, sm_version: int = 86) -> str:
    """Emit complete, valid PTX text for the fused kernel."""
    lines = []

    # ── PTX header ──────────────────────────────────────
    lines += [
        f".version 7.5",
        f".target sm_{sm_version}",
        f".address_size 64",
        "",
    ]

    # ── Kernel signature ────────────────────────────────
    lines.append(f".visible .entry {kernel.name}(")
    for i, param in enumerate(kernel.params):
        sep = "," if i < len(kernel.params) - 1 else ""
        lines.append(f"\t{param.strip()}{sep}")
    lines.append(")")
    lines.append("{")

    # ── .reg declarations ───────────────────────────────
    coalesced = coalesce_reg_decls(kernel.reg_decls)
    for d in coalesced:
        if d.count > 1:
            lines.append(f"\t.reg {d.reg_type} {d.name}<{d.count}>;")
        else:
            lines.append(f"\t.reg {d.reg_type} {d.name};")

    if coalesced:
        lines.append("")

    # ── .shared declarations ─────────────────────────────
    for sv in kernel.shared_vars:
        lines.append(f"\t{sv.strip()}")
    if kernel.shared_vars:
        lines.append("")

    # ── Basic blocks (topologically sorted) ─────────────
    ordered = topo_sort_blocks(kernel)
    for block in ordered:
        # Emit label (skip synthetic entry labels)
        if not block.label.startswith("__entry_") and not block.label.startswith("__bb"):
            lines.append(f"{block.label}:")
        for inst in block.instructions:
            lines.append(emit_instruction(inst))
        lines.append("")

    # ── Return and close ────────────────────────────────
    lines.append("\tret;")
    lines.append("}")

    return "\n".join(lines)
