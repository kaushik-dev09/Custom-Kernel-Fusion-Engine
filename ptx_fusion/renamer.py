# ptx_fusion/renamer.py
"""
Register Renamer — gives every kernel a unique namespace.
Prevents %f0 from kernel A colliding with %f0 from kernel B.
Strategy: prefix all register names with a kernel-specific suffix.
  %f0 → %f_kA_0,  %rd2 → %rd_kB_2
"""

import copy, re
from .parser import PTXKernel, RegDecl, Instruction

# Matches register references: %f0, %rd3, %p1, etc.
RE_REG_REF = re.compile(r'(%\w+)')

def rename_register(reg: str, suffix: str) -> str:
    """
    '%f0'   → '%f_kA_0'
    '%rd12' → '%rd_kA_12'
    Special PTX regs (%tid, %ntid, %ctaid, etc.) are NOT renamed.
    """
    SPECIAL = {
        '%tid', '%ntid', '%ctaid', '%nctaid',
        '%laneid', '%warpid', '%nwarpid',
        '%smid', '%gridid', '%clock', '%clock64',
        '%lanemask_lt', '%lanemask_le', '%lanemask_gt', '%lanemask_ge',
    }
    # Special regs with .x/.y/.z
    if any(reg.startswith(s) for s in SPECIAL):
        return reg

    # Extract base name and index: %f0 → base="%f", idx="0"
    m = re.match(r'(%[a-zA-Z_]+)(\d+)$', reg)
    if m:
        base, idx = m.group(1), m.group(2)
        return f"{base}_{suffix}_{idx}"

    # Scalar without index: %p → %p_kA
    return f"{reg}_{suffix}"

def rename_in_string(s: str, suffix: str) -> str:
    """Replace all register references inside operand/predicate strings."""
    def replacer(m):
        return rename_register(m.group(0), suffix)
    return RE_REG_REF.sub(replacer, s)

def apply_rename_to_kernel(kernel: PTXKernel, suffix: str) -> PTXKernel:
    """Deep-copy kernel and rename all registers with `suffix`."""
    k = copy.deepcopy(kernel)
    k.name = f"{kernel.name}_{suffix}"

    # Rename declarations
    new_decls = []
    for d in k.reg_decls:
        new_name = rename_register(d.name, suffix)
        new_decls.append(RegDecl(d.reg_type, new_name, d.count))
    k.reg_decls = new_decls

    # Rename all instruction operands and predicates
    for block in k.blocks.values():
        for inst in block.instructions:
            if inst.pred:
                inst.pred = rename_in_string(inst.pred, suffix)
            inst.operands = [rename_in_string(op, suffix) for op in inst.operands]

    # Rename block labels (except special PTX labels)
    SKIP_LABELS = {'__EXIT__', 'EXIT', 'exit'}
    old_blocks = k.blocks
    k.blocks = {}
    for lbl, block in old_blocks.items():
        new_lbl = f"{lbl}_{suffix}" if lbl not in SKIP_LABELS else lbl
        block.label = new_lbl
        block.successors = [
            f"{s}_{suffix}" if s not in SKIP_LABELS else s
            for s in block.successors
        ]
        k.blocks[new_lbl] = block

    k.entry_label = f"{kernel.entry_label}_{suffix}"
    return k
