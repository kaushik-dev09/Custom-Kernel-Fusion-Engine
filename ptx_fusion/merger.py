# ptx_fusion/merger.py
"""
CFG Merger — implements the GoPTX-style BasicBlock-level CFG merge.

Fusion strategy: VERTICAL (sequential, data-dependent)
  - Kernel A runs to completion first (all its blocks execute)
  - Then Kernel B runs using A's outputs
  - Output registers of A that feed B are mapped via param substitution

For independent (HORIZONTAL) fusion, we'd thread-partition instead.

Algorithm:
  1. Serialize A's CFG (topological order)
  2. Append B's CFG after A's exit block
  3. Inject a "bridge" block that moves A's output regs to B's input regs
  4. Merge all .reg decls (already renamed, no conflicts)
  5. Merge .param sections (keep both, rename)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from .parser import PTXKernel, BasicBlock, Instruction, RegDecl

@dataclass
class FusionSpec:
    """
    Describes how to connect kernel A's output to kernel B's input.
    output_reg: register in A that holds the result (after renaming)
    input_reg:  register in B that expects that value (after renaming)
    """
    output_reg: str   # e.g. "%f_kA_3"  (relu output)
    input_reg:  str   # e.g. "%f_kB_0"  (multiply input)

def topo_sort_blocks(kernel: PTXKernel) -> List[BasicBlock]:
    """Topological ordering of basic blocks via DFS."""
    visited  = set()
    ordering = []

    def dfs(lbl: str):
        if lbl in visited or lbl not in kernel.blocks:
            return
        visited.add(lbl)
        block = kernel.blocks[lbl]
        for succ in block.successors:
            dfs(succ)
        ordering.append(block)

    dfs(kernel.entry_label)
    # Also visit any unreachable blocks (shouldn't happen in valid PTX)
    for lbl in kernel.blocks:
        if lbl not in visited:
            ordering.append(kernel.blocks[lbl])

    ordering.reverse()
    return ordering

def build_bridge_block(
    bridge_label: str,
    specs: List[FusionSpec],
    next_label: str
) -> BasicBlock:
    """
    Creates a bridge BasicBlock that moves values between kernels.
    Emits PTX `mov` instructions: input_reg ← output_reg
    """
    block = BasicBlock(label=bridge_label, successors=[next_label])
    for spec in specs:
        # mov.f32 %input_reg, %output_reg;
        # Infer type from register name (simplified: check if f/d/u/s)
        reg_type = infer_type(spec.output_reg)
        inst = Instruction(
            label    = None,
            pred     = None,
            opcode   = f"mov{reg_type}",
            operands = [spec.input_reg, spec.output_reg],
            comment  = "bridge: A→B value transfer",
        )
        block.instructions.append(inst)
    return block

def infer_type(reg_name: str) -> str:
    """Infer PTX type suffix from register name heuristic."""
    import re
    m = re.match(r'%([a-zA-Z]+)_', reg_name) or re.match(r'%([a-zA-Z]+)', reg_name)
    if not m:
        return '.f32'
    base = m.group(1).lower().lstrip('_')
    MAP = {
        'f': '.f32', 'fd': '.f64', 'd': '.f64',
        'rd': '.u64', 'r': '.u32', 'rs': '.u16',
        'p': '.pred', 'h': '.f16',
    }
    for k, v in MAP.items():
        if base == k:
            return v
    return '.b32'

def merge_kernels(
    kernel_a: PTXKernel,
    kernel_b: PTXKernel,
    fusion_specs: List[FusionSpec],
    fused_name: str = "fused_kernel"
) -> PTXKernel:
    """
    Vertically fuse kernel_a and kernel_b into one PTXKernel.
    fusion_specs: maps A's outputs → B's inputs.
    """
    # ── 1. Merge register declarations ──────────────────
    merged_decls = kernel_a.reg_decls + kernel_b.reg_decls

    # ── 2. Merge params (keep both, rename handled upstream) ─
    merged_params = kernel_a.params + kernel_b.params

    # ── 3. Merge shared vars ────────────────────────────
    merged_shared = kernel_a.shared_vars + kernel_b.shared_vars

    # ── 4. Build combined block set ─────────────────────
    merged_blocks: Dict[str, BasicBlock] = {}

    # All of A's blocks
    a_blocks_ordered = topo_sort_blocks(kernel_a)
    for blk in a_blocks_ordered:
        merged_blocks[blk.label] = blk

    # All of B's blocks
    b_blocks_ordered = topo_sort_blocks(kernel_b)
    for blk in b_blocks_ordered:
        merged_blocks[blk.label] = blk

    # ── 5. Find A's exit block (last in topo order with ret/exit) ──
    a_exit_block = None
    for blk in reversed(a_blocks_ordered):
        for inst in blk.instructions:
            if inst.opcode in ('ret', 'exit', 'bra.uni') and not inst.operands:
                a_exit_block = blk
                break
        if a_exit_block:
            break
    if a_exit_block is None:
        a_exit_block = a_blocks_ordered[-1]

    # ── 6. Remove `ret` from A's exit (don't return early) ──────
    a_exit_block.instructions = [
        i for i in a_exit_block.instructions
        if i.opcode not in ('ret', 'exit')
    ]

    # ── 7. Insert bridge block between A and B ───────────
    bridge_label = "__bridge__"
    b_entry = kernel_b.entry_label
    bridge  = build_bridge_block(bridge_label, fusion_specs, b_entry)
    merged_blocks[bridge_label] = bridge

    # A's exit now jumps to bridge
    a_exit_block.successors.append(bridge_label)
    a_exit_block.instructions.append(Instruction(
        label    = None,
        pred     = None,
        opcode   = "bra.uni",
        operands = [bridge_label],
        comment  = "jump to bridge → kernel B",
    ))

    # ── 8. Assemble final PTXKernel ──────────────────────
    return PTXKernel(
        name        = fused_name,
        params      = merged_params,
        reg_decls   = merged_decls,
        blocks      = merged_blocks,
        entry_label = kernel_a.entry_label,
        shared_vars = merged_shared,
    )
