# ptx_fusion/parser.py
"""
PTX Parser — converts raw PTX text into a structured IR.
Handles: .reg declarations, .param, .shared, basic blocks, instructions.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto

# ─────────────────────────────────────────────────────────────
#  IR DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

class RegType(Enum):
    PRED  = ".pred"
    B16   = ".b16"
    B32   = ".b32"
    B64   = ".b64"
    U16   = ".u16"
    U32   = ".u32"
    U64   = ".u64"
    S32   = ".s32"
    S64   = ".s64"
    F32   = ".f32"
    F64   = ".f64"

@dataclass
class RegDecl:
    """A .reg directive: .reg .f32 %f<8>;"""
    reg_type: str       # ".f32"
    name:     str       # "%f"
    count:    int       # 8 → %f0..%f7  (or 1 for scalar)

@dataclass
class Instruction:
    """One PTX instruction with optional label and predicate."""
    label:    Optional[str]   # "BB0_1:" if present
    pred:     Optional[str]   # "@%p1" or "@!%p1"
    opcode:   str             # "ld.global.f32", "fma.rn.f32", etc.
    operands: List[str]       # ["%f1", "[%rd0]", "%f2", "%f3"]
    comment:  str = ""        # anything after //
    raw:      str = ""        # original text (for debug)

@dataclass
class BasicBlock:
    label:        str
    instructions: List[Instruction] = field(default_factory=list)
    successors:   List[str]         = field(default_factory=list)  # label names

@dataclass
class PTXKernel:
    name:        str
    params:      List[str]                    # raw param lines
    reg_decls:   List[RegDecl]
    blocks:      Dict[str, BasicBlock]        # label → BasicBlock
    entry_label: str                          # first block label
    shared_vars: List[str]                    # .shared declarations

# ─────────────────────────────────────────────────────────────
#  PARSER
# ─────────────────────────────────────────────────────────────

# Regex patterns
RE_REG_ARRAY  = re.compile(r'\.reg\s+(\.\w+)\s+(%\w+)<(\d+)>\s*;')
RE_REG_SCALAR = re.compile(r'\.reg\s+(\.\w+)\s+(%\w+)\s*;')
RE_LABEL      = re.compile(r'^(\w+)\s*:')
RE_PARAM      = re.compile(r'\.param\s+.*')
RE_SHARED     = re.compile(r'\.shared\s+.*')
RE_PRED_INST  = re.compile(r'^(@[!]?%\w+)\s+(\S+)\s*(.*)')
RE_PLAIN_INST = re.compile(r'^(\S+)\s*(.*)')
RE_KERNEL_SIG = re.compile(r'\.visible\s+\.entry\s+(\w+)\s*\(')

def parse_operands(operand_str: str) -> List[str]:
    """Split 'a, [b+8], c' → ['a', '[b+8]', 'c'] respecting brackets."""
    operands, buf, depth = [], "", 0
    for ch in operand_str.strip().rstrip(';'):
        if ch == '[': depth += 1
        if ch == ']': depth -= 1
        if ch == ',' and depth == 0:
            operands.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        operands.append(buf.strip())
    return operands

def parse_ptx(ptx_text: str) -> PTXKernel:
    """Full PTX kernel parser → PTXKernel IR."""
    lines      = ptx_text.splitlines()
    name       = ""
    params     = []
    reg_decls  = []
    shared_vars = []
    blocks: Dict[str, BasicBlock] = {}
    current_block: Optional[BasicBlock] = None
    entry_label = "__entry__"
    in_kernel   = False
    block_counter = 0

    for raw_line in lines:
        # strip comment but keep it
        comment = ""
        if '//' in raw_line:
            idx = raw_line.index('//')
            comment  = raw_line[idx+2:].strip()
            raw_line = raw_line[:idx]
        line = raw_line.strip()
        if not line:
            continue

        # Kernel entry
        m = RE_KERNEL_SIG.search(line)
        if m:
            name = m.group(1)
            in_kernel = True
            # Synthetic entry block
            entry_label = f"__entry_{name}__"
            current_block = BasicBlock(label=entry_label)
            blocks[entry_label] = current_block
            continue

        if not in_kernel:
            continue

        # .param
        if RE_PARAM.match(line):
            params.append(line)
            continue

        # .shared
        if RE_SHARED.match(line):
            shared_vars.append(line)
            continue

        # .reg declaration
        m = RE_REG_ARRAY.match(line)
        if m:
            reg_decls.append(RegDecl(m.group(1), m.group(2), int(m.group(3))))
            continue
        m = RE_REG_SCALAR.match(line)
        if m:
            reg_decls.append(RegDecl(m.group(1), m.group(2), 1))
            continue

        # Label → new BasicBlock
        m = RE_LABEL.match(line)
        if m:
            lbl = m.group(1)
            if lbl not in blocks:
                blocks[lbl] = BasicBlock(label=lbl)
            # Update predecessor's successor list
            if current_block and current_block.label != lbl:
                current_block.successors.append(lbl)
            current_block = blocks[lbl]
            rest = line[m.end():].strip()
            if rest:
                line = rest  # instruction on same line as label
            else:
                continue

        # Instruction
        pred = None
        m = RE_PRED_INST.match(line)
        if m:
            pred, opcode_str, rest = m.group(1), m.group(2), m.group(3)
        else:
            m2 = RE_PLAIN_INST.match(line)
            if not m2:
                continue
            opcode_str, rest = m2.group(1), m2.group(2)

        opcode   = opcode_str
        operands = parse_operands(rest)

        if current_block is None:
            current_block = BasicBlock(label=f"__bb{block_counter}__")
            blocks[current_block.label] = current_block
            block_counter += 1

        inst = Instruction(
            label    = None,
            pred     = pred,
            opcode   = opcode,
            operands = operands,
            comment  = comment,
            raw      = raw_line
        )
        current_block.instructions.append(inst)

        # Track explicit branches → successors
        if opcode in ('bra', 'bra.uni'):
            target = operands[0].strip() if operands else ""
            current_block.successors.append(target)

        # End of kernel
        if line.strip() == '}' and in_kernel:
            in_kernel = False

    return PTXKernel(
        name        = name,
        params      = params,
        reg_decls   = reg_decls,
        blocks      = blocks,
        entry_label = entry_label,
        shared_vars = shared_vars,
    )
