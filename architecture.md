Kernel A PTX ──┐
               ├─► [Parser] → [CFG Builder] → [Register Renamer]
Kernel B PTX ──┘                                      │
                                               [CFG Merger]
                                                      │
                                          [Instruction Weaver]
                                                      │
                                           [PTX Emitter] → Fused .ptx
                                                      │
                                          [nvrtc/ptxas JIT] → Run
