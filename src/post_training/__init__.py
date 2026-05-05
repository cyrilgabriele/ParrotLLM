"""Post-training stages for ParrotLLM.

Implements the alignment pipeline described in VL07 (SFT) and VL08 (RLHF/DPO).
The 2x2 team split defined in VL1 "Roadmap: Group Phases" and VL08 slide 23
places Pair A (Cyril + Tilman) on SFT and Pair B (Gian + Christof) on DPO.
This package exposes both in sibling sub-packages to keep the codebase single
while preserving clear ownership boundaries per the PikoGPT fact sheet.
"""
