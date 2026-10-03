"""56-H800 topology adapters for the audited DLM RL V1/V2 runners.

The 24-GPU runners remain untouched.  This package only changes the distributed
world-size contract and (for V1) enables a DeepSpeed Universal checkpoint
resume so a 24-way ZeRO-1 optimizer can be repartitioned to 56 ranks.
"""
