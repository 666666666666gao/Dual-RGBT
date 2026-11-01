"""Versioned M1 replication entry; preserve the original evaluator's bytes."""
import eval_e0_m1 as evaluator


HEAD_SHA = "d747dc25f9f8acf0abfe137b268b4a6766a3d8785ea0193865297e116351c20e"


if __name__ == "__main__":
    # This fixed head binding requires its own source/plan review and TRAIN smoke.
    evaluator.HEAD_SHA = HEAD_SHA
    evaluator.main()
