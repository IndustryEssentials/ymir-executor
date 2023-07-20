mining_percent = 0.0
infer_percent = 0.0
run_mining = 0  # 0 or 1
run_infer = 0  # 0 or 1


def get_total_percent() -> float:
    if run_mining == 0 and run_infer == 0:
        return 0

    return (mining_percent * run_mining + infer_percent * run_infer) / (run_mining + run_infer)
