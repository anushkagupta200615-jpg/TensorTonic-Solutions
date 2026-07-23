def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    
    if warmup_steps > 0 and step < warmup_steps:
        return (step / warmup_steps) * initial_lr

    
    if step <= total_steps:
        denom = max(total_steps - warmup_steps, 1)  # avoid division by zero
        return final_lr + (initial_lr - final_lr) * (total_steps - step) / denom

    
    return final_lr
