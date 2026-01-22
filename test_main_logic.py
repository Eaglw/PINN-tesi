import os

# Mock options
layers_options = [[2, 50, 1]]
epochs_options = [10]
activation_options = ['GELU'] # Mocked class as string for simplicity
lr_strategies = ['fixed', 'step_decay']

def format_layers_name(layers):
    return "_".join(map(str, layers))

def get_activation_name(act):
    return act

def main():
    print("Testing Grid Search Logic...")
    
    for layers_config in layers_options:
        for epochs in epochs_options:
            for act_fn in activation_options:
                for lr_strat in lr_strategies:
                    
                    # Logic under test
                    layers_str = format_layers_name(layers_config)
                    act_str = get_activation_name(act_fn)
                    config_name = f"L{layers_str}_E{epochs}_{act_str}_{lr_strat}"
                    
                    base_lr = 1e-3
                    if lr_strat == 'step_decay':
                        final_lr = base_lr * (0.5**4) 
                        lr_log_str = f"[{base_lr} -> {final_lr}]"
                    else:
                        lr_log_str = str(base_lr)
                        
                    print(f"Config: {config_name} | LR Log: {lr_log_str}")
                    
                    # Assertions
                    if lr_strat == 'fixed':
                        if not config_name.endswith('_fixed'): print("FAIL: config_name suffix"); exit(1)
                        if lr_log_str != '0.001': print("FAIL: lr_log_str fixed"); exit(1)
                    elif lr_strat == 'step_decay':
                        if not config_name.endswith('_step_decay'): print("FAIL: config_name suffix"); exit(1)
                        if '[' not in lr_log_str: print("FAIL: lr_log_str step_decay format"); exit(1)

    print("Success: Loop and Formatting logic verified.")

if __name__ == "__main__":
    main()
