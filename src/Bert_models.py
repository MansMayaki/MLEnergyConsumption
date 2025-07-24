import os
import gc
import subprocess
from datetime import datetime, date

import torch
import torch.optim as optim
from torch.cuda.amp import autocast

import pandas as pd
import numpy as np
from thop import profile
from codecarbon import EmissionsTracker
import pynvml

from transformers import BertConfig, BertForMaskedLM, AutoTokenizer

# ──────────────────────────────────────────────────────────────────────────────
# GPU POWER UTILITY & NVML
# ──────────────────────────────────────────────────────────────────────────────
def get_gpu_power():
    cmd = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
    try:
        return float(subprocess.check_output(cmd, shell=True).decode().splitlines()[0])
    except:
        return None

def measure_operation(operation_name, operation, num_repeats,  tracker=None):
    """
    Measures the execution time, estimates energy consumption (nvidia-smi and CodeCarbon),
    and calculates FLOPs for a given operation.

    Args:
        operation_name (str): Name of the operation.
        operation (callable): The function or lambda performing the operation.
        num_repeats (int): Number of times to repeat the operation for measurement.
        flops_fn (callable, optional): Function to calculate FLOPs. Defaults to None.
        tracker (EmissionsTracker, optional): CodeCarbon emissions tracker. Defaults to None.

    Returns:
        dict: A dictionary containing the results (time, estimated energy, codecarbon energy, flops).
    """
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return None

    # Warm-up
    for _ in range(5):
        operation()

#############################################
    elapsed_times,total_energy,total_gpu_energy,total_cpu_energy,total_ram_energy,power_draws,energy_joules_nvmls=[],[],[],[],[],[],[]
    for _ in range(num_repeats):
        torch.cuda.synchronize()
        tracker.start()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        operation()
        end_event.record()
        epoch_emission = tracker.stop()
        power_draw = get_gpu_power()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000  # seconds
        energy_joules_nvml = (power_draw * elapsed_time) if power_draw is not None else None
        energy_watt_sec_nvml = energy_joules_nvml if energy_joules_nvml is not None else None
        
        elapsed_times.append(elapsed_time)
        total_energy.append(tracker._total_energy.kWh)
        total_gpu_energy.append(tracker._total_gpu_energy.kWh)
        total_cpu_energy.append(tracker._total_cpu_energy.kWh)
        total_ram_energy.append(tracker._total_ram_energy.kWh)
        energy_joules_nvmls.append(energy_watt_sec_nvml)
        power_draws.append(power_draw)
    elapsed_time_sec=np.sum(elapsed_times)

###########################################################################################

    results = {
        "operation": operation_name,
        "time_sec": elapsed_time_sec,
        "power_watt_nvml": np.sum(power_draws),
        #"energy_joules_nvml": energy_joules_nvml,
        "energy_joules_nvml": np.sum(energy_joules_nvmls),
        #"energy_watt_sec_nvml": energy_watt_sec_nvml,
        #"energy_wh_codecarbon": codecarbon_energy_wh,
        "num_repeats": num_repeats,
        #"flops": flops,
        "energy_wh_codecarbon": np.sum(total_energy),
        "total_gpu_energy_kWh": np.sum(total_gpu_energy),
        "total_cpu_energy_kWh": np.sum(total_cpu_energy),
        "total_ram_energy_kWh": np.sum(total_ram_energy),
        ######## Variance
        "energy_wh_std_codecarbon": np.std(total_gpu_energy),
        "power_watt_std": np.std(power_draws),
        "time_sec_std": np.std(elapsed_times),
    }
    return results
# ──────────────────────────────────────────────────────────────────────────────
# TRAIN STEP
# ──────────────────────────────────────────────────────────────────────────────
def hf_train_step(model, optimizer, input_ids, attention_mask, labels):
    optimizer.zero_grad()
    with autocast():
        out  = model(input_ids=input_ids,
                     attention_mask=attention_mask,
                     labels=labels)
        loss = out.loss
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

# ──────────────────────────────────────────────────────────────────────────────
# BENCHMARK FUNCTION FOR BERT
# ──────────────────────────────────────────────────────────────────────────────
def bert_from_config(L, E, H, vocab_size, batch_size, seq_length,
                     repeats, tracker, lr=1e-3):
    # Cast to native int
    L, E, H = int(L), int(E), int(H)
    bs, sl  = int(batch_size), int(seq_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build config & model
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=E,
        num_hidden_layers=L,
        num_attention_heads=H,
        max_position_embeddings=sl,
        type_vocab_size=1
    )
    model     = BertForMaskedLM(cfg).to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Dummy data
    input_ids = torch.randint(0, vocab_size, (bs, sl), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()

    # Warm-up
    hf_train_step(model, optimizer, input_ids, attention_mask, labels)

    # Measure training
    name = f"L{L}_E{E}_H{H}_bs{bs}_sl{sl}"
    stats = measure_operation(name,
                              lambda: hf_train_step(model, optimizer,
                                                    input_ids, attention_mask, labels),
                              repeats, tracker)

    # FLOPs & params
    flops2, _  = profile(model, inputs=(input_ids,))
    num_params = sum(p.numel() for p in model.parameters())

    # Annotate
    stats.update({
        "num_layers":      L,
        "d_model":         E,
        "num_heads":       H,
        "batch_size":      bs,
        "seq_length":      sl,
        "params":  num_params,
        "flops2":          flops2,
    })

    # Cleanup
    del model, optimizer, tokenizer, input_ids, attention_mask, labels
    torch.cuda.empty_cache()
    gc.collect()

    return stats

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    today = date.today().strftime("%d%m%Y")
    hour  = datetime.now().hour
    out_csv   = f"bert_customcfg_{today}_{hour}H.csv"
    emiss_csv = f"bert_customcfg_emiss_{today}_{hour}H.csv"
    tracker   = EmissionsTracker(output_file=emiss_csv, log_level="ERROR")

    vocab_size      = 30522    # BERT base vocab size
    repeats         = 3
    seq_length      = 256
    batch_size_list = list(np.arange(32, 32*11, 32))
    layers          = [2, 4, 6, 12]
    d_models        = list(np.arange(64, 64*11, 64))  # multiples of 64
    heads           = [2, 4, 8]

    results = []
    for bs in batch_size_list:
        for L in layers:
            for E in d_models:
                for H in heads:
                    if E % H != 0:
                        continue
                    cfg_str = f"bs={bs},L={L},E={E},H={H}"
                    try:
                        print("Running", cfg_str)
                        stats = bert_from_config(
                            L, E, H, vocab_size,
                            bs, seq_length,
                            repeats, tracker
                        )
                        results.append(stats)
                        pd.DataFrame(results).to_csv(out_csv, index=False)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print("  ✘ OOM on", cfg_str, "— skipping")
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        else:
                            raise

    print("Done. Results saved to", out_csv)

if __name__ == "__main__":
    main()
