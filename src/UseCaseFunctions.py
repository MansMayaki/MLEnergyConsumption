import numpy as np
from typing import Tuple, Dict


def calculate_flops(batch_size: int, sequence_length: int, heads: int, d_model: int) -> Dict:
    """
    Calculate the number of FLOPs for key operations in a Transformer layer.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch (B).
    sequence_length : int
        Length of the input sequence (T).
    heads : int
        Number of attention heads (h).
    d_model : int
        Model hidden dimension (d_model).

    Returns
    -------
    dict
        Dictionary with FLOPs for QKV projections, attention scores, attention output,
        and the final projection.
    """
    d_k = d_model // heads  # Dimension per head

    # FLOPs for Q, K, V projections (three matrices)
    qkv_projections_flops = 2 * sequence_length * batch_size * (3 * d_model * d_model)

    # FLOPs for the final linear projection
    final_projection_flops = 2 * sequence_length * batch_size * (d_model * d_model)

    # Attention scores (Q @ K^T)
    attention_scores_flops = heads * batch_size * (2 * sequence_length**2 * d_k)

    # Attention output (A @ V)
    attention_output_flops = heads * batch_size * (2 * sequence_length**2 * d_k)

    return {
        "t_qkv_projections_flops": qkv_projections_flops,
        "t_score_flops": attention_scores_flops,
        "t_output_flops": attention_output_flops,
        "t_final_projection_flops": final_projection_flops,
    }


def calculate_duration(flops: dict, v_max: float, layers: int) -> Tuple[Dict, Dict]:
    """
    Calculate operation durations based on FLOPs and peak GPU throughput
    using hardware efficiency parameters (A100 80GB PCIe).

    Parameters
    ----------
    flops : dict
        FLOPs per operation (from calculate_flops).
    v_max : float
        Peak throughput of the GPU in FLOPs/s.
    layers : int
        Number of Transformer layers.

    Returns
    -------
    durations : dict
        Duration of each operation in microseconds.
    etas : dict
        Hardware efficiency factors for each operation.
    """
    efficiency_params = {
        "t_qkv_projections": {"eta_max": 69.38, "k": 10.37, "alpha": 0.78},
        "t_final_projection": {"eta_max": 70.43, "k": 6.24, "alpha": 0.77},
        "t_score": {"eta_max": 56.47, "k": 8.09, "alpha": 0.80},
        "t_output": {"eta_max": 66.83, "k": 8.65, "alpha": 0.80},
    }

    durations = {}
    etas = {}
    for op_name, op_flops in flops.items():
        key = op_name.replace("_flops", "")
        params = efficiency_params[key]

        # Efficiency (Equation 4)
        eta = params["eta_max"] * (1 - np.exp(-params["k"] * (op_flops * 1e-12) ** params["alpha"]))
        etas[f"{key}_hef"] = eta

        # Duration (Equation 3) -> convert to microseconds
        duration_s = layers * op_flops / (v_max * eta)
        durations[key] = duration_s * 1e6  # μs

    return durations, etas


def estimate_energy_consumption(durations: dict) -> float:
    """
    Estimate energy consumption using Ridge regression coefficients (Spec B).

    Parameters
    ----------
    durations : dict
        Operation durations in microseconds.

    Returns
    -------
    float
        Estimated energy consumption (arbitrary units).
    """
    # Ridge regression coefficients (Spec B results)
    h_intercept = 3.6292
    h_qkv = -0.1378
    h_score = 0.3041
    h_final = 0.5641
    h_output = 0.3041

    estimated_energy = (
        h_intercept
        + h_qkv * durations["t_qkv_projections"]
        + h_score * durations["t_score"]
        + h_final * durations["t_final_projection"]
        + h_output * durations["t_output"]
    )
    return estimated_energy


def compute_total_energy(
    batch_size: int,
    sequence_length: int,
    layers: int,
    heads: int,
    d_model: int,
    v_max: float = 156e12,
) -> float:
    """
    Compute the total estimated energy consumption for a Transformer model.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch (B).
    sequence_length : int
        Length of the input sequence (T).
    layers : int
        Number of Transformer layers (L).
    heads : int
        Number of attention heads (h).
    d_model : int
        Model hidden dimension (d_model).
    v_max : float, optional
        Peak GPU throughput in FLOPs/s (default: 156e12 for A100 PCIe 80GB).

    Returns
    -------
    float
        Estimated energy consumption.
    """
    flops = calculate_flops(batch_size, sequence_length, heads, d_model)
    durations, etas = calculate_duration(flops, v_max, layers)
    energy = estimate_energy_consumption(durations)
    return energy
