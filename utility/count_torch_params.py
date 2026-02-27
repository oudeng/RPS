#!/usr/bin/env python3
"""Count trainable torch parameters for all 18 method families.

Outputs: outputs/param_counts/torch_param_counts.csv
Columns: method, torch_params (None for non-neural methods)
"""
import os
import sys
import importlib
import pandas as pd

# Ensure code_RPS is on sys.path so AI_RPS can be found
_code_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

# --- agent registry (mirrored from RPS_main.py) ---
AGENT_MODULES = {
    "LSTM_v1": "AI_RPS.LSTM_v1", "LSTM_v2": "AI_RPS.LSTM_v2",
    "A3C_v1": "AI_RPS.A3C_v1", "A3C_v2": "AI_RPS.A3C_v2",
    "MSA_v2": "AI_RPS.MSA_v2",
    "Tr_v1": "AI_RPS.Tr_v1", "Tr_v2": "AI_RPS.Tr_v2",
    "RNN_v2": "AI_RPS.RNN_v2",
    "B_v1": "AI_RPS.B_v1", "B_v2": "AI_RPS.B_v2",
    "M_v1": "AI_RPS.M_v1", "M_v2": "AI_RPS.M_v2",
    "R": "AI_RPS.R", "WL": "AI_RPS.WL", "CG": "AI_RPS.CG",
    "RF": "AI_RPS.RF", "SVM": "AI_RPS.SVM", "XGB": "AI_RPS.XGB",
}

# Methods that have torch nn.Module components
NEURAL_METHODS = {"RNN_v2", "LSTM_v1", "LSTM_v2", "Tr_v1", "Tr_v2",
                  "MSA_v2", "A3C_v1", "A3C_v2"}

# The 18 method families for the paper
METHODS_18 = [
    "R", "CG", "WL", "B_v1", "B_v2", "M_v1", "M_v2",
    "SVM", "RF", "XGB",
    "RNN_v2", "LSTM_v1", "LSTM_v2", "Tr_v1", "Tr_v2", "MSA_v2",
    "A3C_v1", "A3C_v2",
]


def _import_agent(method_name):
    """Import agent class (returns cls with cls(...) -> agent instance)."""
    mod = importlib.import_module(AGENT_MODULES[method_name])
    return mod.Train


def _create_agent(cls, idxname):
    """Try multiple kwarg combinations to instantiate agent."""
    for kwargs in [
        {"idxname": idxname},
        {"name": idxname},
        {},
    ]:
        try:
            return cls(**kwargs)
        except TypeError:
            continue
    raise RuntimeError(f"Cannot instantiate {cls} with any kwargs combination")


def count_params(agent):
    """Count all trainable torch parameters in an agent object."""
    import torch.nn as nn
    seen_ids = set()
    total = 0

    # Walk all attributes looking for nn.Module
    for attr_name in dir(agent):
        try:
            v = getattr(agent, attr_name)
        except Exception:
            continue
        if isinstance(v, nn.Module):
            for p in v.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()

    # Also check __dict__ directly (some agents store models there)
    for v in agent.__dict__.values():
        if isinstance(v, nn.Module):
            for p in v.parameters():
                if p.requires_grad and id(p) not in seen_ids:
                    seen_ids.add(id(p))
                    total += p.numel()

    return total


def main():
    rows = []
    for m in METHODS_18:
        if m not in NEURAL_METHODS:
            rows.append({"method": m, "torch_params": None})
            print(f"[SKIP] {m:10s}  (non-neural)")
            continue

        try:
            cls = _import_agent(m)
            agent = _create_agent(cls, idxname=f"tmp_{m}")
            n = count_params(agent)
            rows.append({"method": m, "torch_params": int(n)})
            print(f"[OK]   {m:10s}  params = {n:,}")
        except Exception as e:
            rows.append({"method": m, "torch_params": "ERROR"})
            print(f"[ERR]  {m:10s}  {e}")

    df = pd.DataFrame(rows)
    out_dir = "outputs/param_counts"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/torch_param_counts.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Done] Wrote {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
