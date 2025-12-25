
import numpy as np
import pandas as pd
import isda

# --- Utilities ---
def _truth(name, intrinsic_dim_expected, blocks_expected, notes=""):
    return {
        "name": name,
        "intrinsic_dim_expected": int(intrinsic_dim_expected),
        "blocks_expected": blocks_expected,
        "notes": notes,
    }

def _mk_block_names(start, size):
    return [f"f{i}" for i in range(start, start + size)]

def _repeat_with_small_noise(base, rng, noise):
    return base + noise * rng.normal(size=base.shape[0])

def _mop_df(Y):
    return pd.DataFrame(Y, columns=[f"f{i+1}" for i in range(Y.shape[1])])

def evaluate_reduced_model_fidelity(results_dict):
    results_summary = []
    for name, data in results_dict.items():
        result_obj = data.get("result_obj")
        truth = data.get("truth", {})
        
        Y = result_obj.Y
        mis = result_obj.best_mis
        mis_indices = mis["mis_indices"] if mis else []
        
        if not mis_indices:
            fidelity = 0.0
            ses = 0.0
        else:
            if result_obj.ses_results:
                fidelity = result_obj.ses_results["F_real"]
                ses = result_obj.ses_results.get("ses", 0.0)
            else:
                ses_out = isda.calculate_ses(Y, mis_indices, n_perm=1, return_details=True)
                fidelity = ses_out["F_real"]
                ses = ses_out.get("ses", 0.0)

        expected_dim = truth.get("intrinsic_dim_expected", None)
        mis_size = len(mis_indices)

        entry = {
            "Case": name,
            "Selected MIS Size": mis_size,
            "Reconstruction Fidelity (F_real)": fidelity,
            "SES (Structural Evidence Score)": ses
        }
        if expected_dim is not None:
            entry["Expected Intrinsic Dim"] = expected_dim

        results_summary.append(entry)

    df_summary = pd.DataFrame(results_summary)
    cols = ["Case", "Selected MIS Size", "Reconstruction Fidelity (F_real)", "SES (Structural Evidence Score)"]
    if "Expected Intrinsic Dim" in df_summary.columns:
        cols.insert(1, "Expected Intrinsic Dim")
        df_summary = df_summary[cols]

    return df_summary

# --- Data Generators Battery 1 ---
def make_case1_independence(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = rng.normal(size=(N, M))
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 1 - Total independence",
        intrinsic_dim_expected=M,
        blocks_expected=[[c] for c in cols],
        notes="Each objective is independent (Gaussian noise)."
    )
    return df, truth

def make_case2_total_redundancy(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(N, 1))
    noise = rng.normal(scale=0.05, size=(N, M))
    Y = latent + noise
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 2 - Total redundancy",
        intrinsic_dim_expected=1,
        blocks_expected=[cols],
        notes="A single latent; all objectives are noisy copies."
    )
    return df, truth

def make_case3_block_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 4))
    Y = np.zeros((N, M))
    for b in range(4):
        for j in range(5):
            idx = 5*b + j
            Y[:, idx] = latent_blocks[:, b] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 3 - Blocks (4 x 5)",
        intrinsic_dim_expected=4,
        blocks_expected=[],
        notes="4 independent latents; each generates 5 objectives."
    )
    return df, truth

def make_case4_two_big_blocks(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    latent_blocks = rng.normal(size=(N, 2))
    Y = np.zeros((N, M))
    for i in range(10):
        Y[:, i] = latent_blocks[:, 0] + rng.normal(scale=0.2, size=N)
    for i in range(10, 20):
        Y[:, i] = latent_blocks[:, 1] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 4 - Blocks (2 x 10)",
        intrinsic_dim_expected=2,
        blocks_expected=[],
        notes="2 independent latents; each generates 10 objectives."
    )
    return df, truth

def make_case5_chain_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    Y = np.zeros((N, M))
    Y[:, 0] = rng.normal(size=N)
    for j in range(1, M):
        Y[:, j] = Y[:, j-1] + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 5 - Chain",
        intrinsic_dim_expected=M,
        blocks_expected=[cols],
        notes="Sequential dependency."
    )
    return df, truth

def make_case6_mixed_structure(N=1000, M=20, seed=123):
    rng = np.random.default_rng(seed)
    assert M == 20
    Y = np.zeros((N, M))
    Y[:, :10] = rng.normal(size=(N, 10))
    latent1 = rng.normal(size=N)
    latent2 = rng.normal(size=N)
    for j in range(10, 15):
        Y[:, j] = latent1 + rng.normal(scale=0.2, size=N)
    for j in range(15, 20):
        Y[:, j] = latent2 + rng.normal(scale=0.2, size=N)
    cols = [f"f{i+1}" for i in range(M)]
    df = pd.DataFrame(Y, columns=cols)
    truth = _truth(
        name="Case 6 - Mixed (indep + latents)",
        intrinsic_dim_expected=12,
        blocks_expected=[],
        notes="f1..f10 independent; f11..f15 latent1; f16..f20 latent2."
    )
    return df, truth

def make_case7_pure_conflict_groups(N=1000, M=20, noise=0.05, seed=123, **kwargs):
    rng = np.random.default_rng(seed)
    if M < 2: raise ValueError("M must be >= 2")
    M_pos = (M + 1) // 2
    M_neg = M - M_pos
    x = rng.normal(size=N)
    Y_pos = np.column_stack([x + noise * rng.normal(size=N) for _ in range(M_pos)])
    Y_neg = np.column_stack([(-x) + noise * rng.normal(size=N) for _ in range(M_neg)])
    Y = np.column_stack([Y_pos, Y_neg])
    cols = [f"f{i+1}" for i in range(M)]
    Y = pd.DataFrame(Y, columns=cols)
    truth = {
        "name": f"Case 7 - Structural conflict (anti-corr) 2-groups",
        "intrinsic_dim_expected": 2,
        "blocks_expected": [cols[:M_pos], cols[M_pos:]],
        "notes": "Conflict groups (+x and -x).",
    }
    return Y, truth

# --- Execution ---
battery1 = [
    ("Case 1 - Total independence", make_case1_independence),
    ("Case 2 - Total redundancy", make_case2_total_redundancy),
    ("Case 3 - Blocks (4 x 5)", make_case3_block_structure),
    ("Case 4 - Blocks (2 x 10)", make_case4_two_big_blocks),
    ("Case 5 - Chain", make_case5_chain_structure),
    ("Case 6 - Mixed (indep + latents)", make_case6_mixed_structure),
    ("Case 7 - Structural conflict (anti-corr) with groups", make_case7_pure_conflict_groups),
]

caution = isda.CONSERVATIVE

def run_cases(cases_list, N=1000):
    all_results = {}
    for name, gen in cases_list:
        Y, truth = gen(N=N)
        # Suppress prints for clean output
        result = isda.analyze(Y, caution=caution, run_ses=True, name=name)
        all_results[name] = {
            "truth": truth,
            "result_obj": result,
        }
    return all_results

battery1_results = run_cases(battery1)
battery1_fidelity_df = evaluate_reduced_model_fidelity(battery1_results)
print(battery1_fidelity_df.to_string(index=False))
