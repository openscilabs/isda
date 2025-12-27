
import dtlz_lib
import misda
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

print("=== FULL SUMMARY for DTLZ5 ===")
Y5, _ = dtlz_lib.generate_dtlz5(N=300, M=3)
res5 = misda.analyze(Y5, caution=0.5, run_ses=True)
print(res5.summary())
