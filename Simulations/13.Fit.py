import pandas as pd

# Load full-range corrected model results
df = pd.read_csv('/mnt/data/uqgpf_model_corrected_fullrange.csv')

# Inspect first few rows to verify contents
df.head()