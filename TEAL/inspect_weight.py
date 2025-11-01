import os
from utils.utils import get_tokenizer, get_sparse_model

CHECKPOINT = "wanda0.3"
MODEL_PATH = f"$SPDP_HOME/wanda/out/llama2_7b/unstructured/{CHECKPOINT}"
OUTPUT_PATH= "$SPDP_HOME/TEAL/models/Llama-2-7B/histogram"

tokenizer = get_tokenizer(MODEL_PATH)
model = get_sparse_model(MODEL_PATH, device='auto', histogram_path=os.path.join(OUTPUT_PATH, "histograms"), grab_acts=True, eval_sparsity=False)

print("Done")