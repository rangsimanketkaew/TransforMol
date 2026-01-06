"""
Usage: python r2s2_predict.py \
    --model r2s2_model_epoch1.pth \
    --test-set test_loader.pth \
    --plot-attention \
    --device cpu

Updates:
    18.10.2025 Initial script [Rangsiman Ketkaew]
"""

import argparse
import torch as t
import matplotlib.pyplot as plt

from r2s2_train import R2S2GATModel, predict
from r2s2_dataset import SolvationDataset, SolvationDatasetFromXYZ, collate_fn

# fmt: off
parser = argparse.ArgumentParser(description="R2S2 GAT Model Prediction")
parser.add_argument("--model", type=str, required=True, help="Path to the trained model")
parser.add_argument("--test-set", type=str, required=True, help="Path to the test data loader")
parser.add_argument("--plot-attention", action="store_true", help="Show attention weight heatmap")
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to run the model on")
args = parser.parse_args()
# fmt: on

device = args.device
test_data = t.load(args.test_set, weights_only=False, map_location=device)
print("Size of test data:", len(test_data.dataset))
model = R2S2GATModel().to(device)
model.load_state_dict(t.load(args.model, map_location=device))
model.eval()

with t.no_grad():
    for solutes, solvents, _ in test_data:
        for s in solutes:
            s = s.to(device)
        for s in solvents:
            s = s.to(device)

        preds, attn_weight = model(solutes, solvents)
        print("Predictions from model:", preds)

if args.plot_attention:
    print("Attention weight shape:", len(attn_weight))
    for i in range(len(attn_weight)):
        plt.imshow(attn_weight[i], aspect="auto", cmap="viridis")
        plt.title("Solute-Solvent Cross-Attention Heatmap")
        plt.xlabel("Solute Nodes")
        plt.ylabel("Solvent Nodes")
        plt.colorbar(label="Attention")
        plt.show()

# test_mae, test_rmse, preds, ys = predict(model, test_data, device=device)
# print("Predictions:", preds)
# print("True values:", ys)
# print(f"Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")
