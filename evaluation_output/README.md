# Evaluation Outputs

This folder contains evaluation results generated from training logs and parsed CSV files.
They provide both **visualizations** (accuracy curves) and **summary tables** (metrics grouped by model or channel).

## Files

### Plots

- `acc_curves_splits_I.png`
  Test accuracy curves for all combinations under **Scenario I**, grouped by split. Helps visualize how models converge over epochs for different seen/unseen configurations.
- `acc_curves_splits_II.png`
  Same as above, but for **Scenario II**.

### Tables

- `summary_table_by_models.csv`Summary of six metrics (Average/Min/Variance for Train & Test Accuracy)compared across **Models (CNN vs ResNet)** under different configurations:
  - `CNN(S)`, `ResNet(S)` → Seen splits
  - `CNN(U)`, `ResNet(U)` → Unseen splits
  - `CNN(I)`, `ResNet(I)` → Scenario I
  - `CNN(II)`, `ResNet(II)` → Scenario II
- `summary_table_by_channels.csv`Same metrics as above, but compared across **Channels (RSSI vs BOTH)**:

  - `RSSI(S)`, `BOTH(S)` → Seen splits
  - `RSSI(U)`, `BOTH(U)` → Unseen splits
  - `RSSI(I)`, `BOTH(I)` → Scenario I
  - `RSSI(II)`, `BOTH(II)` → Scenario II

## Metrics

Each table includes the following rows:

- Average Train Accuracy
- Average Test Accuracy
- Min Train Accuracy
- Min Test Accuracy
- Variance in Train Accuracy
- Variance in Test Accuracy
