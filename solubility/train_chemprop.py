"""
train_chemprop.py
-----------------
Chemprop v2 D-MPNN for aqueous solubility (AqSolDB, scaffold split).

Provides `train_and_evaluate(config, datasets, seed)` — the single objective that
every HPO method (random / Bayesian / LLM-driven) will call. Data is loaded and
featurized ONCE; the model is rebuilt and trained per call.

Run directly (`python train_chemprop.py`) to:
  1. print the mean-predictor baseline
  2. run a 5-seed noise-floor experiment on the default config
"""

from dataclasses import dataclass
import numpy as np
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from chemprop import data, featurizers, models, nn

# RTX 3060 has Tensor Cores; this uses them and silences the Lightning warning.
torch.set_float32_matmul_precision("medium")


# ---------------------------------------------------------------------------
# 1. Hyperparameter config  (defaults == the model you built in the notebook)
# ---------------------------------------------------------------------------
@dataclass
class Config:
    depth: int = 3                 # message-passing steps  -> BondMessagePassing(depth=)
    message_hidden_dim: int = 300  # bond hidden width      -> BondMessagePassing(d_h=)
    ffn_hidden_dim: int = 300      # FFN hidden width       -> RegressionFFN(hidden_dim=)
    ffn_num_layers: int = 1        # FFN depth              -> RegressionFFN(n_layers=)
    dropout: float = 0.0           # applied in both mp and ffn
    aggregation: str = "mean"      # "mean" | "sum" | "norm"
    batch_size: int = 64
    max_lr: float = 1e-3           # peak learning rate     -> MPNN(max_lr=)
    max_epochs: int = 20           # ~enough for this model (val loss plateaus)


# aggregation name -> class (the readout stage; "mean" == your global_mean_pool)
AGG = {
    "mean": nn.MeanAggregation,
    "sum": nn.SumAggregation,
    "norm": nn.NormAggregation,
}


# ---------------------------------------------------------------------------
# 2. Load + featurize the data ONCE  (not per training run)
# ---------------------------------------------------------------------------
def load_split(seed: int = 42):
    """TDC AqSolDB scaffold split (deterministic for a given seed)."""
    from tdc.single_pred import ADME
    return ADME(name="Solubility_AqSolDB").get_split(method="scaffold", seed=seed)


def build_datasets(split):
    """
    SMILES + targets -> featurized MoleculeDatasets.
    Scaler is fit on TRAIN only, applied to VAL. TEST is left unscaled: the model's
    output_transform unscales its predictions, so TEST metrics come out in real logS.
    Consequence: VAL metrics are in SCALED units (fine for ranking configs in HPO),
    TEST metrics are in real logS (for final reporting).
    """
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def make_dp(df):
        smis = df["Drug"].values
        ys = df[["Y"]].values  # double brackets -> shape [N, 1]
        return [data.MoleculeDatapoint.from_smi(s, y) for s, y in zip(smis, ys)]

    train_dset = data.MoleculeDataset(make_dp(split["train"]), featurizer)
    scaler = train_dset.normalize_targets()          # fit + transform train

    val_dset = data.MoleculeDataset(make_dp(split["valid"]), featurizer)
    val_dset.normalize_targets(scaler)               # reuse train's scaler

    test_dset = data.MoleculeDataset(make_dp(split["test"]), featurizer)  # unscaled

    return train_dset, val_dset, test_dset, scaler


# ---------------------------------------------------------------------------
# 3. Build a model from a Config  (same architecture, driven by hyperparameters)
# ---------------------------------------------------------------------------
def build_model(config: Config, scaler) -> models.MPNN:
    # message passing (body, part 1)
    mp = nn.BondMessagePassing(
        d_h=config.message_hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
    )

    # aggregation / readout (body, part 2)
    agg = AGG[config.aggregation]()

    # prediction head (with unscaling baked into the output).
    # NOTE: input_dim MUST match the message-passing width (d_h), because the FFN
    # receives the aggregated d_h-wide molecule vector.
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        input_dim=config.message_hidden_dim,
        hidden_dim=config.ffn_hidden_dim,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
        output_transform=output_transform,
    )

    metrics = [nn.metrics.RMSE(), nn.metrics.MAE()]  # RMSE first -> drives monitoring

    # assemble. max_lr sets the peak LR of Chemprop's built-in Noam scheduler.
    # (If your Chemprop version rejects max_lr here, drop it — LR then uses the
    #  default; everything else still works.)
    return models.MPNN(mp, agg, ffn, batch_norm=True, metrics=metrics, max_lr=config.max_lr)


# ---------------------------------------------------------------------------
# 4. THE OBJECTIVE  (what HPO calls)
# ---------------------------------------------------------------------------
def train_and_evaluate(config: Config, datasets, seed: int = 0) -> dict:
    """
    Train one model with `config` + `seed`; return metrics.
    `datasets` = (train_dset, val_dset, test_dset, scaler) from build_datasets(),
    passed in so featurization isn't repeated every call.

    Returns:
      val_loss   : best validation loss (SCALED MSE) — use THIS to select configs in HPO
      val_rmse   : best validation RMSE (scaled units; None if the key isn't logged)
      test_rmse  : test RMSE in real logS — final reporting only, do NOT tune on this
      test_mae   : test MAE in real logS
    """
    pl.seed_everything(seed, workers=True)  # seeds python + numpy + torch (weight init, shuffling)
    train_dset, val_dset, test_dset, _ = datasets

    train_loader = data.build_dataloader(train_dset, batch_size=config.batch_size, num_workers=0)
    val_loader = data.build_dataloader(val_dset, batch_size=config.batch_size, num_workers=0, shuffle=False)
    test_loader = data.build_dataloader(test_dset, batch_size=config.batch_size, num_workers=0, shuffle=False)

    model = build_model(config, datasets[3])

    checkpointing = ModelCheckpoint(
        dirpath="checkpoints",
        monitor="val/rmse",
        mode="min",
        save_top_k=1,   # keep only the best epoch's weights
    )
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,   # off: too noisy across many HPO runs
        accelerator="auto",          # uses the GPU; "cpu" is a safe fallback
        devices=1,
        max_epochs=config.max_epochs,
        callbacks=[checkpointing],
    )

    trainer.fit(model, train_loader, val_loader)

    # evaluate the BEST checkpoint (lowest val_loss), not the final epoch
    val_res = trainer.validate(model, val_loader, ckpt_path="best",weights_only=False, verbose=False)[0]
    test_res = trainer.test(model, test_loader, ckpt_path="best",weights_only=False, verbose=False)[0]

    best_val_loss = checkpointing.best_model_score
    return {
        "val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "val_rmse": val_res.get("val/rmse"),   # scaled units
        "test_rmse": test_res.get("test/rmse"),  # real logS
        "test_mae": test_res.get("test/mae"),    # real logS
    }


# ---------------------------------------------------------------------------
# 5. Mean-predictor baseline  (the honest floor for every comparison)
# ---------------------------------------------------------------------------
def mean_predictor_baseline(split) -> float:
    """Predict the TRAIN mean for every TEST molecule; return test RMSE (logS).
    Under the scaffold shift this is ~2.44, NOT the train std of ~2.25."""
    train_mean = split["train"]["Y"].mean()
    y_true = split["test"]["Y"].values
    return float(np.sqrt(np.mean((y_true - train_mean) ** 2)))


# ---------------------------------------------------------------------------
# 6. Run directly: baseline + noise-floor experiment
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    split = load_split(seed=42)
    datasets = build_datasets(split)     # featurize ONCE
    config = Config()                    # defaults == your notebook model

    baseline = mean_predictor_baseline(split)
    print(f"\nMean-predictor baseline (test RMSE): {baseline:.3f} logS\n")

    # NOISE FLOOR: same config, 5 seeds. Establishes run-to-run variation, which
    # every HPO improvement must exceed to be real.
    test_rmses = []
    for seed in range(5):
        m = train_and_evaluate(config, datasets, seed=seed)
        test_rmses.append(m["test_rmse"])
        print(f"seed {seed}:  test RMSE {m['test_rmse']:.3f} | "
              f"test MAE {m['test_mae']:.3f} | val_loss {m['val_loss']:.3f}")

    arr = np.array(test_rmses, dtype=float)
    print(f"\nNoise floor: test RMSE {arr.mean():.3f} ± {arr.std():.3f} logS  (n=5)")
    print("=> HPO differences smaller than this std are noise, not signal.\n")


    