from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import os
from pathlib import Path
from datetime import datetime
import yaml
import csv
import time

from torch_geometric.nn import SimpleConv

from Dataset.LoadDataset import load_dataset
from GNNModel.GNNModel import GNNModel
from Train.Train import train
from Tool.VisualizeData import plot_line, plot_scatter
from Tool.Normalizer import Normalizer
import torch.nn as nn


def Training(cfg, bs):
    start_time = time.time()
    config = cfg

    # Force random split mode (override config if needed)
    config["split_ratio_random"] = True

    base_dir = config["base_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"{timestamp}_arg{bs}"
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log = output_dir / "training.csv"

    # Save modified config to output directory
    copied_config_path = output_dir / 'Config.yaml'
    with open(copied_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    print(f"Config saved to: {copied_config_path}")

    # ========== Step1. Load dataset ==========
    print(f"Loading dataset from: {config['dataset_path']}")
    dataset, edge_in_dim_after = load_dataset(
        dataset_path=config["dataset_path"],
        Bessel_parameter=config["Bessel_parameter"],
        gaussian_parameter=config["gaussian_parameter"]
    )

    # Dataset split logic: fixed random split (100 train / 10 val / 10 test)
    if config.get("split_ratio_random", False):
        total_samples = len(dataset)
        # Validate dataset size is sufficient
        required_samples = 100 + 10 + 10
        assert total_samples >= required_samples, \
            f"Dataset size ({total_samples}) is smaller than required ({required_samples})"

        # Generate random permutation of indices
        all_indices = torch.randperm(total_samples)

        # Fixed split sizes
        train_size = 100
        val_size = 10
        test_size = 10

        # Split indices
        train_idx = all_indices[:train_size]
        val_idx = all_indices[train_size: train_size + val_size]
        test_idx = all_indices[train_size + val_size: train_size + val_size + test_size]

        # Create datasets from indices
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]

        # Print dataset sizes and detailed indices
        print(
            f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")
        print(f"Train indices (randomly selected): {train_idx.tolist()}")
        print(f"Validation indices (randomly selected): {val_idx.tolist()}")
        print(f"Test indices (randomly selected): {test_idx.tolist()}")
    else:
        # Keep original fixed index split as fallback (not used in this modification)
        train_dataset = dataset[:config["train_end"]]
        val_dataset = dataset[config["train_end"]:config["val_end"]]
        test_dataset = dataset[config["val_end"]:config["test_end"]]
        print(
            f"Train set size: {len(train_dataset)}, Val set size: {len(val_dataset)}, Test set size: {len(test_dataset)}")

    # Initialize and save normalizer
    normalizer = Normalizer()
    normalizer.fit(train_dataset.y)
    normalizer_path = output_dir / "normalizer_params.pth"
    torch.save(normalizer.state_dict(), normalizer_path)
    print(f"Normalizer parameters saved to: {normalizer_path}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Initialize GNN model
    model = GNNModel(
        node_in_dim=config["node_in_dim"],
        edge_in_dim=edge_in_dim_after,
        hidden_dim=config["hidden_dim"],
        node_emb_dim=config["node_emb_dim"],
        edge_emb_dim=config["edge_emb_dim"],
        n_rec=config["num_layers"],
        num_heads=config["num_heads"],
        decoder_hidden=config["decoder_hidden"],
        Dropout_encoder=config["dropout_encoder"],
        Dropout_decoder=config["dropout_decoder"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model initialized on device: {device}")

    # Training components
    criterion = nn.MSELoss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    # Early stopping configuration
    best_val_pcc = -1.0
    best_model_path = None
    patience = config["early_stop_patience"]
    patience_counter = 0

    # Training loop
    for epoch in range(1, config["total_epochs"] + 1):
        trained_model = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            normalizer=normalizer,
            loss_function=criterion,
            optimizer=optimizer,
            device=device
        )
        scheduler.step()

        val_loss = trained_model["Val Loss"]
        val_pcc = trained_model["Val Pearson"]

        # Log training metrics
        if epoch % config["print_every"] == 0 or epoch == 1:
            print(f"\nEpoch {epoch}: TrainLoss={trained_model['Train Loss']:.4f}, "
                  f"ValLoss={val_loss:.4f}, TrainPCC={trained_model['Train Pearson']:.4f}, "
                  f"ValPCC={val_pcc:.4f}, LR={optimizer.param_groups[0]['lr']:.6f}, "
                  f"BestValPCC={best_val_pcc:.4f}")

            with open(training_log, 'a', newline='') as f:
                writer = csv.writer(f)
                if f.tell() == 0:
                    writer.writerow(['Epoch', 'Train Loss', 'Train Pearson', 'Val Loss', 'Val Pearson'])
                writer.writerow([epoch,
                                 round(trained_model["Train Loss"], 4),
                                 round(trained_model['Train Pearson'], 4),
                                 round(val_loss, 4),
                                 round(val_pcc, 4)])

        # Early stopping logic
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            patience_counter = 0
            # Remove previous best model
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            # Save new best model
            best_model_path = output_dir / f"best_model_epoch{epoch}_valpcc{val_pcc:.4f}.pth"
            torch.save(model, best_model_path)
            print(f"Epoch {epoch}: New best ValPCC {val_pcc:.4f} (model saved)")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}: ValPCC no improvement ({patience_counter}/{patience})")

        # Trigger early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered: no ValPCC improvement for {patience} epochs")
            break

        # Generate visualization plots
        if epoch % config["process_loss_interval"] == 0 or epoch == config["total_epochs"]:
            plot_line(training_log, output_dir / "LossVsEpoch.png")
            plot_line(training_log, output_dir / "Score.png", y=['Train Pearson', 'Val Pearson'])
        if epoch % config["process_graph_interval"] == 0 or epoch == config["total_epochs"]:
            plot_scatter(trained_model['y_true_train'], trained_model['y_pred_train'],
                         output_dir / f"Train-{epoch:05d}-{trained_model['Train Pearson']:.4f}.png")
            plot_scatter(trained_model['y_true_val'], trained_model['y_pred_val'],
                         output_dir / f"Val-{epoch:05d}-{val_pcc:.4f}.png")

    # Training summary
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time / 60:.2f} minutes")
    print(f"Best Validation PCC = {best_val_pcc:.4f}")
    return best_val_pcc, output_dir


if __name__ == "__main__":
    with open("Config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    Training(cfg, bs="Test")