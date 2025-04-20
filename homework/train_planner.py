import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb

from models import MLPPlanner
from metrics import compute_errors
from datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "mlp_planner",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set random seed for reproducibility
    torch.manual_seed(seed)

    # Set up logging directory
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Initialize model
    model = MLPPlanner(**kwargs).to(device)
    model.train()

    # Load data
    train_data = load_data("../drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("../drive_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # Training loop
    for epoch in range(num_epoch):
        # Metrics to accumulate errors
        train_longitudinal_errors = []
        train_lateral_errors = []
        val_longitudinal_errors = []
        val_lateral_errors = []

        model.train()
        for batch in train_data:
            track_left = batch["track_left"].to(device, dtype=torch.float)
            track_right = batch["track_right"].to(device, dtype=torch.float)
            waypoints = batch["waypoints"].to(device, dtype=torch.float)
            mask = batch["mask"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            pred_waypoints = model(track_left, track_right)
            loss = loss_func(pred_waypoints, waypoints)
            loss.backward()
            optimizer.step()

            longitudinal_error, lateral_error = compute_errors(pred_waypoints, waypoints)
            train_longitudinal_errors.append(longitudinal_error)
            train_lateral_errors.append(lateral_error)

        model.eval()
        with torch.no_grad():
            for batch in val_data:
                track_left = batch["track_left"].to(device, dtype=torch.float)
                track_right = batch["track_right"].to(device, dtype=torch.float)
                waypoints = batch["waypoints"].to(device, dtype=torch.float)
                mask = batch["mask"].to(device, dtype=torch.float)

                pred_waypoints = model(track_left, track_right)
                longitudinal_error, lateral_error = compute_errors(pred_waypoints, waypoints)
                val_longitudinal_errors.append(longitudinal_error)
                val_lateral_errors.append(lateral_error)

        # Convert lists to tensors and compute means
        epoch_train_longitudinal_error = torch.tensor(train_longitudinal_errors).mean()
        epoch_train_lateral_error = torch.tensor(train_lateral_errors).mean()
        epoch_val_longitudinal_error = torch.tensor(val_longitudinal_errors).mean()
        epoch_val_lateral_error = torch.tensor(val_lateral_errors).mean()

        logger.add_scalar("train/longitudinal_error", epoch_train_longitudinal_error, global_step)
        logger.add_scalar("train/lateral_error", epoch_train_lateral_error, global_step)
        logger.add_scalar("val/longitudinal_error", epoch_val_longitudinal_error, global_step)
        logger.add_scalar("val/lateral_error", epoch_val_lateral_error, global_step)

        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_longitudinal_error={epoch_train_longitudinal_error:.4f} "
                f"train_lateral_error={epoch_train_lateral_error:.4f} "
                f"val_longitudinal_error={epoch_val_longitudinal_error:.4f} "
                f"val_lateral_error={epoch_val_lateral_error:.4f}"
            )

        global_step += 1

    # Save model
    model_path = log_dir / f"{model_name}.th"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    train(**vars(parser.parse_args()))

