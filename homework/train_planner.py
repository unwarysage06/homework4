
import argparse
from datetime import datetime
from pathlib import Path
from datetime import datetime


import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import  MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from .datasets.road_dataset import load_data


def train(
    model_name: str = "mlp_planner",
    transform_pipeline="state_only",
    num_workers: int = 2,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    exp_dir: str = "logs",
    **kwargs,
):
    # Setup
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Load model
    model = load_model(model_name, **kwargs).to(device)
    model.train()

    # Load datasets
    train_loader = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = load_data("drive_data/val", transform_pipeline=transform_pipeline, shuffle=False, batch_size=batch_size, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    global_step = 0

    for epoch in range(num_epoch):
        train_losses = []
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()

            if model_name == "cnn_planner":
                images = batch["image"].to(device)
                targets = batch["waypoints"].to(device)
                preds = model(image=images)
            else:  # MLP or Transformer
                track_left = batch["track_left"].to(device)
                track_right = batch["track_right"].to(device)
                targets = batch["waypoints"].to(device)
                preds = model(track_left=track_left, track_right=track_right)

            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            global_step += 1

        avg_train_loss = sum(train_losses) / len(train_losses)

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if model_name == "cnn_planner":
                    images = batch["image"].to(device)
                    targets = batch["waypoints"].to(device)
                    preds = model(image=images)
                else:
                    track_left = batch["track_left"].to(device)
                    track_right = batch["track_right"].to(device)
                    targets = batch["waypoints"].to(device)
                    preds = model(track_left=track_left, track_right=track_right)

                val_loss = loss_fn(preds, targets)
                val_losses.append(val_loss.item())

        avg_val_loss = sum(val_losses) / len(val_losses)
        logger.add_scalar("train_loss", avg_train_loss, global_step)
        logger.add_scalar("val_loss", avg_val_loss, global_step)

        print(
            f"Epoch {epoch+1:02d}/{num_epoch} "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # Save models
    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--transformer_pipeline", type=str, default="state_only")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--exp_dir", type=str, default="logs")

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
