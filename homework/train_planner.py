
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import  MLPPlanner, TransformerPlanner, CNNPlanner, load_model, save_model
from .datasets.road_dataset import load_data
from .utils import load_train_data, load_val_data


def train(
    model_name: str = "linear_planner",
    transform_pipeline="state_only",
    num_workers: int = 2,
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    exp_dir: str = "logs",
    **kwargs,
):
    num_workers = num_workers
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)


    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", transform_pipeline=transform_pipeline, shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)


    # create loss function and optimizer
    # optimizer = ...
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key,value  in metrics:
            metrics[key].clear()

        model.train()
        for batch_idx, (img, label) in enumerate(train_data):  
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(img)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(pred, label)
            loss.backward()
            optimizer.step()
            global_step += 1

            # Compute and store training accuracy
            acc = (pred.argmax(dim=1) == label).float().mean()
            metrics["train_acc"].append(acc)
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                pred = model(img)
                acc = (pred.argmax(dim=1) == label).float().mean()  
                metrics["val_acc"].append(acc)
        # log average train and val accuracy to tensorboard
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()
        logger.add_scalar("train_accuracy", epoch_train_acc, global_step=global_step)
        logger.add_scalar("val_accuracy", epoch_val_acc, global_step=global_step)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
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
