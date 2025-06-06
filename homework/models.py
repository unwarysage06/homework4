from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        input_dim = n_track * 2 * 2
        hidden_dim = 128
        output_dim = n_waypoints * 2

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        
        x = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        x = x.view(x.size(0), -1)  # flatten to (b, 2*n_track*2)

        out = self.mlp(x)  # (b, n_waypoints * 2)
        out = out.view(-1, self.n_waypoints, 2)  # reshape to (b, n_waypoints, 2)
        return out


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Project 2D inputs to d_model
        self.input_proj = nn.Linear(2, d_model)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 2 * n_track, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Query embedding (already in your class)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Decoder attention layer
        self.decoder = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, batch_first=True)

        # Project decoder output to 2D waypoints
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.size(0)

        # Concatenate track boundaries: (b, 2 * n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)

        # Project to d_model: (b, 2 * n_track, d_model)
        x = self.input_proj(x)

        # Add positional encoding: (b, 2 * n_track, d_model)
        x = x + self.pos_embed

        # Transformer encoder
        memory = self.encoder(x)  # (b, 2 * n_track, d_model)

        # Query embedding: (b, n_waypoints, d_model)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)

        # Decoder attention: (b, n_waypoints, d_model)
        decoded, _ = self.decoder(query=queries, key=memory, value=memory)

        # Project to 2D coordinates: (b, n_waypoints, 2)
        out = self.output_proj(decoded)

        return out

class CNNPlanner(nn.Module):
    class Block(nn.Module):
        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            kernel_size = 3
            padding = (kernel_size - 1) // 2

            self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
            self.n1 = nn.GroupNorm(1, out_channels)
            self.relu1 = nn.ReLU()

            self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
            self.n2 = nn.GroupNorm(1, out_channels)
            self.relu2 = nn.ReLU()

            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()

        def forward(self, x0):
            x = self.relu1(self.n1(self.c1(x0)))
            x = self.relu2(self.n2(self.c2(x)))
            return x + self.skip(x0)

    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(INPUT_STD).view(1, 3, 1, 1), persistent=False)

        channel_output = 64
        n_blocks = 2

        layers = [
            nn.Conv2d(3, channel_output, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
        ]

        c1 = channel_output
        for _ in range(n_blocks):
            c2 = c1 * 2
            layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean) / self.input_std
        x = self.backbone(x)
        x = self.head(x)
        return x.view(image.size(0), self.n_waypoints, 2)

class CNNPlanner(torch.nn.Module):
    class Block(nn.Module):
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()
                kernel_size = 3
                padding = (kernel_size - 1) // 2

                self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
                self.n1 = nn.GroupNorm(1, out_channels)
                self.relu1 = nn.ReLU()

                self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
                self.n2 = nn.GroupNorm(1, out_channels)
                self.relu2 = nn.ReLU()

                self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride) if in_channels != out_channels else nn.Identity()

            def forward(self, x0):
                x = self.relu1(self.n1(self.c1(x0)))
                x = self.relu2(self.n2(self.c2(x)))
                return x + self.skip(x0)

    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.tensor(INPUT_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_std", torch.tensor(INPUT_STD).view(1, 3, 1, 1), persistent=False)

        channel_output = 64
        n_blocks = 2

        layers = [
            nn.Conv2d(3, channel_output, kernel_size=11, stride=2, padding=5),
            nn.ReLU(),
        ]

        c1 = channel_output
        for _ in range(n_blocks):
            c2 = c1 * 2
            layers.append(self.Block(c1, c2, stride=2))
            c1 = c2

        self.backbone = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        x = (image - self.input_mean) / self.input_std
        x = self.backbone(x)
        x = self.head(x)
        return x.view(image.size(0), self.n_waypoints, 2)


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024



# class Classifier(nn.Module):
#     class Block(nn.Module):
#         def __init__(self, in_channels, out_channels,stride):
#             super().__init__()
#             kernel_size = 3
#             padding = (kernel_size - 1) // 2

#             self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
#             self.n1 = nn.GroupNorm(1, out_channels)
#             self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
#             self.n2 = nn.GroupNorm(1, out_channels)
#             self.relu1 = nn.ReLU()
#             self.relu2 = nn.ReLU()


#             self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0) if in_channels != out_channels else torch.nn.Identity()

#         def forward(self, x0):
#             x = self.relu1(self.n1(self.c1(x0)))
#             x = self.relu2(self.n2(self.c2(x)))
#             return self.skip(x0) + x
#     def __init__(
#         self,
#         channel_output: int = 64,
#         n_blocks: int = 2,
#     ):
#         """
#         A convolutional network for image classification.

#         Args:
#             in_channels: int, number of input channels
#             num_classes: int
#         """
#         super().__init__()

#         self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN))
#         self.register_buffer("input_std", torch.as_tensor(INPUT_STD))

#         # TODO: implement
#         layers = [  
#             torch.nn.Conv2d(3, channel_output, kernel_size=11, stride=2, padding=5),
#             torch.nn.ReLU(),
#             ]
        
#         c1 = channel_output
#         for i in range(n_blocks):
#             c2 = c1 * 2
#             layers.append(self.Block(c1, c2, stride=2))
#             c1 = c2
#         layers.append(torch.nn.Conv2d(c1, 6, kernel_size=1, stride=2, padding=0))
#         self.model = torch.nn.Sequential(*layers)


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: tensor (b, 3, h, w) image

#         Returns:
#             tensor (b, num_classes) logits
#         """
#         # optional: normalizes the input
#         #z = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

#         # TODO: replace with actual forward pass
#         logits = self.model(x).mean(dim=-1).mean(dim=-1)

#         return logits

#     def predict(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Used for inference, returns class labels
#         This is what the AccuracyMetric uses as input (this is what the grader will use!).
#         You should not have to modify this function.

#         Args:
#             x (torch.FloatTensor): image with shape (b, 3, h, w) and vals in [0, 1]

#         Returns:
#             pred (torch.LongTensor): class labels {0, 1, ..., 5} with shape (b, h, w)
#         """
#         return self(x).argmax(dim=1)