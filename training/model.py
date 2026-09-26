"""ST-GCN for exercise recognition, adapted from a pre-trained checkpoint.

The architecture and parameter names match the published ST-GCN configuration
(ten blocks, 64 channels widening to 128 at block five and 256 at block eight,
with temporal downsampling at the same blocks), so its state_dict loads without
installing MMAction2. Only the classification head is new: the 60-class head is
replaced by a three-class one for squat, lunge and other. freeze_stages keeps
the early blocks fixed during adaptation.
"""

import torch
import torch.nn as nn


class UnitGCN(nn.Module):
    """Graph convolution over the skeleton, one weight set per partition."""

    def __init__(self, in_channels, out_channels, adjacency, adaptive="importance"):
        super().__init__()
        self.num_subsets = adjacency.shape[0]
        self.adaptive = adaptive

        self.register_buffer("A", adjacency.clone())

        # The published model learns a per-edge importance weight multiplying
        # the fixed adjacency, so some connections matter more than others.
        # It is a trained parameter and must be loaded with the rest.
        if adaptive == "importance":
            self.PA = nn.Parameter(adjacency.clone())
            nn.init.constant_(self.PA, 1)

        self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        n, _, t, v = x.shape
        a = self.A * self.PA if self.adaptive == "importance" else self.A

        x = self.conv(x)
        x = x.view(n, self.num_subsets, -1, t, v)
        x = torch.einsum("nkctv,kvw->nctw", x, a).contiguous()
        return self.act(self.bn(x))


class UnitTCN(nn.Module):
    """Temporal convolution along the frame axis."""

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              stride=(stride, 1), padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1,
                 residual=True):
        super().__init__()
        self.gcn = UnitGCN(in_channels, out_channels, adjacency)
        self.tcn = UnitTCN(out_channels, out_channels, 9, stride)
        self.relu = nn.ReLU()

        if not residual:
            self.residual = None
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = UnitTCN(in_channels, out_channels, 1, stride)

    def forward(self, x):
        shortcut = 0 if self.residual is None else self.residual(x)
        return self.relu(self.tcn(self.gcn(x)) + shortcut)


class ExerciseRecogniser(nn.Module):
    """Classify a window of skeleton frames as squat, lunge or other."""

    def __init__(self, adjacency, num_classes=3, in_channels=3,
                 base_channels=64, num_stages=10, inflate_stages=(5, 8),
                 down_stages=(5, 8), dropout=0.5):
        super().__init__()

        a = torch.as_tensor(adjacency, dtype=torch.float32)
        joints = a.shape[1]

        self.data_bn = nn.BatchNorm1d(in_channels * joints)

        blocks = [STGCNBlock(in_channels, base_channels, a, 1, residual=False)]
        channels = base_channels
        for stage in range(2, num_stages + 1):
            out_channels = channels * (2 if stage in inflate_stages else 1)
            stride = 2 if stage in down_stages else 1
            blocks.append(STGCNBlock(channels, out_channels, a, stride))
            channels = out_channels
        self.gcn = nn.ModuleList(blocks)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels, num_classes)
        self.out_channels = channels

    def forward(self, x):
        # x arrives as (N, T, V, C) from the dataset. The reference model
        # normalises per joint and channel across time, then convolves in a
        # (N, C, T, V) layout.
        n, t, v, c = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block in self.gcn:
            x = block(x)

        # Average over the remaining frames and every joint, so the decision
        # rests on the whole window rather than one frame or one limb.
        x = x.mean(dim=(2, 3))
        return self.classifier(self.dropout(x))

    def freeze_stages(self, n):
        """Hold the input normalisation and the first n blocks fixed.

        Their batch-norm layers are also kept in evaluation mode, otherwise
        their running statistics would drift toward the small training set even
        with the weights frozen.
        """
        for param in self.data_bn.parameters():
            param.requires_grad = False
        self.data_bn.eval()

        for block in self.gcn[:n]:
            for param in block.parameters():
                param.requires_grad = False
            block.eval()

    def train(self, mode=True):
        """Keep frozen batch-norm layers in evaluation mode across train()."""
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    if not any(p.requires_grad for p in module.parameters()):
                        module.eval()
        return self


def _strip(key):
    """Reduce a checkpoint key to the name this model would use."""
    for prefix in ("module.", "backbone.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_pretrained(model, path, verbose=True):
    """Load a published checkpoint into this model and print what loaded.

    A partial load would train mostly from random initialisation without any
    error, so every tensor is reported as loaded, skipped or missing.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)

    own = model.state_dict()
    loaded, shape_mismatch, not_in_model = [], [], []

    for key, value in state.items():
        name = _strip(key)
        if name not in own:
            not_in_model.append(key)
        elif own[name].shape != value.shape:
            shape_mismatch.append((name, tuple(value.shape), tuple(own[name].shape)))
        else:
            own[name] = value
            loaded.append(name)

    model.load_state_dict(own)

    expected_new = {"classifier.weight", "classifier.bias"}
    missing = [k for k in own
               if k not in loaded and k not in expected_new
               and not k.endswith("num_batches_tracked")]

    if verbose:
        print(f"checkpoint {path}")
        print(f"  loaded          {len(loaded)} tensors")
        print(f"  new head        {sorted(expected_new)}")
        if shape_mismatch:
            print(f"  shape mismatch  {len(shape_mismatch)}")
            for name, got, want in shape_mismatch[:8]:
                print(f"      {name}: checkpoint {got} vs model {want}")
        if not_in_model:
            print(f"  ignored         {len(not_in_model)} "
                  f"(e.g. {not_in_model[:3]})")
        if missing:
            print(f"  NOT LOADED      {len(missing)} tensors left at their "
                  f"initial values:")
            for name in missing[:12]:
                print(f"      {name}")
            print("  A large count here means the architecture does not match "
                  "the checkpoint.")
        else:
            print("  every parameter below the head came from the checkpoint")

    return {"loaded": loaded, "missing": missing,
            "shape_mismatch": shape_mismatch, "ignored": not_in_model}


def inspect_checkpoint(path, limit=40):
    """Print a checkpoint's contents, for confirming its architecture."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    print(f"{len(state)} tensors in {path}")
    total = 0
    for i, (key, value) in enumerate(state.items()):
        total += value.numel() if hasattr(value, "numel") else 0
        if i < limit:
            shape = tuple(value.shape) if hasattr(value, "shape") else type(value)
            print(f"  {key:<55} {shape}")
    if len(state) > limit:
        print(f"  ... {len(state) - limit} more")
    print(f"total parameters: {total:,}")


def count_parameters(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)
