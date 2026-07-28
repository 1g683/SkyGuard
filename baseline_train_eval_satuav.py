import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    import timm
except ImportError as exc:
    raise ImportError("This script needs timm. Install it with: pip install timm") from exc

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
except ImportError as exc:
    raise ImportError("This script needs scikit-learn. Install it with: pip install scikit-learn") from exc


FSRA_MODEL = "vit_small_patch16_224"
FSRA_BLOCK = 1
FSRA_IMAGE_SIZE = 256
FSRA_LR = 0.01
FSRA_WEIGHT_DECAY = 5e-4
FSRA_SHARE_WEIGHT = True
FSRA_FEATURE_MODE = "raw"
FSRA_SALIENCY_MODE = "mean"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_satuav_csv(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            a_name = row["A"].strip()
            b_name = row["B"].strip()
            sample_id = Path(a_name).stem.split("_")[0]
            rows.append({"id": sample_id, "sat": a_name, "uav": b_name})
    return rows


def image_train_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def image_eval_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class SatUAVTrainDataset(Dataset):
    def __init__(self, dataset_dir, split_csv, image_dir, transform):
        self.dataset_dir = Path(dataset_dir)
        self.image_root = self.dataset_dir / image_dir
        self.rows = read_satuav_csv(self.dataset_dir / split_csv)
        self.transform = transform
        self.id_to_label = {row["id"]: idx for idx, row in enumerate(self.rows)}

    def __len__(self):
        return len(self.rows)

    def _load(self, name):
        image = Image.open(self.image_root / name).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx):
        row = self.rows[idx]
        sat = self._load(row["sat"])
        uav = self._load(row["uav"])
        label = self.id_to_label[row["id"]]
        return sat, uav, label


class SatUAVDetectionDataset(Dataset):
    def __init__(self, dataset_dir, split_csv, image_dir, transform, negative_shift=1):
        self.dataset_dir = Path(dataset_dir)
        self.image_root = self.dataset_dir / image_dir
        self.rows = read_satuav_csv(self.dataset_dir / split_csv)
        self.transform = transform
        self.pairs = []

        n = len(self.rows)
        for i, row in enumerate(self.rows):
            self.pairs.append((row["sat"], row["uav"], 0, row["id"], row["id"]))
            neg = self.rows[(i + negative_shift) % n]
            self.pairs.append((neg["sat"], row["uav"], 1, neg["id"], row["id"]))

    def __len__(self):
        return len(self.pairs)

    def _load(self, name):
        image = Image.open(self.image_root / name).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx):
        sat_name, uav_name, label, sat_id, uav_id = self.pairs[idx]
        sat = self._load(sat_name)
        uav = self._load(uav_name)
        return sat, uav, torch.tensor(label, dtype=torch.long), sat_id, uav_id


class TimmSharedInfoNCE(nn.Module):
    def __init__(self, model_name, image_size, pretrained):
        super().__init__()
        kwargs = {"pretrained": pretrained, "num_classes": 0}
        if "vit" in model_name:
            kwargs["img_size"] = image_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_sat(self, x):
        return self.backbone(x)

    def encode_uav(self, x):
        return self.backbone(x)

    def forward(self, sat, uav):
        return self.encode_sat(sat), self.encode_uav(uav)


class DualTimmClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained, share_weight=False):
        super().__init__()
        self.sat_backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.uav_backbone = self.sat_backbone if share_weight else timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        dim = self.sat_backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def encode_sat(self, x):
        return self.sat_backbone(x)

    def encode_uav(self, x):
        return self.uav_backbone(x)

    def forward(self, sat, uav):
        return self.classifier(self.encode_sat(sat)), self.classifier(self.encode_uav(uav))


class LPNBranch(nn.Module):
    def __init__(self, backbone_name, block, pretrained):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.block = block

    def forward_features_map(self, x):
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.act1(x) if hasattr(m, "act1") else m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        return x

    def get_part_pool(self, x):
        result = []
        pooling = nn.AdaptiveAvgPool2d((1, 1))
        h, w = x.size(2), x.size(3)
        c_h, c_w = int(h / 2), int(w / 2)
        per_h, per_w = math.floor(h / (2 * self.block)), math.floor(w / (2 * self.block))
        if per_h < 1 or per_w < 1:
            x = F.interpolate(x, size=(max(h, self.block * 2), max(w, self.block * 2)), mode="bilinear", align_corners=True)
            h, w = x.size(2), x.size(3)
            c_h, c_w = int(h / 2), int(w / 2)
            per_h, per_w = math.floor(h / (2 * self.block)), math.floor(w / (2 * self.block))

        for i in range(1, self.block + 1):
            if i < self.block:
                part = x[:, :, c_h - i * per_h : c_h + i * per_h, c_w - i * per_w : c_w + i * per_w]
            else:
                part = x
            result.append(pooling(part))
        return torch.cat(result, dim=2).view(x.size(0), x.size(1), self.block)

    def forward(self, x):
        return self.get_part_pool(self.forward_features_map(x))


class LPNClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes, block, pretrained, share_weight=False):
        super().__init__()
        self.block = block
        self.sat_branch = LPNBranch(backbone_name, block, pretrained)
        self.uav_branch = self.sat_branch if share_weight else LPNBranch(backbone_name, block, pretrained)
        dim = self.sat_branch.backbone.num_features
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, num_classes),
                )
                for _ in range(block)
            ]
        )

    def _classify_parts(self, x):
        return [self.classifiers[i](x[:, :, i]) for i in range(self.block)]

    def encode_sat(self, x):
        return self.sat_branch(x).flatten(1)

    def encode_uav(self, x):
        return self.uav_branch(x).flatten(1)

    def forward(self, sat, uav):
        return self._classify_parts(self.sat_branch(sat)), self._classify_parts(self.uav_branch(uav))


class BottleneckClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, drop_rate=0.5):
        super().__init__()
        self.add_block = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
        )
        self.classifier = nn.Linear(512, num_classes)

    def feature(self, x):
        return self.add_block(x)

    def forward(self, x):
        return self.classifier(self.feature(x))


class FSRABranch(nn.Module):
    def __init__(self, model_name, block, pretrained, image_size, saliency_mode="mean"):
        super().__init__()
        kwargs = {
            "pretrained": pretrained,
            "num_classes": 0,
            "img_size": image_size,
            "global_pool": "",
        }
        self.backbone = timm.create_model(model_name, **kwargs)
        self.block = block
        self.dim = self.backbone.num_features
        self.saliency_mode = saliency_mode

    def forward(self, x):
        tokens = self.backbone.forward_features(x)
        if isinstance(tokens, dict):
            tokens = tokens.get("x", tokens.get("last_hidden_state"))
        if tokens.dim() != 3:
            raise RuntimeError(
                "FSRA needs ViT patch tokens from timm.forward_features(). "
                "Use a ViT model such as vit_base_patch16_384."
            )

        cls_feature = tokens[:, 0, :]
        patch_features = tokens[:, 1:, :]
        if self.saliency_mode == "mean":
            saliency = torch.mean(patch_features, dim=-1)
        elif self.saliency_mode == "norm":
            saliency = patch_features.norm(dim=-1)
        else:
            raise ValueError(f"Unknown FSRA saliency mode: {self.saliency_mode}")
        order = torch.argsort(saliency, dim=1, descending=True)
        ordered = patch_features.gather(1, order.unsqueeze(-1).expand(-1, -1, patch_features.size(-1)))
        split_each = ordered.size(1) / self.block
        split_sizes = [int(split_each) for _ in range(self.block - 1)]
        split_sizes.append(ordered.size(1) - sum(split_sizes))
        chunks = torch.split(ordered, split_sizes, dim=1)
        region_features = torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=2)
        return cls_feature, region_features


class FSRAClassifier(nn.Module):
    def __init__(self, model_name, num_classes, block, pretrained, image_size, share_weight=True, feature_mode="bottleneck", saliency_mode="mean"):
        super().__init__()
        self.block = block
        self.feature_mode = feature_mode
        self.sat_branch = FSRABranch(model_name, block, pretrained, image_size, saliency_mode=saliency_mode)
        self.uav_branch = self.sat_branch if share_weight else FSRABranch(model_name, block, pretrained, image_size, saliency_mode=saliency_mode)
        dim = self.sat_branch.dim
        self.global_classifier = BottleneckClassifier(dim, num_classes, drop_rate=0.5)
        self.region_classifiers = nn.ModuleList(
            [BottleneckClassifier(dim, num_classes, drop_rate=0.5) for _ in range(block)]
        )

    def _classify(self, outputs):
        cls_feature, region_features = outputs
        logits = [self.global_classifier(cls_feature)]
        logits.extend(self.region_classifiers[i](region_features[:, :, i]) for i in range(self.block))
        return logits

    def _encode(self, outputs):
        cls_feature, region_features = outputs
        if self.feature_mode == "raw":
            return torch.cat([region_features, cls_feature.unsqueeze(2)], dim=2)
        if self.feature_mode != "bottleneck":
            raise ValueError(f"Unknown FSRA feature mode: {self.feature_mode}")
        encoded_regions = [
            self.region_classifiers[i].feature(region_features[:, :, i])
            for i in range(self.block)
        ]
        encoded_global = self.global_classifier.feature(cls_feature)
        return torch.stack(encoded_regions + [encoded_global], dim=2)

    def encode_sat(self, x):
        return self._encode(self.sat_branch(x))

    def encode_uav(self, x):
        return self._encode(self.uav_branch(x))

    def forward(self, sat, uav):
        return self._classify(self.sat_branch(sat)), self._classify(self.uav_branch(uav))


def info_nce_loss(sat_features, uav_features, logit_scale):
    sat_features = F.normalize(sat_features, dim=-1)
    uav_features = F.normalize(uav_features, dim=-1)
    logits = logit_scale.exp() * sat_features @ uav_features.t()
    labels = torch.arange(logits.size(0), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def build_model(args, num_classes):
    if args.method == "sample4geo":
        return TimmSharedInfoNCE(args.sample4geo_model, effective_image_size(args), args.pretrained)
    if args.method == "sues":
        return DualTimmClassifier(args.sues_model, num_classes, args.pretrained, args.share_weight)
    if args.method == "lpn":
        return LPNClassifier(args.lpn_model, num_classes, args.lpn_block, args.pretrained, args.share_weight)
    if args.method == "fsra":
        return FSRAClassifier(
            FSRA_MODEL,
            num_classes,
            FSRA_BLOCK,
            args.pretrained,
            effective_image_size(args),
            FSRA_SHARE_WEIGHT,
            FSRA_FEATURE_MODE,
            FSRA_SALIENCY_MODE,
        )
    raise ValueError(f"Unknown method: {args.method}")


def effective_image_size(args):
    return FSRA_IMAGE_SIZE if args.method == "fsra" else args.image_size


def build_optimizer(args, model):
    if args.method != "fsra":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
        return optimizer, scheduler

    backbone_param_ids = set()
    for branch in [model.sat_branch, model.uav_branch]:
        backbone_param_ids.update(id(param) for param in branch.backbone.parameters())
    backbone_params = [param for param in model.parameters() if id(param) in backbone_param_ids]
    extra_params = [param for param in model.parameters() if id(param) not in backbone_param_ids]
    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": 0.3 * FSRA_LR},
            {"params": extra_params, "lr": FSRA_LR},
        ],
        weight_decay=FSRA_WEIGHT_DECAY,
        momentum=0.9,
        nesterov=True,
    )
    milestones = [max(1, int(args.epochs * 0.6)), max(2, int(args.epochs * 0.9))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return optimizer, scheduler


def train_one_method(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    img_size = effective_image_size(args)
    train_dataset = SatUAVTrainDataset(
        args.dataset_dir,
        args.train_csv,
        args.image_dir,
        image_train_transform(img_size),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=args.method == "sample4geo",
    )

    model = build_model(args, len(train_dataset)).to(device)
    optimizer, scheduler = build_optimizer(args, model)
    ce_loss = nn.CrossEntropyLoss()

    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct_sat = 0
        correct_uav = 0
        total = 0
        start = time.time()

        for sat, uav, labels in train_loader:
            sat = sat.to(device, non_blocking=True)
            uav = uav.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if args.method == "sample4geo":
                sat_features, uav_features = model(sat, uav)
                loss = info_nce_loss(sat_features, uav_features, model.logit_scale)
            elif args.method in ["lpn", "fsra"]:
                sat_logits, uav_logits = model(sat, uav)
                loss = sum(ce_loss(logit, labels) for logit in sat_logits) / len(sat_logits)
                loss = loss + sum(ce_loss(logit, labels) for logit in uav_logits) / len(uav_logits)
                correct_sat += (sum(sat_logits).argmax(1) == labels).sum().item()
                correct_uav += (sum(uav_logits).argmax(1) == labels).sum().item()
                total += labels.size(0)
            else:
                sat_logits, uav_logits = model(sat, uav)
                loss = ce_loss(sat_logits, labels) + ce_loss(uav_logits, labels)
                correct_sat += (sat_logits.argmax(1) == labels).sum().item()
                correct_uav += (uav_logits.argmax(1) == labels).sum().item()
                total += labels.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        avg_loss = running_loss / max(1, len(train_dataset))
        print(f"[{args.method}] epoch {epoch}/{args.epochs}: loss={avg_loss:.4f}, time={time.time() - start:.1f}s")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_model(args, model, device, print_prefix=f"[{args.method}] val")
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, result_dir / f"{args.method}_best.pth")

    model.load_state_dict(best_state)
    final_metrics = evaluate_model(args, model, device, print_prefix=f"[{args.method}] best")
    with (result_dir / f"{args.method}_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[{args.method}] saved best checkpoint and metrics to {result_dir}")


def evaluate_checkpoint(args):
    if args.method == "all":
        raise ValueError("--eval-only needs one specific --method, not --method all.")
    if not args.checkpoint:
        raise ValueError("--eval-only requires --checkpoint PATH.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_rows = read_satuav_csv(Path(args.dataset_dir) / args.train_csv)
    model = build_model(args, len(train_rows)).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    metrics = evaluate_model(args, model, device, print_prefix=f"[{args.method}] eval")

    result_dir = Path(args.output_dir) / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / f"{args.method}_eval_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


@torch.no_grad()
def evaluate_model(args, model, device, print_prefix="val"):
    model.eval()
    img_size = effective_image_size(args)
    eval_dataset = SatUAVDetectionDataset(
        args.dataset_dir,
        args.val_csv,
        args.image_dir,
        image_eval_transform(img_size),
        negative_shift=args.negative_shift,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    labels_all = []
    scores_all = []
    start = time.time()
    for sat, uav, labels, _, _ in eval_loader:
        sat = sat.to(device, non_blocking=True)
        uav = uav.to(device, non_blocking=True)
        sat_features = model.encode_sat(sat)
        uav_features = model.encode_uav(uav)
        if sat_features.dim() == 3:
            sat_features = F.normalize(sat_features, p=2, dim=1) / math.sqrt(sat_features.size(-1))
            uav_features = F.normalize(uav_features, p=2, dim=1) / math.sqrt(uav_features.size(-1))
            sat_features = sat_features.flatten(1)
            uav_features = uav_features.flatten(1)
        else:
            sat_features = F.normalize(sat_features, dim=-1)
            uav_features = F.normalize(uav_features, dim=-1)
        cosine_sim = (sat_features * uav_features).sum(dim=1)
        distance = 1.0 - cosine_sim
        scores_all.extend(distance.detach().cpu().numpy().tolist())
        labels_all.extend(labels.numpy().tolist())

    y_true = np.array(labels_all, dtype=np.int64)
    y_score = np.array(scores_all, dtype=np.float64)
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    threshold = float(thresholds[best_idx])
    y_pred = (y_score > threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr_value = tp / (tp + fn) if (tp + fn) else 0.0
    tnr_value = tn / (tn + fp) if (tn + fp) else 0.0
    metrics = {
        "auc": float(auc),
        "threshold": threshold,
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred)),
        "tpr": float(tpr_value),
        "tnr": float(tnr_value),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "seconds_per_pair": float((time.time() - start) / max(1, len(eval_dataset))),
        "num_pairs": len(eval_dataset),
    }
    print(
        f"{print_prefix}: AUC={metrics['auc']:.4f}, ACC={metrics['acc']:.4f}, "
        f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
        f"F1={metrics['f1']:.4f}, TPR={metrics['tpr']:.4f}, TNR={metrics['tnr']:.4f}, "
        f"sec/pair={metrics['seconds_per_pair']:.6f}"
    )
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate cross-view baselines on SatUAV detection protocol.")
    parser.add_argument("--method", choices=["sample4geo", "sues", "lpn", "fsra", "all"], default="all")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--image-dir", type=str, default="full_960x720")
    parser.add_argument("--train-csv", type=str, default="train.csv")
    parser.add_argument("--val-csv", type=str, default="val.csv")
    parser.add_argument("--output-dir", type=str, default="baseline_results")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=100.0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--negative-shift", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--share-weight", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample4geo-model", type=str, default="convnext_base.fb_in22k_ft_in1k_384")
    parser.add_argument("--sues-model", type=str, default="resnet50")
    parser.add_argument("--lpn-model", type=str, default="seresnet50")
    parser.add_argument("--lpn-block", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.eval_only:
        evaluate_checkpoint(args)
        raise SystemExit(0)

    methods = ["sample4geo", "sues", "lpn", "fsra"] if args.method == "all" else [args.method]
    base_output = args.output_dir
    for method in methods:
        method_args = copy.copy(args)
        method_args.method = method
        method_args.output_dir = str(Path(base_output) / method)
        train_one_method(method_args)
