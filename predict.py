import os
from pathlib import Path as SysPath
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from cog import BasePredictor, Input, Path

from models.bisenet import BiSeNet
from utils.common import vis_parsing_maps

# Pillow resampling (backwards compatible)
try:
    Resampling = Image.Resampling  # Pillow >=9
except AttributeError:
    Resampling = Image  # Pillow <9

# Keep CPU containers polite by default; let env override
if "OMP_NUM_THREADS" not in os.environ:
    torch.set_num_threads(1)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 19


def prepare_image(image: Image.Image, input_size: Tuple[int, int] = (512, 512)) -> torch.Tensor:
    """Resize + normalize to tensor batch[1,C,H,W]."""
    resized = image.resize(input_size, resample=Resampling.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(resized).unsqueeze(0)


def resolve_weights_dir() -> SysPath:
    """Resolve weights directory (env WEIGHTS_DIR > repo /weights > alongside file)."""
    env = os.getenv("WEIGHTS_DIR")
    if env:
        return SysPath(env)
    here = SysPath(__file__).resolve().parent
    candidates = [here / "weights", here.parent / "weights"]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default location even if it doesn't exist yet


def load_model(backbone: str, num_classes: int, weight_path: SysPath, device: torch.device) -> torch.nn.Module:
    """Instantiate and load BiSeNet with robust state_dict handling."""
    model = BiSeNet(num_classes, backbone_name=backbone).to(device)
    if not weight_path.exists():
        raise FileNotFoundError(f"Weights not found at: {weight_path}")

    sd = torch.load(str(weight_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    # Strip possible 'module.' prefixes from DataParallel checkpoints
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load weights from {weight_path} for backbone '{backbone}'. "
            f"Make sure NUM_CLASSES={num_classes} matches the checkpoint. Original error: {e}"
        )
    model.eval()
    return model


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load default model for efficiency; can be swapped in predict()."""
        self.device = pick_device()
        self.num_classes = NUM_CLASSES
        self.current_backbone: Optional[str] = None
        self.model: Optional[torch.nn.Module] = None
        self.weights_dir = resolve_weights_dir()

        default_backbone = "resnet18"
        default_weights = self.weights_dir / "resnet18.pt"
        if default_weights.exists():
            self.model = load_model(default_backbone, self.num_classes, default_weights, self.device)
            self.current_backbone = default_backbone

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image."),
        backbone: str = Input(description="Backbone", choices=["resnet18", "resnet34"], default="resnet18"),
        input_size: int = Input(description="Square model input resolution.", ge=128, le=1024, default=512),
        use_half: bool = Input(description="Use FP16 on CUDA for speed/memory (experimental).", default=True),
    ) -> List[Path]:
        """
        Face parsing on a single image.

        Outputs:
          - <stem>.overlay.png : color overlay at original resolution
          - <stem>.mask.png    : uint8 label mask (0..18) at original resolution
        """
        weight_path = self.weights_dir / f"{backbone}.pt"

        if self.model is None or self.current_backbone != backbone:
            self.model = load_model(backbone, self.num_classes, weight_path, self.device)
            self.current_backbone = backbone

        # Optional FP16 on CUDA only
        if use_half and self.device.type == "cuda":
            self.model.half()
        else:
            self.model.float()

        overlay, restored_mask = self._process_single_image(
            img_path=SysPath(str(image)),
            input_size=(input_size, input_size),
            use_half=use_half and self.device.type == "cuda",
        )

        stem = "output"
        out_overlay = SysPath(f"{stem}.0.png")
        out_mask = SysPath(f"{stem}.1.png")

        Image.fromarray(overlay).save(out_overlay)
        restored_mask.save(out_mask)

        return [Path(str(out_overlay)), Path(str(out_mask))]

    def _process_single_image(
        self,
        img_path: SysPath,
        input_size: Tuple[int, int],
        use_half: bool = False,
    ) -> Tuple[np.ndarray, Image.Image]:
        """Run inference for one image and return (overlay RGB np.uint8, restored PIL mask)."""
        image = Image.open(img_path).convert("RGB")
        original_size = image.size  # (W, H)

        batch = prepare_image(image, input_size=input_size)
        batch = batch.half() if (use_half and self.device.type == "cuda") else batch
        batch = batch.to(self.device, non_blocking=True)

        # Forward (BiSeNet often returns a tuple; take main logits)
        logits = self.model(batch)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]

        # logits: [1, num_classes, H, W] -> (H, W) argmax
        pred = logits.squeeze(0).float().detach().cpu().numpy().argmax(0).astype(np.uint8)

        raw_mask_img = Image.fromarray(pred)
        restored_mask = raw_mask_img.resize(original_size, resample=Resampling.NEAREST)
        restored_np = np.array(restored_mask, dtype=np.uint8)

        overlay = vis_parsing_maps(image=image, segmentation_mask=restored_np)  # expects HxWx3 uint8
        if overlay.dtype != np.uint8:
            overlay = overlay.astype(np.uint8)

        return overlay, restored_mask
