import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from safetensors.torch import load_file
from typing import Any
from utils.paths import ckpt_path_flairhub_ir, ckpt_path_flairhub_rgb_large, ckpt_path_flairhub_rgb_base, ckpt_path_flairhub_rgb_small, ckpt_path_flairhub_rgb_tiny


def fuse_features(c2, c3, c4, c5):
    target = c2.shape[-2:]  # 128x128

    # Upsample coarser maps to 128x128 (bilinear, no learned weights)
    c3_up = torch.nn.functional.interpolate(c3, size=target, mode='bilinear', align_corners=False)
    c4_up = torch.nn.functional.interpolate(c4, size=target, mode='bilinear', align_corners=False)
    c5_up = torch.nn.functional.interpolate(c5, size=target, mode='bilinear', align_corners=False)

    # Option A — concat: [B, 1920, 128, 128]  (128+256+512+1024)
    fused = torch.cat([c2, c3_up, c4_up, c5_up], dim=1)
    return fused 


class DecoderWrapper(nn.Module):
    def __init__(self, decoder: nn.Module, segmentation_head: nn.Module) -> None:
        super().__init__()
        self.decoder = decoder
        self.segmentation_head = segmentation_head

    def forward(self, *features: Any) -> torch.Tensor:
        decoder_output = self.decoder(*features)
        return self.segmentation_head(decoder_output)


class FlairHubModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        encoder_input_dim: int,
        decoder_input_dim: int,
        num_classes: int,
        encoder_arch: str,
        img_size: int = 512,
        encoder_only: bool = False,
        use_float16: bool = False,
    ) -> None:
        super().__init__()

        encoder_name, decoder_name = encoder_arch.split("-", 1)
        self.encoder_only = encoder_only
        self.use_float16 = use_float16

        # Build encoder
        encoder_model = self._build_smp_model(
            decoder_name, encoder_name, num_classes, encoder_input_dim, img_size
        )
        self.encoder = encoder_model.encoder

        if not self.encoder_only:
            # Build decoder with its own input_dim
            decoder_model = self._build_smp_model(
                decoder_name, encoder_name, num_classes, decoder_input_dim, img_size
            )
            self.decoder = DecoderWrapper(
                decoder_model.decoder, decoder_model.segmentation_head
            )

        self.ckpt_path = ckpt_path
        self._load_weights(ckpt_path)
        
        # Convert to float16 if requested
        if self.use_float16:
            self.encoder.half()
            if not self.encoder_only:
                self.decoder.half()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_smp_model(
        arch: str,
        encoder_name: str,
        classes: int,
        in_channels: int,
        img_size: int,
    ) -> nn.Module:
        """Try several fallback strategies when creating the SMP model."""
        for enc, use_img_size in [
            (encoder_name, True),
            ("tu-" + encoder_name, True),
            ("tu-" + encoder_name, False),
        ]:
            try:
                kwargs = dict(
                    arch=arch,
                    encoder_name=enc,
                    classes=classes,
                    in_channels=in_channels,
                )
                if use_img_size:
                    kwargs["img_size"] = img_size
                return smp.create_model(**kwargs)
            except (KeyError, TypeError):
                continue
        raise ValueError(
            f"Could not instantiate SMP model for encoder='{encoder_name}', arch='{arch}'"
        )

    def _load_weights(self, ckpt_path: str) -> None:
        """Load and remap encoder + decoder weights from the checkpoint."""
        state_dict = load_file(ckpt_path)

        encoder_sd = {
            k.replace("model.encoders.AERIAL_RGBI.seg_model.", ""): v
            for k, v in state_dict.items()
            if k.startswith("model.encoders.AERIAL_RGBI.seg_model.")
        }

        self.encoder.load_state_dict(encoder_sd, strict=True)

        if not self.encoder_only:
            decoder_sd = {
                k.replace("model.main_decoders.AERIAL_LABEL-COSIA.seg_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("model.main_decoders.AERIAL_LABEL-COSIA.seg_model.")
            }
        
            self.decoder.load_state_dict(decoder_sd, strict=True)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # Ensure input matches model dtype
        if self.use_float16:
            imgs = imgs.half()
        
        if "RGB" in self.ckpt_path:
            features = self.encoder(imgs[:, [0, 1, 2]])
        elif "IR" in self.ckpt_path:
            features = self.encoder(imgs[:, [3, 0, 1]])

        if self.encoder_only:
            _, _, c2, c3, c4, c5 = features
            fused_features = fuse_features(c2, c3, c4, c5)
            return fused_features
        else:
            out = self.decoder(*features)[
                :, [0, 4, 3, 5, 6, 13, 12, 14, 11, 8, 9, 10, 2, 7, 1]
            ]
            return out


class FlairHubWrapper(nn.Module):
    def __init__(self,
        rgb_only: bool,
        size: str = "base",  # tiny, small, base, large
        encoder_only: bool = True,
        fuse_mode: str = "concat",  # or average
        use_float16=False,
    ) -> None:
        super().__init__()
        self.rgb_only = rgb_only
        self.fuse_mode = fuse_mode

        encoder_arch = {
            'tiny': "swin_tiny_patch4_window7_224-upernet",
            'small': "swin_small_patch4_window7_224-upernet",
            'base': "swin_base_patch4_window12_384-upernet",
            'large': "swin_large_patch4_window12_384-upernet"
        }[size]

        ckpt_path_flairhub_rgb = {
            'tiny': ckpt_path_flairhub_rgb_tiny,
            'small': ckpt_path_flairhub_rgb_small,
            'base': ckpt_path_flairhub_rgb_base,
            'large': ckpt_path_flairhub_rgb_large
        }[size]

        self.flairhub_model_rgb = FlairHubModel(
            ckpt_path=ckpt_path_flairhub_rgb,
            encoder_input_dim=3,
            decoder_input_dim=1,
            num_classes=19,
            encoder_arch=encoder_arch,
            img_size=512,
            encoder_only=encoder_only,
            use_float16=use_float16
        )
        if not rgb_only:
            self.flairhub_model_ir = FlairHubModel(
                ckpt_path=ckpt_path_flairhub_ir,
                encoder_input_dim=3,
                decoder_input_dim=1,
                num_classes=19,
                encoder_arch=encoder_arch,
                img_size=512,
                encoder_only=encoder_only,
                use_float16=use_float16
            )
        else:
            self.flairhub_model_ir = None

    def forward(self, x: torch.Tensor, n_channels: torch.Tensor) -> torch.Tensor:
        features_rgb = self.flairhub_model_rgb(x)
        if not self.rgb_only:
            features_ir = self.flairhub_model_ir(x)
            if self.fuse_mode == "concat":
                output = torch.cat([features_rgb, features_ir], dim=1)
            elif self.fuse_mode == "average":
                output = torch.where(
                    (n_channels.sum(-1) == 4)[..., None, None, None],
                    (features_ir + features_rgb) / 2,
                    features_rgb,
                )
            else:
                raise ValueError(f"Invalid fuse_mode='{self.fuse_mode}'")
        else:
            output = features_rgb
        return output