import torch
import torch.nn.functional as F
import CLIP.clip as clip
from PIL import Image
import os

class CLIPLoss:
    def __init__(self, device="cuda", model_name="ViT-B/32", save_dir="test_img"):
        self.device = device
        self.model, _ = clip.load(model_name, device=device)

        self.clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
        self.clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

        self.save_dir = save_dir
        if save_dir:
            os.makedirs(os.path.join(save_dir, "render"), exist_ok=True)
            os.makedirs(os.path.join(save_dir, "target"), exist_ok=True)

    def __call__(self, render_rgb: torch.Tensor, target_rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            render_rgb: [B, H, W, 3] - predicted patches (float in [0,1])
            target_rgb: [B, H, W, 3] - ground truth patches (float in [0,1])
        Returns:
            CLIP-based cosine similarity loss
        """
        B = render_rgb.shape[0]

        # [B, 3, H, W]
        render_rgb = render_rgb.permute(0, 3, 1, 2)
        target_rgb = target_rgb.permute(0, 3, 1, 2)

        # Resize to CLIP input size
        render_rgb = F.interpolate(render_rgb, size=(224, 224), mode='bilinear', align_corners=False)
        target_rgb = F.interpolate(target_rgb, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize
        render_norm = (render_rgb.clamp(0, 1) - self.clip_mean) / self.clip_std
        target_norm = ((target_rgb.detach().clamp(0, 1) - self.clip_mean) / self.clip_std)

  
        # Encode with CLIP
        render_features = self.model.encode_image(render_norm)
        target_features = self.model.encode_image(target_norm)

        # Normalize features
        render_features = render_features / render_features.norm(dim=-1, keepdim=True)
        target_features = target_features / target_features.norm(dim=-1, keepdim=True)

        # Cosine similarity & loss
        cosine_similarities = torch.sum(render_features * target_features, dim=-1)  # [B]
        loss = (1 - cosine_similarities).mean()

        """
        # Save debug images
        if self.save_dir:
            self.save_debug_images(render_rgb, target_rgb)
        """
        return loss

    def save_debug_images(self, render_tensor: torch.Tensor, target_tensor: torch.Tensor):
        """Save render and target images for debugging (after interpolation)"""
        render_imgs_np = render_tensor.detach().permute(0, 2, 3, 1).cpu().numpy()
        target_imgs_np = target_tensor.detach().permute(0, 2, 3, 1).cpu().numpy()

        for i in range(min(render_imgs_np.shape[0], 8)):
            render_img = Image.fromarray((render_imgs_np[i] * 255).astype('uint8'))
            target_img = Image.fromarray((target_imgs_np[i] * 255).astype('uint8'))

            render_img.save(os.path.join(self.save_dir, "render", f"render_{i}.png"))
            target_img.save(os.path.join(self.save_dir, "target", f"target_{i}.png"))

