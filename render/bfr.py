from gfpgan.utils import GFPGANv1Clean
import torch
class GFPGAN(GFPGANv1Clean):

    def forward(self, x, normalize = True, return_latents=False, return_rgb=True, randomize_noise=True, **kwargs):
        if normalize:
            x = (x - 0.5) / 0.5
        image = super().forward(x, return_latents=return_latents, return_rgb=return_rgb, randomize_noise=randomize_noise, **kwargs)
        if type(image) is tuple:
            image = image[0]
        image = (image + 1) / 2
        image = image.clamp(0, 1)
        return image

    def restore_from_render(self, x, **kwargs):
        lq_image = x.permute(0, 3, 1, 2).contiguous()
        shape = lq_image.shape[-2:]
        lq_image = torch.nn.functional.interpolate(lq_image, (512, 512), mode = 'bilinear', align_corners = True)
        hq_image = self.forward(lq_image, **kwargs)
        hq_image = torch.nn.functional.interpolate(hq_image, shape, mode = 'bilinear', align_corners = True)
        hq_image = hq_image.permute(0, 2, 3, 1).contiguous()
        return hq_image

class DummyGFPGAN():
    def __call__(self, x):
        return x

    def restore_from_render(self, x):
        return x
