import torch
import numpy as np
import torch.nn.functional as F
import cv2
from dehaze.src import dehaze

# def numpy_sigmoid(x):
#     return 1/(1+np.exp(-x))


def sigmoid_inverse(y):
    epsilon = 10 ** (-3)
    y = F.relu(y - epsilon) + epsilon
    y = 1 - epsilon - F.relu((1 - epsilon) - y)
    y = (1 / y) - 1
    output = -torch.log(y)
    return output


class Sigmoid:
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, images):
        return torch.sigmoid(images)


class SigmoidInverse:

    def __init__(self):
        self.num_parameters = 0

    def __call__(self, images):
        return sigmoid_inverse(images)


new_sig_inv = SigmoidInverse()


class AdjustContrast:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["contrast"]

    def __call__(self, images: torch.Tensor, parameters: torch.Tensor):

        assert images.dim() == 4
        assert images.shape[0] == parameters.shape[0]

        batch_size = parameters.shape[0]
        mean = images.view(batch_size, -1).mean(1)
        mean = mean.view(batch_size, 1, 1, 1)
        parameters = parameters.view(batch_size, 1, 1, 1)
        editted = (images - mean) * (parameters + 1) + mean
        editted = F.relu(editted)
        editted = 1 - F.relu(1 - editted)
        return editted


class AdjustDehaze:

    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["dehaze"]

    def __call__(self, images, parameters):
        """
        Takes a batch of images where B (the last dim) is the batch size
        args:
            images: torch.Tensor # B H W C
            parameters :torch.Tensor # N
        return:
            output: torch.Tensor #  B H W C
        """
        assert images.dim() == 4
        batch_size = parameters.shape[0]
        output = []
        for image_index in range(batch_size):
            image = images[image_index].numpy()
            scale = max((image.shape[:2])) / 512.0
            omega = float(parameters[image_index])
            editted = (
                dehaze.DarkPriorChannelDehaze(
                    wsize=int(15 * scale),
                    radius=int(80 * scale),
                    omega=omega,
                    t_min=0.25,
                    refine=True,
                )(image * 255.0)
                / 255.0
            )
            editted = torch.tensor(editted)
            editted = F.relu(editted)
            editted = 1 - F.relu(1 - editted)
            output.append(editted)
        output = torch.stack(output)
        return output


class AdjustClarity:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["clarity"]

    def __call__(self, images, parameters):
        """
        Takes a batch of images where B (the last dim) is the batch size
        args:
            images: torch.Tensor # B H W C
            parameters :torch.Tensor # N
        return:
            output: torch.Tensor #  B H W C
        """
        assert images.dim() == 4
        batch_size = parameters.shape[0]
        output = []
        clarity = parameters.view(batch_size, 1, 1, 1)
        for image in images:
            input = image.numpy()
            scale = max((input.shape[:2])) / 512.0
            unsharped = (
                cv2.bilateralFilter(
                    (input * 255.0).astype(np.uint8),
                    int(32 * scale),
                    50,
                    10 * scale,
                )
                / 255.0
            )
            output.append(torch.tensor(unsharped))
        output = torch.stack(output)
        editted_images = images + (images - output) * clarity

        return editted_images


class AdjustExposure:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["exposure"]

    def __call__(self, images, parameters):
        batch_size = parameters.shape[0]
        exposure = parameters.view(batch_size, 1, 1, 1)
        output = images + exposure * 5
        return output


class AdjustTemp:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["temp"]

    def __call__(self, images, parameters):
        batch_size = parameters.shape[0]
        temp = parameters.view(batch_size, 1, 1, 1)
        editted = torch.clone(images)

        index_high = (temp > 0).view(-1)
        index_low = (temp <= 0).view(-1)

        editted[index_high, :, :, 1] += temp[index_high, :, :, 0] * 1.6
        editted[index_high, :, :, 2] += temp[index_high, :, :, 0] * 2
        editted[index_low, :, :, 0] -= temp[index_low, :, :, 0] * 2.0
        editted[index_low, :, :, 1] -= temp[index_low, :, :, 0] * 1.0

        return editted


class AdjustTint:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["tint"]

    def __call__(self, images, parameters):
        batch_size = parameters.shape[0]
        tint = parameters.view(batch_size, 1, 1, 1)
        editted = torch.clone(images)

        index_high = (tint > 0).view(-1)
        index_low = (tint <= 0).view(-1)

        editted[index_high, :, :, 0] += tint[index_high, :, :, 0] * 2
        editted[index_high, :, :, 2] += tint[index_high, :, :, 0] * 1
        editted[index_low, :, :, 1] -= tint[index_low, :, :, 0] * 2
        editted[index_low, :, :, 2] -= tint[index_low, :, :, 0] * 1

        return editted


class AdjustShadows:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["shadows"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        shadows = parameters.view(batch_size, 1, 1)

        v = list_hsv[2]

        # Calculate shadows mask

        shadows_mask = 1 - torch.sigmoid((v - 0.0) * 5.0)
        # Adjust v channel based on shadows mask
        adjusted_v = v * (1 + shadows_mask * shadows * 5.0)

        return [list_hsv[0], list_hsv[1], adjusted_v]


class AdjustHighlights:  # I should change the sigmoid to torch.sigmoid
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["highlights"]

    # def custom_sigmoid(self, x):
    #     return 1 / (1 + torch.exp(-x))

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        highlights = parameters.view(batch_size, 1, 1)

        v = list_hsv[2]

        # Calculate highlights mask using custom sigmoid function
        highlights_mask = torch.sigmoid((v - 1) * 5)

        # Adjust v channel based on highlights mask
        adjusted_v = 1 - (1 - v) * (1 - highlights_mask * highlights * 5)

        return [list_hsv[0], list_hsv[1], adjusted_v]


class AdjustBlacks:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["blacks"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        blacks = parameters.view(batch_size, 1, 1)
        blacks = blacks + 1
        v = list_hsv[2]

        # Calculate the adjustment factor
        adjustment_factor = (torch.sqrt(blacks) - 1) * 0.2

        # Adjust the v channel
        adjusted_v = v + (1 - v) * adjustment_factor

        return [list_hsv[0], list_hsv[1], adjusted_v]


class AdjustWhites:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["whites"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        whites = parameters.view(batch_size, 1, 1)
        whites = whites + 1
        v = list_hsv[2]

        # Calculate the adjustment factor
        adjustment_factor = (torch.sqrt(whites) - 1) * 0.2

        # Adjust the v channel
        adjusted_v = v + v * adjustment_factor

        return [list_hsv[0], list_hsv[1], adjusted_v]


class Bgr2Hsv:
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, images):
        editted = images

        max_bgr, _ = editted.max(dim=-1, keepdim=True)
        min_bgr, _ = editted.min(dim=-1, keepdim=True)

        b = editted[..., 0]
        g = editted[..., 1]
        r = editted[..., 2]

        b_g = b - g
        g_r = g - r
        r_b = r - b

        b_min_flg = (1 - F.relu(torch.sign(b_g))) * F.relu(torch.sign(r_b))
        g_min_flg = (1 - F.relu(torch.sign(g_r))) * F.relu(torch.sign(b_g))
        r_min_flg = (1 - F.relu(torch.sign(r_b))) * F.relu(torch.sign(g_r))

        epsilon = 10 ** (-5)
        h1 = 60 * g_r / (max_bgr.squeeze() - min_bgr.squeeze() + epsilon) + 60
        h2 = 60 * b_g / (max_bgr.squeeze() - min_bgr.squeeze() + epsilon) + 180
        h3 = 60 * r_b / (max_bgr.squeeze() - min_bgr.squeeze() + epsilon) + 300
        h = h1 * b_min_flg + h2 * r_min_flg + h3 * g_min_flg

        v = max_bgr.squeeze()
        s = (max_bgr.squeeze() - min_bgr.squeeze()) / (
            max_bgr.squeeze() + epsilon
        )

        return [h, s, v]


class AdjustVibrance:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["vibrance"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        vibrance = parameters.view(batch_size, 1, 1)
        vibrance = vibrance + 1
        s = list_hsv[1]

        # Calculate vibrance flag using custom sigmoid function
        vibrance_flg = -torch.sigmoid((s - 0.5) * 10) + 1

        # Adjust the s channel
        adjusted_s = s * vibrance * vibrance_flg + s * (1 - vibrance_flg)

        return [list_hsv[0], adjusted_s, list_hsv[2]]


class AdjustSaturation:
    def __init__(self):
        self.num_parameters = 1
        self.window_names = ["parameter"]
        self.slider_names = ["saturation"]

    def __call__(self, list_hsv, parameters):
        batch_size = parameters.shape[0]
        saturation = parameters.view(batch_size, 1, 1)
        saturation = saturation + 1
        s = list_hsv[1]

        # Adjust the saturation
        s_ = s * saturation
        s_ = F.relu(s_)
        s_ = 1 - F.relu(1 - s_)

        return [list_hsv[0], s_, list_hsv[2]]


class Hsv2Bgr:
    def __init__(self):
        self.num_parameters = 0

    def __call__(self, list_hsv):
        h, s, v = list_hsv

        # Adjust h values
        h = (
            h
            * torch.relu(torch.sign(h - 0))
            * (1 - torch.relu(torch.sign(h - 360)))
            + (h - 360)
            * torch.relu(torch.sign(h - 360))
            * (1 - torch.relu(torch.sign(h - 720)))
            + (h + 360)
            * torch.relu(torch.sign(h + 360))
            * (1 - torch.relu(torch.sign(h - 0)))
        )

        # Calculate h flags
        h60_flg = torch.relu(torch.sign(h - 0)) * (
            1 - torch.relu(torch.sign(h - 60))
        )
        h120_flg = torch.relu(torch.sign(h - 60)) * (
            1 - torch.relu(torch.sign(h - 120))
        )
        h180_flg = torch.relu(torch.sign(h - 120)) * (
            1 - torch.relu(torch.sign(h - 180))
        )
        h240_flg = torch.relu(torch.sign(h - 180)) * (
            1 - torch.relu(torch.sign(h - 240))
        )
        h300_flg = torch.relu(torch.sign(h - 240)) * (
            1 - torch.relu(torch.sign(h - 300))
        )
        h360_flg = torch.relu(torch.sign(h - 300)) * (
            1 - torch.relu(torch.sign(h - 360))
        )

        C = v * s
        b = (
            v
            - C
            + C * (h240_flg + h300_flg)
            + C * ((h / 60 - 2) * h180_flg + (6 - h / 60) * h360_flg)
        )
        g = (
            v
            - C
            + C * (h120_flg + h180_flg)
            + C * ((h / 60) * h60_flg + (4 - h / 60) * h240_flg)
        )
        r = (
            v
            - C
            + C * (h60_flg + h360_flg)
            + C * ((h / 60 - 4) * h300_flg + (2 - h / 60) * h120_flg)
        )

        # Add an extra dimension to b, g, r to concatenate them correctly
        b = b.unsqueeze(-1)
        g = g.unsqueeze(-1)
        r = r.unsqueeze(-1)

        bgr = torch.cat([b, g, r], dim=-1)

        return bgr


# class Srgb2Photopro:
#     def __init__(self):
#         self.num_parameters = 0

#     def __call__(self, images):
#         srgb = images.clone()
#         k = 0.055
#         thre_srgb = 0.04045

#         a = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
#                           [0.2126729, 0.7151522, 0.0721750],
#                           [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
#         b = torch.tensor([[1.3459433, -0.2556075, -0.0511118],
#                           [-0.5445989, 1.5081673, 0.0205351],
#                           [0.0000000, 0.0000000, 1.2118128]], dtype=torch.float32)

#         M = torch.matmul(b, a)
#         M = M / M.sum(dim=1, keepdim=True)

#         thre_photopro = 1 / 512.0

#         # sRGB to linear RGB
#         srgb = torch.where(srgb <= thre_srgb, srgb / 12.92, ((srgb + k) / (1 + k)) ** 2.4)

#         sb = srgb[..., 0:1]
#         sg = srgb[..., 1:2]
#         sr = srgb[..., 2:3]

#         photopror = sr * M[0][0] + sg * M[0][1] + sb * M[0][2]
#         photoprog = sr * M[1][0] + sg * M[1][1] + sb * M[1][2]
#         photoprob = sr * M[2][0] + sg * M[2][1] + sb * M[2][2]

#         photopro = torch.cat((photoprob, photoprog, photopror), dim=-1)
#         photopro = torch.clamp(photopro, 0, 1)
#         photopro = torch.where(photopro >= thre_photopro, photopro ** (1 / 1.8), photopro * 16)

#         return photopro


class Srgb2Photopro:
    def __init__(self):
        self.num_parameters = 0
        k = 0.055
        thre_srgb = 0.04045

        self.k = k
        self.thre_srgb = thre_srgb
        self.thre_photopro = 1 / 512.0

        # Transformation matrices
        a = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [
                [1.3459433, -0.2556075, -0.0511118],
                [-0.5445989, 1.5081673, 0.0205351],
                [0.0000000, 0.0000000, 1.2118128],
            ],
            dtype=torch.float32,
        )

        self.M = torch.matmul(b, a)
        self.M = self.M / self.M.sum(dim=1, keepdim=True)

    def __call__(self, images):
        srgb = images.clone()

        with torch.no_grad():  # Disable gradient computation for inference
            # sRGB to linear RGB
            srgb = torch.where(
                srgb <= self.thre_srgb,
                srgb / 12.92,
                ((srgb + self.k) / (1 + self.k)) ** 2.4,
            )

            sb = srgb[..., 0:1]
            sg = srgb[..., 1:2]
            sr = srgb[..., 2:3]

            # Apply the transformation matrix
            photopror = (
                sr * self.M[0][0] + sg * self.M[0][1] + sb * self.M[0][2]
            )
            photoprog = (
                sr * self.M[1][0] + sg * self.M[1][1] + sb * self.M[1][2]
            )
            photoprob = (
                sr * self.M[2][0] + sg * self.M[2][1] + sb * self.M[2][2]
            )

            photopro = torch.cat((photoprob, photoprog, photopror), dim=-1)
            photopro = torch.clamp(photopro, 0, 1)

            # Apply the Photopro gamma correction
            photopro = torch.where(
                photopro >= self.thre_photopro,
                photopro ** (1 / 1.8),
                photopro * 16,
            )

            # Clear intermediate tensors
            del srgb, sb, sg, sr, photopror, photoprog, photoprob
            torch.cuda.empty_cache()

        return photopro


# class Photopro2Srgb:
#     def __init__(self):
#         self.num_parameters = 0

#     def __call__(self, photopro_tensor):
#         photopro = photopro_tensor.clone()  # Make a copy to avoid modifying the input tensor
#         thre_photopro = 1/512.0*16

#         a = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
#                           [0.2126729, 0.7151522, 0.0721750],
#                           [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float32)
#         b = torch.tensor([[1.3459433, -0.2556075, -0.0511118],
#                           [-0.5445989, 1.5081673, 0.0205351],
#                           [0.0000000, 0.0000000, 1.2118128]], dtype=torch.float32)
#         M = torch.matmul(b, a)
#         M = M / M.sum(dim=1, keepdim=True)
#         M = torch.linalg.inv(M)
#         k = 0.055
#         thre_srgb = 0.04045 / 12.92

#         # Apply transformations
#         mask = photopro < thre_photopro
#         photopro[mask] *= 1.0 / 16
#         photopro[~mask] = photopro[~mask] ** 1.8

#         photoprob = photopro[:, :, :, 0:1]
#         photoprog = photopro[:, :, :, 1:2]
#         photopror = photopro[:, :, :, 2:3]

#         sr = photopror * M[0, 0] + photoprog * M[0, 1] + photoprob * M[0, 2]
#         sg = photopror * M[1, 0] + photoprog * M[1, 1] + photoprob * M[1, 2]
#         sb = photopror * M[2, 0] + photoprog * M[2, 1] + photoprob * M[2, 2]

#         srgb = torch.cat((sb, sg, sr), dim=-1)

#         # Clip and apply final transformations
#         srgb = torch.clamp(srgb, 0, 1)
#         mask = srgb > thre_srgb
#         srgb[mask] = (1 + k) * srgb[mask] ** (1 / 2.4) - k
#         srgb[~mask] *= 12.92

#         return srgb


class Photopro2Srgb:
    def __init__(self):
        self.num_parameters = 0
        self.k = 0.055
        self.thre_srgb = 0.04045 / 12.92
        self.thre_photopro = 1 / 512.0 * 16

        # Transformation matrices
        a = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=torch.float32,
        )
        b = torch.tensor(
            [
                [1.3459433, -0.2556075, -0.0511118],
                [-0.5445989, 1.5081673, 0.0205351],
                [0.0000000, 0.0000000, 1.2118128],
            ],
            dtype=torch.float32,
        )

        self.M = torch.matmul(b, a)
        self.M = self.M / self.M.sum(dim=1, keepdim=True)
        self.M_inv = torch.linalg.inv(self.M)

    def __call__(self, photopro_tensor):
        with torch.no_grad():  # Disable gradient computation for inference
            photopro = (
                photopro_tensor.clone()
            )  # Make a copy to avoid modifying the input tensor
            # photopro = photopro.to(torch.float16)
            # Apply gamma correction
            mask = photopro < self.thre_photopro
            photopro[mask] *= 1.0 / 16
            photopro[~mask] = photopro[~mask] ** 1.8

            # Separate channels
            photoprob = photopro[..., 0:1]
            photoprog = photopro[..., 1:2]
            photopror = photopro[..., 2:3]

            # Apply the inverse transformation matrix
            sr = (
                photopror * self.M_inv[0, 0]
                + photoprog * self.M_inv[0, 1]
                + photoprob * self.M_inv[0, 2]
            )
            sg = (
                photopror * self.M_inv[1, 0]
                + photoprog * self.M_inv[1, 1]
                + photoprob * self.M_inv[1, 2]
            )
            sb = (
                photopror * self.M_inv[2, 0]
                + photoprog * self.M_inv[2, 1]
                + photoprob * self.M_inv[2, 2]
            )
            del photopror, photoprog, photoprob
            srgb = torch.cat((sb, sg, sr), dim=-1)
            del sr, sg, sb
            # Apply sRGB transformation
            srgb = torch.clamp(srgb, 0, 1)
            mask = srgb > self.thre_srgb
            srgb[mask] = (1 + self.k) * srgb[mask] ** (1 / 2.4) - self.k
            srgb[~mask] *= 12.92

            # Clear intermediate tensors
            del mask
            torch.cuda.empty_cache()

        return srgb


class PhotoEditor:
    def __init__(self, sliders="all"):
        self.edit_funcs = [
            Srgb2Photopro(),
            AdjustDehaze(),
            AdjustClarity(),
            AdjustContrast(),
            SigmoidInverse(),
            AdjustExposure(),
            AdjustTemp(),
            AdjustTint(),
            Sigmoid(),
            Bgr2Hsv(),
            AdjustWhites(),
            AdjustBlacks(),
            AdjustHighlights(),
            AdjustShadows(),
            AdjustVibrance(),
            AdjustSaturation(),
            Hsv2Bgr(),
            Photopro2Srgb(),
        ]
        self.sliders = sliders
        self.num_parameters = 0
        if sliders == "all":
            for edit_func in self.edit_funcs:
                self.num_parameters += edit_func.num_parameters
        else:
            for edit_func in self.edit_funcs:
                if edit_func.num_parameters == 0:
                    self.num_parameters += edit_func.num_parameters
                elif edit_func.slider_names[0] in sliders:
                    self.num_parameters += edit_func.num_parameters

    def __call__(self, images, parameters):
        editted_images = images.clone()
        num_parameters = 0
        assert (
            images.shape[-1] == 3
        )  # make sure that the image shape is (B,H,W,C)
        assert images.dim() == 4  # make sure that the image is batched
        for edit_func in self.edit_funcs:

            if self.sliders == "all":

                if edit_func.num_parameters == 0:
                    editted_images = edit_func(editted_images)
                else:
                    start = num_parameters
                    end = num_parameters + edit_func.num_parameters
                    editted_images = edit_func(
                        editted_images, parameters[:, start:end]
                    )
                num_parameters = num_parameters + edit_func.num_parameters

            else:

                if edit_func.num_parameters == 0:
                    editted_images = edit_func(editted_images)
                else:
                    if edit_func.slider_names[0] in self.sliders:
                        start = num_parameters
                        end = num_parameters + edit_func.num_parameters
                        editted_images = edit_func(
                            editted_images, parameters[:, start:end]
                        )
                        num_parameters = (
                            num_parameters + edit_func.num_parameters
                        )

        editted_images = editted_images.type(torch.float32)

        return editted_images
