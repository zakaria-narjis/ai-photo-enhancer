import unittest

import torch

from src.envs.new_edit_photo import SLIDER_ORDER, PhotoEditor


def action(editor: PhotoEditor, **values: float) -> torch.Tensor:
    return torch.tensor([[values.get(name, 0.0) for name in editor.sliders]])


class PhotoEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.editor = PhotoEditor()

    def test_action_contract_is_the_legacy_ten_slider_order(self) -> None:
        self.assertEqual(self.editor.sliders, SLIDER_ORDER)
        self.assertEqual(self.editor.num_parameters, 10)
        subset = PhotoEditor(["saturation", "temp", "exposure"])
        self.assertEqual(subset.sliders, ("exposure", "temp", "saturation"))
        self.assertEqual(subset.num_parameters, 3)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            PhotoEditor(["clarity"])

    def test_neutral_actions_are_an_identity(self) -> None:
        image = torch.rand(2, 37, 53, 3)
        output = self.editor(image, torch.zeros(2, 10))
        self.assertLess((output - image).abs().max().item(), 2e-5)

    def test_output_is_batched_finite_and_bounded_at_extremes(self) -> None:
        image = torch.rand(4, 32, 48, 3)
        actions = torch.tensor(
            [
                [-1.0] * 10,
                [1.0] * 10,
                [-1.0, 1.0] * 5,
                [1.0, -1.0] * 5,
            ]
        )
        output = self.editor(image, actions)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreaterEqual(output.min().item(), 0.0)
        self.assertLessEqual(output.max().item(), 1.0)

    def test_exposure_is_ev_scaled_and_monotone(self) -> None:
        image = torch.full((1, 16, 16, 3), 0.18)
        dark = self.editor(image, action(self.editor, exposure=-0.2)).mean()
        neutral = self.editor(image, action(self.editor)).mean()
        bright = self.editor(image, action(self.editor, exposure=0.2)).mean()
        self.assertLess(dark.item(), neutral.item())
        self.assertGreater(bright.item(), neutral.item())

    def test_white_balance_directions_match_lightroom(self) -> None:
        image = torch.full((1, 16, 16, 3), 0.4)
        warm = self.editor(image, action(self.editor, temp=0.7)).mean((0, 1, 2))
        cool = self.editor(image, action(self.editor, temp=-0.7)).mean((0, 1, 2))
        magenta = self.editor(image, action(self.editor, tint=0.7)).mean((0, 1, 2))
        green = self.editor(image, action(self.editor, tint=-0.7)).mean((0, 1, 2))
        self.assertGreater(warm[0].item(), warm[2].item())
        self.assertGreater(cool[2].item(), cool[0].item())
        self.assertLess(magenta[1].item(), (magenta[0] + magenta[2]).item() / 2)
        self.assertGreater(green[1].item(), (green[0] + green[2]).item() / 2)

    def test_shadows_and_highlights_target_their_regions(self) -> None:
        image = torch.empty(1, 32, 32, 3)
        image[:, :, :16] = 0.06
        image[:, :, 16:] = 0.75
        neutral = self.editor(image, action(self.editor))
        shadows = self.editor(image, action(self.editor, shadows=0.8))
        highlights = self.editor(image, action(self.editor, highlights=0.8))

        shadow_change = (shadows - neutral).mean((0, 1, 3))
        highlight_change = (highlights - neutral).mean((0, 1, 3))
        self.assertGreater(shadow_change[:16].mean(), shadow_change[16:].mean())
        self.assertGreater(highlight_change[16:].mean(), highlight_change[:16].mean())

    def test_contrast_and_black_white_points_move_in_expected_direction(self) -> None:
        levels = torch.tensor([0.03, 0.18, 0.75]).view(1, 1, 3, 1).expand(-1, 8, -1, 3)
        neutral = self.editor(levels, action(self.editor))
        contrast = self.editor(levels, action(self.editor, contrast=0.8))
        whites = self.editor(levels, action(self.editor, whites=0.8))
        blacks = self.editor(levels, action(self.editor, blacks=-0.8))
        self.assertLess(contrast[:, :, 0].mean(), neutral[:, :, 0].mean())
        self.assertGreater(contrast[:, :, 2].mean(), neutral[:, :, 2].mean())
        self.assertGreater(whites[:, :, 2].mean(), neutral[:, :, 2].mean())
        self.assertLess(blacks[:, :, 0].mean(), neutral[:, :, 0].mean())

    def test_saturation_minus_one_produces_neutral_colour(self) -> None:
        image = torch.tensor([[[[0.8, 0.25, 0.1], [0.1, 0.35, 0.9]]]])
        output = self.editor(image, action(self.editor, saturation=-1.0))
        channel_spread = output.amax(dim=-1) - output.amin(dim=-1)
        self.assertLess(channel_spread.max().item(), 2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_and_cuda_are_close(self) -> None:
        image = torch.rand(2, 64, 64, 3)
        parameters = torch.rand(2, 10) * 2.0 - 1.0
        cpu = self.editor(image, parameters)
        gpu_editor = PhotoEditor().cuda()
        gpu = gpu_editor(image.cuda(), parameters.cuda()).cpu()
        self.assertLess((cpu - gpu).abs().max().item(), 2e-4)


if __name__ == "__main__":
    unittest.main()
