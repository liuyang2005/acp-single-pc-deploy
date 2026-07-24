from __future__ import annotations

import numpy as np

from acp_single_pc_deploy.robot.sensors import resize_rgb_for_policy, write_rgb_png


def test_wrist_rgb_is_center_cropped_to_policy_shape_without_channel_swap() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :80] = [255, 0, 0]
    image[:, 80:560] = [0, 255, 0]
    image[:, 560:] = [0, 0, 255]
    resized = resize_rgb_for_policy(image, width=224, height=224)
    assert resized.shape == (224, 224, 3)
    assert resized.dtype == np.uint8
    assert np.all(resized[112, 112] == [0, 255, 0])
    assert resized[..., 1].mean() > resized[..., 0].mean()
    assert resized[..., 1].mean() > resized[..., 2].mean()


def test_dry_run_png_preserves_rgb_color_order(tmp_path) -> None:
    import cv2

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = [255, 0, 0]
    path = tmp_path / "frame.png"
    write_rgb_png(path, image)
    loaded_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert np.all(loaded_bgr[0, 0] == [0, 0, 255])
