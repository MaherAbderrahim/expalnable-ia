"""Test the data loading (no network needed: uses the official repo's test case)."""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def test_official_case_loads(official_case):
    assert official_case.case_id == "10032_1000032"
    for mod in ("t2w", "adc", "hbv"):
        assert mod in official_case.images
        img = official_case.images[mod]
        # 3D volumes with physical spacing
        assert img.GetDimension() == 3
        assert all(s > 0 for s in img.GetSpacing())


def test_official_case_modalities_shapes(official_case):
    t2w = sitk.GetArrayFromImage(official_case.images["t2w"])
    adc = sitk.GetArrayFromImage(official_case.images["adc"])
    hbv = sitk.GetArrayFromImage(official_case.images["hbv"])
    # all three are 3D volumes, axial orientation (z = slices)
    assert t2w.ndim == 3 and adc.ndim == 3 and hbv.ndim == 3
    # physical spacing present and sane
    for img in (official_case.images["t2w"], official_case.images["adc"], official_case.images["hbv"]):
        assert all(0.1 < s < 10.0 for s in img.GetSpacing())
    assert int(t2w.max()) > 0


def test_reference_outputs_are_present(official_case):
    """The shipped reference outputs must be readable (LFS integrity check)."""
    assert "reference_case_level_likelihood" in official_case.clinical
    ref_map = official_case.clinical.get("reference_detection_map")
    assert ref_map is not None
    # real LFS content, not a 134-byte pointer: has genuine shape
    assert ref_map.GetSize()[0] == 384


def test_weights_are_real_not_lfs_pointers():
    """AI models often break silently here: 134-byte Git LFS pointers."""
    from src.config import WEIGHTS_DIR
    for fold in range(5):
        p = WEIGHTS_DIR / f"unet_F{fold}.pt"
        assert p.exists(), f"missing checkpoint {p} (run 'git lfs pull')"
        assert p.stat().st_size == 127229381, (
            f"{p.name} is {p.stat().st_size} bytes — an LFS pointer is ~134 bytes; "
            "run 'git lfs pull' to replace it with the real weights."
        )
