"""End-to-end inference test on ONE case (the official repo test case).

This is the golden test: it compares our pipeline with the reference
outputs shipped by the model authors with those exact inputs.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from src.inference import run_inference


def test_single_case_inference(device, loaded_models):
    from src.data_loader import load_case
    case = load_case("10032_1000032")
    models, _ = loaded_models
    result = run_inference(case, device=device, models=models)

    # output shapes
    assert result.preprocessed_shape == (1, 3, 20, 256, 256)
    assert result.raw_softmax_heatmap.shape == (20, 256, 256)
    t2w_shape = sitk.GetArrayFromImage(case.images["t2w"]).shape
    assert result.detection_map.shape == t2w_shape

    # probability in [0, 1]
    assert 0.0 <= result.case_level_likelihood <= 1.0


def test_golden_reference_reproduction(device, loaded_models):
    """Reproduce the authors' reference outputs on the official test case."""
    from src.data_loader import load_case
    case = load_case("10032_1000032")
    ref_lik = case.clinical["reference_case_level_likelihood"]
    ref_map = sitk.GetArrayFromImage(case.clinical["reference_detection_map"])

    models, _ = loaded_models
    result = run_inference(case, device=device, models=models)

    # case-level likelihood must match to 1e-3 (official tolerance)
    assert abs(result.case_level_likelihood - ref_lik) < 1e-3, (
        f"likelihood differs: ours {result.case_level_likelihood} "
        f"vs reference {ref_lik}"
    )

    # detection map: allowed platform-level differences are tiny —
    # at most a small fraction of edge voxels beyond the 1e-3 tolerance
    diff = np.abs(result.detection_map - ref_map)
    frac_different = float((diff > 1e-3).sum()) / diff.size
    assert frac_different < 1e-3, (
        f"{(diff > 1e-3).sum()} voxels differ beyond 1e-3 "
        f"({frac_different:.3%})"
    )
    # and the maximum value must match closely
    assert np.abs(result.detection_map.max() - ref_map.max()) < 1e-3
