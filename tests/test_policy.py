from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT, SchemaError
from acp_single_pc_deploy.inference.policy import (
    ACPPolicyAdapter,
    _camera_view_from_checkpoint_name,
    contract_from_shape_meta,
)
from acp_single_pc_deploy.tests.test_common import make_observation


def make_shape_meta() -> dict:
    return {
        "id_list": [0],
        "action": {"shape": [19]},
        "obs": {
            "rgb_0": {"shape": [3, 224, 224], "type": "rgb"},
            "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
            "robot0_eef_rot_axis_angle": {"shape": [6], "type": "low_dim"},
            "robot0_eef_wrench": {"shape": [6], "type": "low_dim"},
        },
        "sample": {
            "obs": {
                "sparse": {
                    "rgb_0": {"horizon": 2, "down_sample_steps": 10},
                    "robot0_eef_pos": {"horizon": 3, "down_sample_steps": 5},
                    "robot0_eef_rot_axis_angle": {"horizon": 3, "down_sample_steps": 5},
                    "robot0_eef_wrench": {"horizon": 32, "down_sample_steps": 1},
                }
            },
            "action": {"sparse": {"horizon": 16, "down_sample_steps": 50}},
        },
    }


def test_contract_is_extracted_from_checkpoint_shape_meta() -> None:
    assert contract_from_shape_meta(make_shape_meta()) == EXPECTED_CONTRACT


def test_contract_rejects_nonmatching_wrench_stride() -> None:
    shape_meta = make_shape_meta()
    shape_meta["sample"]["obs"]["sparse"]["robot0_eef_wrench"]["down_sample_steps"] = 4
    with pytest.raises(SchemaError, match="wrench_stride"):
        contract_from_shape_meta(shape_meta)


def test_contract_rejects_wrong_rgb_shape() -> None:
    shape_meta = make_shape_meta()
    shape_meta["obs"]["rgb_0"]["shape"] = [3, 256, 256]
    with pytest.raises(SchemaError, match="rgb_0"):
        contract_from_shape_meta(shape_meta)


class FakePolicy:
    def predict_action(self, value):
        assert value == "prepared"
        return np.zeros((16, 19), dtype=np.float32)


def test_test_adapter_preserves_request_id_and_semantic_shapes() -> None:
    def prepare(packet):
        np.testing.assert_array_equal(packet.wrench, make_observation().wrench)
        return "prepared", "context"

    def postprocess(action, context):
        assert action.shape == (16, 19)
        assert context == "context"
        reference = np.zeros((16, 7), dtype=np.float64)
        virtual = np.zeros((16, 7), dtype=np.float64)
        reference[:, 3] = 1.0
        virtual[:, 3] = 1.0
        return reference, virtual, np.full(16, 500.0)

    adapter = ACPPolicyAdapter.for_test(
        policy=FakePolicy(),
        prepare_observation=prepare,
        extract_action=np.asarray,
        postprocess_action=postprocess,
        action_period_s=0.15,
    )
    chunk = adapter.infer(make_observation(request_id=17))
    assert chunk.request_id == 17
    assert chunk.reference_pose7.shape == (16, 7)
    assert chunk.virtual_pose7.shape == (16, 7)
    assert chunk.stiffness.shape == (16,)


def test_checkpoint_name_identifies_camera_view() -> None:
    assert _camera_view_from_checkpoint_name("conv_wrist_190hz") == "wrist"
    assert _camera_view_from_checkpoint_name("conv_main_190hz") == "main"
    with pytest.raises(SchemaError, match="does not identify"):
        _camera_view_from_checkpoint_name("conv_unknown")
