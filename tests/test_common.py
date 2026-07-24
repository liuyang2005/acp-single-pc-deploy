from __future__ import annotations

import numpy as np
import pytest

from acp_single_pc_deploy.common.config import load_yaml_mapping, require_keys
from acp_single_pc_deploy.common.logging_utils import create_run_directory, to_jsonable
from acp_single_pc_deploy.common.protocol import ProtocolError, decode_message, encode_message
from acp_single_pc_deploy.common.schemas import (
    ActionChunk,
    EXPECTED_CONTRACT,
    ModelContract,
    ObservationPacket,
    SchemaError,
)


def make_observation(request_id: int = 3) -> ObservationPacket:
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    return ObservationPacket(
        request_id=request_id,
        rgb=np.zeros((2, 224, 224, 3), dtype=np.uint8),
        pose7=pose,
        wrench=np.arange(32 * 6, dtype=np.float64).reshape(32, 6),
        timestamps={
            "rgb": np.array([1.0, 1.1]),
            "pose": np.array([1.0, 1.05, 1.1]),
            "wrench": np.arange(32, dtype=np.float64) / 200.0 + 1.0,
        },
        latest_age_s={"rgb": 0.01, "pose": 0.01, "wrench": 0.005},
    )


def make_action(request_id: int = 3) -> ActionChunk:
    reference = np.zeros((16, 7), dtype=np.float64)
    virtual = np.zeros((16, 7), dtype=np.float64)
    reference[:, 3] = 1.0
    virtual[:, 3] = 1.0
    return ActionChunk(
        request_id=request_id,
        reference_pose7=reference,
        virtual_pose7=virtual,
        stiffness=np.full(16, 500.0),
        action_period_s=0.15,
        inference_latency_s=0.02,
    )


def test_expected_contract_is_fixed_to_trained_single_arm_model() -> None:
    assert EXPECTED_CONTRACT == ModelContract(1, 19, 2, 10, 3, 5, 32, 1, 16, 50)
    EXPECTED_CONTRACT.validate_expected()


def test_observation_preserves_raw_wrench() -> None:
    packet = make_observation()
    expected = packet.wrench.copy()
    packet.validate(EXPECTED_CONTRACT)
    np.testing.assert_array_equal(packet.wrench, expected)


def test_observation_rejects_non_normalized_quaternion() -> None:
    packet = make_observation()
    packet.pose7[0, 3] = 2.0
    with pytest.raises(SchemaError, match="quaternion"):
        packet.validate(EXPECTED_CONTRACT)


def test_action_chunk_validates_exact_shapes() -> None:
    chunk = make_action()
    chunk.validate(EXPECTED_CONTRACT)
    chunk.reference_pose7 = chunk.reference_pose7[:-1]
    with pytest.raises(SchemaError, match="reference_pose7"):
        chunk.validate(EXPECTED_CONTRACT)


def test_protocol_round_trip_preserves_dtype_and_shape() -> None:
    rgb = np.arange(24, dtype=np.uint8).reshape(2, 3, 4)
    wrench = np.arange(12, dtype=np.float64).reshape(2, 6)
    frames = encode_message({"type": "observation", "request_id": 7}, {"rgb": rgb, "wrench": wrench})
    metadata, arrays = decode_message(frames)
    assert metadata == {"type": "observation", "request_id": 7}
    np.testing.assert_array_equal(arrays["rgb"], rgb)
    np.testing.assert_array_equal(arrays["wrench"], wrench)
    assert arrays["rgb"].dtype == np.uint8
    assert arrays["wrench"].dtype == np.float64


def test_protocol_rejects_object_arrays_and_trailing_frames() -> None:
    with pytest.raises(ProtocolError, match="object"):
        encode_message({"type": "bad"}, {"bad": np.array([object()], dtype=object)})
    frames = encode_message({"type": "ok"}, {}) + [b"trailing"]
    with pytest.raises(ProtocolError, match="trailing"):
        decode_message(frames)


def test_yaml_loader_and_required_keys(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("robot:\n  serial: Rizon4s-063586\n", encoding="utf-8")
    config = load_yaml_mapping(path)
    require_keys(config, ("robot",), "config")
    with pytest.raises(KeyError, match="camera"):
        require_keys(config, ("robot", "camera"), "config")


def test_run_directories_are_non_overwriting(tmp_path) -> None:
    first = create_run_directory(tmp_path, "dry_run", timestamp="20260724T120000")
    assert first.is_dir()
    with pytest.raises(FileExistsError):
        create_run_directory(tmp_path, "dry_run", timestamp="20260724T120000")


def test_to_jsonable_handles_numpy_without_mutation() -> None:
    value = np.array([1.0, 2.0], dtype=np.float32)
    assert to_jsonable({"value": value}) == {"value": [1.0, 2.0]}
