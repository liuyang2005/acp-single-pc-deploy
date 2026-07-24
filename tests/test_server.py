from __future__ import annotations

import numpy as np

from acp_single_pc_deploy.common.protocol import decode_message, encode_message
from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT
from acp_single_pc_deploy.inference.server import InferenceService
from acp_single_pc_deploy.tests.test_common import make_action, make_observation


class FakeAdapter:
    contract = EXPECTED_CONTRACT
    action_period_s = 0.15

    def infer(self, packet):
        return make_action(packet.request_id)


def test_handshake_exposes_checkpoint_and_contract() -> None:
    service = InferenceService(FakeAdapter(), checkpoint_sha256="abc123")
    metadata, arrays = decode_message(service.handle(encode_message({"type": "handshake"}, {})))
    assert arrays == {}
    assert metadata["type"] == "handshake_ok"
    assert metadata["checkpoint_sha256"] == "abc123"
    assert metadata["contract"] == EXPECTED_CONTRACT.to_dict()


def test_inference_response_preserves_request_id() -> None:
    service = InferenceService(FakeAdapter(), checkpoint_sha256="abc123")
    packet = make_observation(request_id=41)
    request = encode_message({"type": "infer", **packet.metadata()}, packet.arrays())
    metadata, arrays = decode_message(service.handle(request))
    assert metadata["type"] == "action"
    assert metadata["request_id"] == 41
    assert arrays["reference_pose7"].shape == (16, 7)


def test_malformed_request_returns_error_without_action_arrays() -> None:
    service = InferenceService(FakeAdapter(), checkpoint_sha256="abc123")
    metadata, arrays = decode_message(service.handle(encode_message({"type": "infer", "request_id": 1}, {})))
    assert metadata["type"] == "error"
    assert metadata["request_id"] == 1
    assert "error_type" in metadata
    assert arrays == {}


def test_unknown_message_returns_structured_error() -> None:
    service = InferenceService(FakeAdapter(), checkpoint_sha256="abc123")
    metadata, _ = decode_message(service.handle(encode_message({"type": "unknown"}, {})))
    assert metadata["type"] == "error"
    assert "unknown" in metadata["message"]
