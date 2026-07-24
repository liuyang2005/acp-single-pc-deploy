from __future__ import annotations

import socket
import threading

import numpy as np
import pytest

from acp_single_pc_deploy.common.schemas import EXPECTED_CONTRACT
from acp_single_pc_deploy.inference.policy import ACPPolicyAdapter
from acp_single_pc_deploy.inference.server import InferenceService, run_server
from acp_single_pc_deploy.robot.client import InferenceClient, InferenceTimeout
from acp_single_pc_deploy.tests.test_common import make_observation


def _free_endpoint() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"tcp://127.0.0.1:{port}"


def _adapter() -> ACPPolicyAdapter:
    reference = np.tile([0, 0, 0, 1, 0, 0, 0], (16, 1)).astype(float)
    virtual = reference.copy()
    stiffness = np.full(16, 500.0)
    policy = type("Policy", (), {"predict_action": lambda self, _obs: {"sparse": np.zeros((16, 19))}})()
    return ACPPolicyAdapter.for_test(
        policy=policy,
        prepare_observation=lambda packet: (packet, None),
        extract_action=lambda result: result["sparse"],
        postprocess_action=lambda action, context: (reference, virtual, stiffness),
        action_period_s=0.15,
    )


def test_loopback_client_handshake_and_infer_round_trip() -> None:
    endpoint = _free_endpoint()
    stop = threading.Event()
    thread = threading.Thread(
        target=run_server,
        args=(InferenceService(_adapter(), "b" * 64), endpoint, stop),
        daemon=True,
    )
    thread.start()
    client = InferenceClient(endpoint, timeout_s=1.0)
    try:
        handshake = client.handshake()
        assert handshake["contract"] == EXPECTED_CONTRACT.to_dict()
        action = client.infer(make_observation(request_id=7))
        assert action.request_id == 7
        assert action.reference_pose7.shape == (16, 7)
    finally:
        client.close()
        stop.set()
        thread.join(timeout=2.0)
    assert not thread.is_alive()


def test_timeout_does_not_return_a_stale_action() -> None:
    client = InferenceClient(_free_endpoint(), timeout_s=0.02)
    try:
        with pytest.raises(InferenceTimeout):
            client.infer(make_observation(request_id=1))
    finally:
        client.close()


def test_client_rejects_old_response_request_id() -> None:
    client = InferenceClient.for_test(lambda frames: _wrong_id_response(frames), timeout_s=1.0)
    with pytest.raises(RuntimeError, match="request ID"):
        client.infer(make_observation(request_id=9))


def _wrong_id_response(frames: list[bytes]) -> list[bytes]:
    from acp_single_pc_deploy.common.protocol import decode_message, encode_message
    from acp_single_pc_deploy.tests.test_common import make_action

    metadata, _ = decode_message(frames)
    chunk = make_action(request_id=int(metadata["request_id"]) - 1)
    return encode_message({"type": "action", **chunk.metadata()}, chunk.arrays())
