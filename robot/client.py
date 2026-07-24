from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any

from acp_single_pc_deploy.common.protocol import decode_message, encode_message
from acp_single_pc_deploy.common.schemas import (
    ActionChunk,
    EXPECTED_CONTRACT,
    ObservationPacket,
    SchemaError,
)


class InferenceTimeout(TimeoutError):
    pass


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str) or not endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("inference client must connect to tcp://127.0.0.1:<port>")


class InferenceClient:
    def __init__(self, endpoint: str, timeout_s: float) -> None:
        _validate_endpoint(endpoint)
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.timeout_s = float(timeout_s)
        self._transport: Callable[[list[bytes]], list[bytes]] | None = None
        self._closed = False

    @classmethod
    def for_test(
        cls,
        transport: Callable[[list[bytes]], list[bytes]],
        timeout_s: float,
    ) -> InferenceClient:
        client = cls("tcp://127.0.0.1:1", timeout_s)
        client._transport = transport
        return client

    def _request(self, metadata: dict[str, Any], arrays: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("inference client is closed")
        frames = encode_message(metadata, arrays)
        if self._transport is not None:
            return decode_message(self._transport(frames))

        import zmq

        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        timeout_ms = max(1, int(round(self.timeout_s * 1000.0)))
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send_multipart(frames)
            response = socket.recv_multipart()
        except zmq.Again as exc:
            raise InferenceTimeout(
                f"inference request timed out after {self.timeout_s:.3f}s"
            ) from exc
        finally:
            socket.close(linger=0)
        return decode_message(response)

    @staticmethod
    def _raise_remote_error(metadata: dict[str, Any]) -> None:
        if metadata.get("type") == "error":
            error_type = metadata.get("error_type", "RemoteError")
            message = metadata.get("message", "inference service returned an error")
            raise RuntimeError(f"{error_type}: {message}")

    def handshake(self) -> dict[str, Any]:
        metadata, arrays = self._request({"type": "handshake"}, {})
        self._raise_remote_error(metadata)
        if arrays or metadata.get("type") != "handshake_ok":
            raise RuntimeError("invalid inference handshake response")
        self._validate_contract_metadata(metadata)
        return metadata

    @staticmethod
    def _validate_contract_metadata(metadata: dict[str, Any]) -> None:
        try:
            contract = metadata["contract"]
            if contract != EXPECTED_CONTRACT.to_dict():
                raise SchemaError(f"inference contract mismatch: {contract!r}")
            if metadata["color_order"] != "RGB" or metadata["image_shape"] != [224, 224, 3]:
                raise SchemaError("inference image contract mismatch")
        except KeyError as exc:
            raise SchemaError(f"inference handshake missing {exc.args[0]}") from exc

    def health(self) -> dict[str, Any]:
        metadata, arrays = self._request({"type": "health"}, {})
        self._raise_remote_error(metadata)
        if arrays or metadata.get("type") != "health_ok":
            raise RuntimeError("invalid inference health response")
        self._validate_contract_metadata(metadata)
        return metadata

    def infer(self, packet: ObservationPacket) -> ActionChunk:
        packet.validate(EXPECTED_CONTRACT)
        metadata, arrays = self._request(
            {"type": "infer", **packet.metadata()},
            packet.arrays(),
        )
        self._raise_remote_error(metadata)
        if metadata.get("type") != "action":
            raise RuntimeError(f"expected action response, got {metadata.get('type')!r}")
        chunk = ActionChunk.from_wire(metadata, arrays)
        chunk.validate(EXPECTED_CONTRACT)
        if chunk.request_id != packet.request_id:
            raise RuntimeError(
                f"response request ID {chunk.request_id} does not match {packet.request_id}"
            )
        return chunk

    def close(self) -> None:
        self._closed = True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the local ACP inference service")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--health", action="store_true", required=True)
    args = parser.parse_args(argv)
    client = InferenceClient(args.endpoint, args.timeout)
    try:
        client.health()
        return 0
    except Exception as exc:
        print(f"inference health check failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
