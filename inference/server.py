from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path
from typing import Any

from acp_single_pc_deploy.common.config import load_yaml_mapping, require_keys
from acp_single_pc_deploy.common.protocol import decode_message, encode_message
from acp_single_pc_deploy.common.schemas import ObservationPacket
from acp_single_pc_deploy.inference.policy import ACPPolicyAdapter, file_sha256, resolve_checkpoint_path


class InferenceService:
    def __init__(self, adapter: ACPPolicyAdapter, checkpoint_sha256: str) -> None:
        self.adapter = adapter
        self.checkpoint_sha256 = checkpoint_sha256

    def handle(self, frames: list[bytes]) -> list[bytes]:
        request_id: int | None = None
        try:
            metadata, arrays = decode_message(frames)
            message_type = metadata.get("type")
            if "request_id" in metadata:
                request_id = int(metadata["request_id"])
            if message_type in {"handshake", "health"}:
                return encode_message(
                    {
                        "type": "handshake_ok" if message_type == "handshake" else "health_ok",
                        "checkpoint_sha256": self.checkpoint_sha256,
                        "contract": self.adapter.contract.to_dict(),
                        "action_period_s": self.adapter.action_period_s,
                        "checkpoint_name": self.adapter.checkpoint_name,
                        "checkpoint_epoch": self.adapter.checkpoint_epoch,
                        "checkpoint_camera_view": self.adapter.checkpoint_camera_view,
                        "color_order": "RGB",
                        "image_shape": [224, 224, 3],
                    },
                    {},
                )
            if message_type != "infer":
                raise ValueError(f"unknown message type: {message_type!r}")
            packet = ObservationPacket.from_wire(metadata, arrays)
            chunk = self.adapter.infer(packet)
            return encode_message({"type": "action", **chunk.metadata()}, chunk.arrays())
        except Exception as exc:
            return encode_message(
                {
                    "type": "error",
                    "request_id": request_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                {},
            )


def _validate_loopback_endpoint(endpoint: str) -> None:
    if not endpoint.startswith("tcp://127.0.0.1:"):
        raise ValueError("inference service must bind to tcp://127.0.0.1:<port>")


def run_server(service: InferenceService, endpoint: str, stop_event: threading.Event) -> None:
    import zmq

    _validate_loopback_endpoint(endpoint)
    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.bind(endpoint)
    try:
        while not stop_event.is_set():
            try:
                frames = socket.recv_multipart()
            except zmq.Again:
                continue
            socket.send_multipart(service.handle(frames))
    finally:
        socket.close(linger=0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local ACP checkpoint inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bind", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_yaml_mapping(args.config)
        require_keys(config, ("network", "model"), "inference config")
        network = config["network"]
        model = config["model"]
        endpoint = args.bind or network["bind_endpoint"]
        _validate_loopback_endpoint(endpoint)
        checkpoint = resolve_checkpoint_path(args.checkpoint)
        adapter = ACPPolicyAdapter.load(
            acp_root=model["acp_root"],
            checkpoint=checkpoint,
            device=model["device"],
            action_period_s=float(model["action_period_s"]),
            inference_seed=int(model["inference_seed"]),
            expected_camera_view=str(model["expected_camera_view"]),
            minimum_checkpoint_epoch=int(model["minimum_checkpoint_epoch"]),
        )
        service = InferenceService(adapter, file_sha256(checkpoint))
        stop_event = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop_event.set())
        signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
        logging.info("ACP contract: %s", adapter.contract.to_dict())
        logging.info("checkpoint sha256: %s", service.checkpoint_sha256)
        logging.info("listening on %s", endpoint)
        run_server(service, endpoint, stop_event)
        return 0
    except Exception:
        logging.exception("inference server failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
