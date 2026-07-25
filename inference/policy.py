from __future__ import annotations

import hashlib
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acp_single_pc_deploy.common.schemas import (
    ActionChunk,
    EXPECTED_CONTRACT,
    ModelContract,
    ObservationPacket,
    SchemaError,
)


def resolve_checkpoint_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        resolved = resolved / "checkpoints" / "latest.ckpt"
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_acp_import_paths(acp_root: str | Path) -> Path:
    root = Path(acp_root).expanduser().resolve()
    required = (root / "PyriteUtility", root / "PyriteConfig", root / "PyriteML")
    if not root.is_dir() or any(not path.is_dir() for path in required):
        raise FileNotFoundError(f"ACP root is missing required Pyrite packages: {root}")
    for entry in (root, root / "PyriteML"):
        text = str(entry)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"shape_meta {name} must be a mapping")
    return value


def _shape(entry: Mapping[str, Any], expected: tuple[int, ...], name: str) -> None:
    try:
        actual = tuple(int(item) for item in entry["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"{name} shape is invalid") from exc
    if actual != expected:
        raise SchemaError(f"{name} shape must be {expected}, got {actual}")


def _sample(sample: Mapping[str, Any], key: str) -> tuple[int, int]:
    try:
        entry = _mapping(sample[key], key)
        return int(entry["horizon"]), int(entry["down_sample_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"missing or invalid sample contract for {key}") from exc


def contract_from_shape_meta(shape_meta: Mapping[str, Any]) -> ModelContract:
    root = _mapping(shape_meta, "root")
    if tuple(root.get("id_list", ())) != (0,):
        raise SchemaError(f"id_list must be [0], got {root.get('id_list')!r}")
    action = _mapping(root.get("action"), "action")
    _shape(action, (19,), "action")
    obs = _mapping(root.get("obs"), "obs")
    required_shapes = {
        "rgb_0": (3, 224, 224),
        "robot0_eef_pos": (3,),
        "robot0_eef_rot_axis_angle": (6,),
        "robot0_eef_wrench": (6,),
    }
    for key, expected in required_shapes.items():
        if key not in obs:
            raise SchemaError(f"shape_meta missing observation {key}")
        _shape(_mapping(obs[key], key), expected, key)
    sample = _mapping(root.get("sample"), "sample")
    sample_obs = _mapping(_mapping(sample.get("obs"), "sample.obs").get("sparse"), "sample.obs.sparse")
    rgb_horizon, rgb_stride = _sample(sample_obs, "rgb_0")
    pose_horizon, pose_stride = _sample(sample_obs, "robot0_eef_pos")
    rot_horizon, rot_stride = _sample(sample_obs, "robot0_eef_rot_axis_angle")
    wrench_horizon, wrench_stride = _sample(sample_obs, "robot0_eef_wrench")
    if (rot_horizon, rot_stride) != (pose_horizon, pose_stride):
        raise SchemaError("pose rotation horizon/stride must match pose position")
    sample_action = _mapping(_mapping(sample.get("action"), "sample.action").get("sparse"), "sample.action.sparse")
    try:
        action_horizon = int(sample_action["horizon"])
        action_stride = int(sample_action["down_sample_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError("invalid action sample contract") from exc
    contract = ModelContract(
        1,
        19,
        rgb_horizon,
        rgb_stride,
        pose_horizon,
        pose_stride,
        wrench_horizon,
        wrench_stride,
        action_horizon,
        action_stride,
    )
    contract.validate_expected()
    return contract


def _camera_view_from_checkpoint_name(name: str) -> str:
    normalized = name.lower()
    if "wrist" in normalized:
        return "wrist"
    if "main" in normalized:
        return "main"
    raise SchemaError(
        f"checkpoint name {name!r} does not identify a wrist or main camera model"
    )


@dataclass(frozen=True)
class ACPPolicyAdapter:
    contract: ModelContract
    action_period_s: float
    checkpoint_name: str
    checkpoint_epoch: int
    checkpoint_camera_view: str
    _policy: Any
    _prepare_observation: Callable[[ObservationPacket], tuple[Any, Any]]
    _extract_action: Callable[[Any], np.ndarray]
    _postprocess_action: Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray, np.ndarray]]
    _inference_context: Callable[[], AbstractContextManager[Any]]
    _seed_inference: Callable[[], None]

    @classmethod
    def for_test(
        cls,
        policy: Any,
        prepare_observation: Callable[[ObservationPacket], tuple[Any, Any]],
        extract_action: Callable[[Any], np.ndarray],
        postprocess_action: Callable[[np.ndarray, Any], tuple[np.ndarray, np.ndarray, np.ndarray]],
        action_period_s: float,
        checkpoint_name: str = "test_wrist",
        checkpoint_epoch: int = 999,
        checkpoint_camera_view: str = "wrist",
    ) -> ACPPolicyAdapter:
        return cls(
            contract=EXPECTED_CONTRACT,
            action_period_s=float(action_period_s),
            checkpoint_name=checkpoint_name,
            checkpoint_epoch=int(checkpoint_epoch),
            checkpoint_camera_view=checkpoint_camera_view,
            _policy=policy,
            _prepare_observation=prepare_observation,
            _extract_action=extract_action,
            _postprocess_action=postprocess_action,
            _inference_context=nullcontext,
            _seed_inference=lambda: None,
        )

    @classmethod
    def load(
        cls,
        acp_root: str | Path,
        checkpoint: str | Path,
        device: str,
        action_period_s: float,
        inference_seed: int,
        expected_camera_view: str,
        minimum_checkpoint_epoch: int,
    ) -> ACPPolicyAdapter:
        add_acp_import_paths(acp_root)
        checkpoint_path = resolve_checkpoint_path(checkpoint)
        if action_period_s <= 0.0:
            raise ValueError("action_period_s must be positive")
        if expected_camera_view not in {"wrist", "main"}:
            raise ValueError("expected_camera_view must be wrist or main")
        if minimum_checkpoint_epoch < 0:
            raise ValueError("minimum_checkpoint_epoch must be nonnegative")

        import dill
        import hydra
        import torch
        from PyriteConfig.tasks.common.common_type_conversions import (
            action19_postprocess,
            sparse_obs_to_obs_sample,
        )
        from PyriteML.diffusion_policy.workspace.base_workspace import BaseWorkspace
        from PyriteUtility.spatial_math import spatial_utilities as su

        payload = torch.load(
            checkpoint_path.open("rb"), map_location="cpu", pickle_module=dill
        )
        cfg = payload["cfg"]
        checkpoint_name = str(cfg.name)
        checkpoint_camera_view = _camera_view_from_checkpoint_name(checkpoint_name)
        checkpoint_epoch = int(dill.loads(payload["pickles"]["epoch"]))
        if checkpoint_camera_view != expected_camera_view:
            raise SchemaError(
                f"checkpoint camera view is {checkpoint_camera_view}, expected {expected_camera_view}"
            )
        if checkpoint_epoch < minimum_checkpoint_epoch:
            raise SchemaError(
                f"checkpoint epoch {checkpoint_epoch} is below required minimum "
                f"{minimum_checkpoint_epoch}"
            )
        workspace_cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = workspace_cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.ema_model if cfg.training.use_ema else workspace.model
        policy.num_inference_steps = cfg.policy.num_inference_steps
        policy.eval().to(device)
        policy.reset()
        shape_meta = cfg.task.shape_meta
        contract = contract_from_shape_meta(shape_meta)

        def prepare(packet: ObservationPacket) -> tuple[Any, Any]:
            pose9 = su.SE3_to_pose9(su.pose7_to_SE3(packet.pose7))
            sparse = {
                "rgb_0": packet.rgb,
                "robot0_eef_pos": pose9[:, :3],
                "robot0_eef_rot_axis_angle": pose9[:, 3:],
                "robot0_eef_wrench": packet.wrench,
            }
            processed, base_se3 = sparse_obs_to_obs_sample(
                obs_sparse=sparse,
                shape_meta=shape_meta,
                reshape_mode="reshape",
                id_list=[0],
                ignore_rgb=False,
            )
            tensors = {
                key: torch.from_numpy(np.expand_dims(value, axis=0)).to(device)
                for key, value in processed.items()
            }
            return {"sparse": tensors}, base_se3

        def extract(result: Any) -> np.ndarray:
            if not isinstance(result, Mapping) or "sparse" not in result:
                raise SchemaError("policy output must contain sparse action")
            action = result["sparse"]
            if hasattr(action, "detach"):
                action = action.detach().to("cpu").numpy()
            array = np.asarray(action)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            return array

        def postprocess(action: np.ndarray, base_se3: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            reference_se3, virtual_se3, stiffness = action19_postprocess(action, base_se3, [0])
            reference = su.SE3_to_pose7(np.asarray(reference_se3[0]))
            virtual = su.SE3_to_pose7(np.asarray(virtual_se3[0]))
            return reference, virtual, np.asarray(stiffness[0], dtype=np.float64)

        def seed_inference() -> None:
            torch.manual_seed(int(inference_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(inference_seed))

        return cls(
            contract=contract,
            action_period_s=float(action_period_s),
            checkpoint_name=checkpoint_name,
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_camera_view=checkpoint_camera_view,
            _policy=policy,
            _prepare_observation=prepare,
            _extract_action=extract,
            _postprocess_action=postprocess,
            _inference_context=torch.inference_mode,
            _seed_inference=seed_inference,
        )

    def infer(self, packet: ObservationPacket) -> ActionChunk:
        packet.validate(self.contract)
        started = time.perf_counter()
        policy_input, context = self._prepare_observation(packet)
        self._seed_inference()
        with self._inference_context():
            result = self._policy.predict_action(policy_input)
        action = np.asarray(self._extract_action(result), dtype=np.float64)
        if action.shape != (self.contract.action_horizon, self.contract.action_dim):
            raise SchemaError(
                f"policy action must have shape {(self.contract.action_horizon, self.contract.action_dim)}, got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise SchemaError("policy action contains non-finite values")
        reference, virtual, stiffness = self._postprocess_action(action, context)
        chunk = ActionChunk(
            request_id=packet.request_id,
            reference_pose7=np.asarray(reference, dtype=np.float64),
            virtual_pose7=np.asarray(virtual, dtype=np.float64),
            stiffness=np.asarray(stiffness, dtype=np.float64),
            action_period_s=self.action_period_s,
            inference_latency_s=time.perf_counter() - started,
        )
        chunk.validate(self.contract)
        return chunk
