from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import logging
from multiprocessing.managers import BaseManager

import torch
import torch.nn as nn
import torch.optim as optim

from ltx_core.types import Audio

logger = logging.getLogger(__name__)


class RemoteVideoRewardManager(BaseManager):
    pass


@dataclass(frozen=True)
class BestOfNConfig:
    samples: int = 1
    server_port: int = 5001
    reward_key: str = "scores"
    align_key: str | None = None
    reward_weight: float = 1.0
    align_weight: float = 0.0
    topk_min: float = 0.4
    keep_candidates: bool = False
    aggregation_method: str = "weighted"
    arw_lr: float = 0.05
    arw_max_iter: int = 1
    arw_optimizer: str = "Adam"

    @property
    def enabled(self) -> bool:
        return self.samples > 1

    @property
    def metric_weights(self) -> dict[str, float]:
        weights = {self.reward_key: self.reward_weight}
        if self.align_key:
            weights[self.align_key] = self.align_weight
        return weights

    @property
    def metric_keys(self) -> list[str]:
        return list(self.metric_weights.keys())


def materialize_video_tensor(video: torch.Tensor | Iterator[torch.Tensor]) -> torch.Tensor:
    if isinstance(video, torch.Tensor):
        return video

    chunks = list(video)
    if not chunks:
        raise ValueError("Video iterator did not yield any frames.")
    if len(chunks) == 1:
        return chunks[0]
    return torch.cat(chunks, dim=0)


def compute_bon_score(score_values: dict[str, float], config: BestOfNConfig) -> tuple[float, float, float | None]:
    if config.reward_key not in score_values:
        available = ", ".join(sorted(score_values))
        raise ValueError(f"BON reward key '{config.reward_key}' was not returned by the server. Available keys: {available}")

    reward_score = score_values[config.reward_key]
    align_score = None

    if config.align_key:
        if config.align_key not in score_values:
            available = ", ".join(sorted(score_values))
            raise ValueError(
                f"BON align key '{config.align_key}' was not returned by the server. Available keys: {available}"
            )
        align_score = score_values[config.align_key]
        total_score = config.reward_weight * reward_score + config.align_weight * align_score
    else:
        total_score = reward_score

    return total_score, reward_score, align_score


class AdaptiveRewardWeighter(nn.Module):
    def __init__(self, reward_names: list[str], lr: float = 0.05, max_iter: int = 1, optimizer_type: str = "Adam"):
        super().__init__()
        self.max_iter = max_iter
        self.lr = lr
        self.optimizer_type = optimizer_type
        with torch.inference_mode(False):
            self.log_vars = nn.ParameterDict({key: nn.Parameter(torch.zeros(1)) for key in reward_names})
        self.history = {key: [] for key in reward_names}
        self.optimizer = self._build_optimizer()
        self.last_fit_stats: dict[str, object] = {}

    def _build_optimizer(self):
        optimizer_type = self.optimizer_type.lower()
        if optimizer_type == "adam":
            return optim.Adam(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "adamw":
            return optim.AdamW(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "rmsprop":
            return optim.RMSprop(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "sgd":
            return optim.SGD(self.log_vars.parameters(), lr=self.lr, momentum=0.9)
        if optimizer_type == "adagrad":
            return optim.Adagrad(self.log_vars.parameters(), lr=self.lr)
        if optimizer_type == "lbfgs":
            return optim.LBFGS(self.log_vars.parameters(), lr=self.lr, max_iter=5, history_size=10, line_search_fn="strong_wolfe")
        raise ValueError(f"Unsupported ARW optimizer_type: {self.optimizer_type}")

    def _build_loss_inputs(self, keys: list[str]) -> dict[str, torch.Tensor]:
        loss_inputs = {}
        for key in keys:
            if key not in self.history or not self.history[key]:
                continue
            data = torch.cat(self.history[key]).to(torch.float32)
            if data.numel() > 1:
                loss_inputs[key] = (data - data.mean()).pow(2) + 1e-6
        return loss_inputs

    @staticmethod
    def _compute_total_loss(
        log_vars: nn.ParameterDict, loss_inputs: dict[str, torch.Tensor], keys: list[str]
    ) -> torch.Tensor | None:
        total_loss = None
        for key in keys:
            if key not in log_vars or key not in loss_inputs:
                continue
            log_var = log_vars[key]
            precision = torch.exp(-log_var)
            loss_component = torch.mean(0.5 * precision * loss_inputs[key] + 0.5 * log_var)
            total_loss = loss_component if total_loss is None else total_loss + loss_component
        return total_loss

    def fit_sigma(self, score_dict: dict[str, torch.Tensor]) -> None:
        keys = list(score_dict.keys())
        with torch.inference_mode(False), torch.enable_grad():
            for key, score in score_dict.items():
                history_value = score.detach().clone().to(dtype=torch.float32, device="cpu")
                self.history.setdefault(key, []).append(history_value)

            if not keys:
                return

            total_len = sum(len(x) for x in self.history[keys[0]])
            if total_len <= 1:
                return

            loss_inputs = self._build_loss_inputs(keys)
            if not loss_inputs:
                return

            self.train()
            history_sizes = {key: sum(chunk.numel() for chunk in self.history.get(key, [])) for key in keys}
            latest_means = {key: float(score_dict[key].detach().mean().item()) for key in keys}
            logger.info("ARW fit start: history_sizes=%s latest_means=%s", history_sizes, latest_means)
            sigma_before = self.get_sigmas()
            loss_before_tensor = self._compute_total_loss(self.log_vars, loss_inputs, keys)
            loss_before = float(loss_before_tensor.detach().item()) if loss_before_tensor is not None else None

            for _ in range(self.max_iter):
                if self.optimizer_type.lower() == "lbfgs":

                    def closure():
                        self.optimizer.zero_grad()
                        total_loss = self._compute_total_loss(self.log_vars, loss_inputs, keys)
                        if total_loss is None:
                            raise RuntimeError("ARW LBFGS closure received no loss.")
                        total_loss.backward()
                        return total_loss

                    self.optimizer.step(closure)
                else:
                    self.optimizer.zero_grad()
                    total_loss = self._compute_total_loss(self.log_vars, loss_inputs, keys)
                    if total_loss is None:
                        return
                    total_loss.backward()
                    self.optimizer.step()

            loss_after_tensor = self._compute_total_loss(self.log_vars, loss_inputs, keys)
            loss_after = float(loss_after_tensor.detach().item()) if loss_after_tensor is not None else None
            sigma_after = self.get_sigmas()

        sigma_delta = {key: sigma_after[key] - sigma_before.get(key, 0.0) for key in sigma_after}
        self.last_fit_stats = {
            "history_sizes": history_sizes,
            "latest_means": latest_means,
            "loss_before": loss_before,
            "loss_after": loss_after,
            "loss_delta": None if loss_before is None or loss_after is None else loss_after - loss_before,
            "sigma_before": sigma_before,
            "sigma_after": sigma_after,
            "sigma_delta": sigma_delta,
        }
        logger.info(
            "ARW fit end: loss_before=%s loss_after=%s loss_delta=%s sigmas=%s sigma_delta=%s",
            self.last_fit_stats["loss_before"],
            self.last_fit_stats["loss_after"],
            self.last_fit_stats["loss_delta"],
            self.last_fit_stats["sigma_after"],
            self.last_fit_stats["sigma_delta"],
        )

    def get_weighted_score(self, score_dict: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
        self.eval()
        total = torch.zeros_like(next(iter(score_dict.values())), dtype=torch.float32)
        with torch.inference_mode(False), torch.no_grad():
            for key, score in score_dict.items():
                sigma = torch.exp(0.5 * self.log_vars[key]) if key in self.log_vars else torch.tensor(1.0)
                total = total + weights.get(key, 1.0) * (score.to(torch.float32) / (sigma + 1e-6))
        return total

    def get_sigmas(self) -> dict[str, float]:
        with torch.no_grad():
            return {key: float(torch.exp(0.5 * value).item()) for key, value in self.log_vars.items()}


class BonScoreAggregator:
    def __init__(self, config: BestOfNConfig):
        self.config = config
        self._weighter = (
            AdaptiveRewardWeighter(
                reward_names=config.metric_keys,
                lr=config.arw_lr,
                max_iter=config.arw_max_iter,
                optimizer_type=config.arw_optimizer,
            )
            if config.aggregation_method == "arw"
            else None
        )

    @staticmethod
    def _rank_normalize(values: torch.Tensor) -> torch.Tensor:
        if values.numel() <= 1:
            return torch.ones_like(values, dtype=torch.float32)
        order = torch.argsort(values, descending=False)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(values.numel(), dtype=torch.float32, device=values.device)
        return ranks / float(values.numel() - 1)

    @staticmethod
    def _minmax_normalize(values: torch.Tensor) -> torch.Tensor:
        min_value = values.min()
        max_value = values.max()
        denom = max_value - min_value
        if float(denom.abs().item()) < 1e-8:
            return torch.full_like(values, 0.5, dtype=torch.float32)
        return (values - min_value) / denom

    def aggregate(self, score_records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
        if not score_records:
            return score_records, {"method": self.config.aggregation_method}

        with torch.inference_mode(False):
            score_dict = {
                key: torch.tensor(
                    [float(record["raw_scores"][key]) for record in score_records],  # type: ignore[index]
                    dtype=torch.float32,
                ).clone()
                for key in self.config.metric_keys
            }
        raw_summary = {
            key: [float(value) for value in values.tolist()]
            for key, values in score_dict.items()
        }
        logger.info("BON raw score summary: %s", raw_summary)

        normalized_scores = None
        if self.config.aggregation_method == "weighted":
            total_scores = torch.zeros(len(score_records), dtype=torch.float32)
            for key, weight in self.config.metric_weights.items():
                total_scores = total_scores + weight * score_dict[key]
            details = {
                "method": "weighted",
                "weights": self.config.metric_weights,
            }
        elif self.config.aggregation_method == "rank":
            normalized_scores = {key: self._rank_normalize(values) for key, values in score_dict.items()}
            total_scores = torch.zeros(len(score_records), dtype=torch.float32)
            for key, weight in self.config.metric_weights.items():
                total_scores = total_scores + weight * normalized_scores[key]
            details = {
                "method": "rank",
                "weights": self.config.metric_weights,
            }
        elif self.config.aggregation_method == "minmax":
            normalized_scores = {key: self._minmax_normalize(values) for key, values in score_dict.items()}
            total_scores = torch.zeros(len(score_records), dtype=torch.float32)
            for key, weight in self.config.metric_weights.items():
                total_scores = total_scores + weight * normalized_scores[key]
            details = {
                "method": "minmax",
                "weights": self.config.metric_weights,
            }
        elif self.config.aggregation_method == "arw":
            if self._weighter is None:
                raise RuntimeError("ARW aggregator was not initialized.")
            self._weighter.fit_sigma(score_dict)
            total_scores = self._weighter.get_weighted_score(score_dict, self.config.metric_weights)
            details = {
                "method": "arw",
                "weights": self.config.metric_weights,
                "sigmas": self._weighter.get_sigmas(),
                "optimizer": self.config.arw_optimizer,
                "lr": self.config.arw_lr,
                "max_iter": self.config.arw_max_iter,
                "fit_stats": self._weighter.last_fit_stats,
            }
            history_summary = {
                key: {
                    "count": int(sum(chunk.numel() for chunk in self._weighter.history.get(key, []))),
                    "mean": float(torch.cat(self._weighter.history.get(key, [])).mean().item())
                    if self._weighter.history.get(key)
                    else None,
                }
                for key in self.config.metric_keys
            }
            logger.info("ARW history summary: %s", history_summary)
        else:
            raise ValueError(f"Unsupported BON aggregation method: {self.config.aggregation_method}")

        if self.config.aggregation_method == "arw" and self._weighter is not None:
            normalized_scores = {}
            for key, values in score_dict.items():
                sigma = self._weighter.get_sigmas().get(key, 1.0)
                normalized_scores[key] = values / (sigma + 1e-6)

        for idx, record in enumerate(score_records):
            record["total_score"] = float(total_scores[idx].item())
            record["reward_score"] = float(score_dict[self.config.reward_key][idx].item())
            record["align_score"] = (
                float(score_dict[self.config.align_key][idx].item()) if self.config.align_key else None
            )
            if normalized_scores is not None:
                record["normalized_scores"] = {
                    key: float(values[idx].item()) for key, values in normalized_scores.items()
                }

        logger.info("BON aggregated total scores: %s", [float(score) for score in total_scores.tolist()])
        if normalized_scores is not None:
            logger.info(
                "%s normalized scores: %s",
                self.config.aggregation_method.upper(),
                {key: [float(value) for value in values.tolist()] for key, values in normalized_scores.items()},
            )

        return score_records, details


class BonVideoRewardScorer:
    def __init__(self, server_port: int):
        RemoteVideoRewardManager.register("process_VideoReward")
        manager = RemoteVideoRewardManager(address=("localhost", server_port), authkey=b"secret")
        manager.connect()
        self._score_fn = manager.process_VideoReward

    def score(self, video: torch.Tensor, audio: Audio, prompt: str, topk_min: float) -> dict[str, float]:
        prepared_video = _prepare_video_for_reward_model(video)
        response = self._score_fn(
            {
                "videos": prepared_video.numpy(),
                "audios": _prepare_audio_waveform(audio).numpy(),
                "text": prompt,
                "topk_min": topk_min,
                "arw": 0.0,
            }
        )
        return {key: _extract_scalar(value) for key, value in response.items()}


def _prepare_audio_waveform(audio: Audio) -> torch.Tensor:
    waveform = audio.waveform.detach().cpu()

    if waveform.ndim == 2:
        if waveform.shape[-1] <= 8:
            waveform = waveform.to(torch.float32).mean(dim=-1)
        elif waveform.shape[0] <= 8:
            waveform = waveform.to(torch.float32).mean(dim=0)
        else:
            waveform = waveform.reshape(-1)

    if waveform.dtype != torch.int16:
        waveform = torch.clamp(waveform, -1.0, 1.0)
        waveform = (waveform * 32767.0).to(torch.int16)

    return waveform.contiguous()


def _prepare_video_for_reward_model(video: torch.Tensor, target_frames: int = 16) -> torch.Tensor:
    frames = video.detach().cpu()

    if frames.ndim != 4:
        raise ValueError(f"Expected decoded video with shape [T, H, W, C], got {tuple(frames.shape)}")

    num_frames = frames.shape[0]
    if num_frames == target_frames:
        return frames.contiguous()

    if num_frames < target_frames:
        repeat_factor = (target_frames + num_frames - 1) // num_frames
        frames = frames.repeat((repeat_factor, 1, 1, 1))[:target_frames]
        return frames.contiguous()

    indices = torch.linspace(0, num_frames - 1, target_frames, dtype=torch.float32)
    indices = torch.round(indices).to(torch.long)
    return frames.index_select(0, indices).contiguous()


def _extract_scalar(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("Expected a non-empty score list from BON scorer.")
        return _extract_scalar(value[0])
    return float(value)
