import csv
import json
import logging
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, default_1_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.bon import BestOfNConfig, BonScoreAggregator, BonVideoRewardScorer, materialize_video_tensor
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()


def load_prompt_items(path: str) -> list[tuple[str, str]]:
    prompt_path = Path(path)
    if prompt_path.suffix.lower() == ".csv":
        return load_prompt_items_from_csv(prompt_path)

    items: list[tuple[str, str]] = []
    with prompt_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            items.append((parts[0], parts[1]))
    return items


def serialize_prompt_encoding(prompt_encoding) -> dict[str, Any]:
    return {
        "video_encoding": prompt_encoding.video_encoding.detach().to(device="cpu"),
        "audio_encoding": None
        if prompt_encoding.audio_encoding is None
        else prompt_encoding.audio_encoding.detach().to(device="cpu"),
        "attention_mask": prompt_encoding.attention_mask.detach().to(device="cpu"),
    }


def load_prompt_encoding(path: str | Path, device: torch.device, dtype: torch.dtype):
    payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dict in prompt embedding file: {path}")
    if "video_encoding" not in payload or "attention_mask" not in payload:
        raise ValueError(f"Prompt embedding file is missing required keys: {path}")

    video_encoding = payload["video_encoding"].to(device=device, dtype=dtype)
    audio_encoding = payload.get("audio_encoding")
    if audio_encoding is not None:
        audio_encoding = audio_encoding.to(device=device, dtype=dtype)
    attention_mask = payload["attention_mask"].to(device=device)
    return EmbeddingsProcessorOutput(
        video_encoding=video_encoding,
        audio_encoding=audio_encoding,
        attention_mask=attention_mask,
    )


def resolve_prompt_encoding_path(prompt_embeds_dir: Path | None, video_name: str) -> Path | None:
    if prompt_embeds_dir is None:
        return None
    return prompt_embeds_dir / f"{Path(video_name).stem}.pt"


def load_prompt_items_from_csv(path: Path) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    prompt_field_candidates = ("text", "prompt", "caption", "video_text", "audio_text")

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV prompt file has no header: {path}")

        sample_index = 0
        for row_index, row in enumerate(reader, start=2):
            prompt = next((row.get(field, "").strip() for field in prompt_field_candidates if row.get(field, "").strip()), "")

            if not prompt:
                continue

            output_path = Path(f"sample_{sample_index:04d}.mp4")
            items.append((str(output_path), prompt))
            sample_index += 1

    return items


def shard_items(items: list[tuple[str, str]], num_shards: int, shard_index: int) -> list[tuple[str, str]]:
    if num_shards <= 0:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}")
    return [item for i, item in enumerate(items) if i % num_shards == shard_index]


def encode_prompt_with_models(prompt: str, text_encoder, embeddings_processor):
    hidden_states, mask = text_encoder.encode(prompt)
    return embeddings_processor.process_hidden_states(hidden_states, mask)


def generate_candidate_with_reuse(  # noqa: PLR0913
    *,
    pipeline: "TI2VidOneStagePipeline",
    prompt: str,
    seed: int,
    args,
    text_encoder,
    embeddings_processor,
    video_encoder,
    transformer,
    video_decoder,
    audio_decoder,
    vocoder,
    v_context_n: torch.Tensor,
    a_context_n: torch.Tensor,
    video_guider_params: MultiModalGuiderParams,
    audio_guider_params: MultiModalGuiderParams,
    prompt_encoding: EmbeddingsProcessorOutput | None = None,
) -> tuple[Iterator[torch.Tensor], Audio]:
    dtype = pipeline.dtype
    model_device = pipeline.device
    generator = torch.Generator(device=model_device).manual_seed(seed)
    noiser = GaussianNoiser(generator=generator)
    stepper = EulerDiffusionStep()
    if prompt_encoding is None:
        prompt_encoding = encode_prompt_with_models(prompt, text_encoder, embeddings_processor)
    v_context_p = prompt_encoding.video_encoding
    a_context_p = prompt_encoding.audio_encoding

    output_shape = VideoPixelShape(
        batch=1,
        frames=args.num_frames,
        width=args.width,
        height=args.height,
        fps=args.frame_rate,
    )
    conditionings = combined_image_conditionings(
        images=args.images,
        height=output_shape.height,
        width=output_shape.width,
        video_encoder=video_encoder,
        dtype=dtype,
        device=model_device,
    )
    sigmas = LTX2Scheduler().execute(steps=args.num_inference_steps).to(dtype=torch.float32, device=model_device)

    video_guider_factory = create_multimodal_guider_factory(
        params=video_guider_params,
        negative_context=v_context_n,
    )
    audio_guider_factory = create_multimodal_guider_factory(
        params=audio_guider_params,
        negative_context=a_context_n,
    )

    def denoising_loop(
        loop_sigmas: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        loop_stepper: DiffusionStepProtocol,
    ) -> tuple[LatentState, LatentState]:
        return euler_denoising_loop(
            sigmas=loop_sigmas,
            video_state=video_state,
            audio_state=audio_state,
            stepper=loop_stepper,
            denoise_fn=multi_modal_guider_factory_denoising_func(
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
                v_context=v_context_p,
                a_context=a_context_p,
                transformer=transformer,
            ),
        )

    video_state, audio_state = denoise_audio_video(
        output_shape=output_shape,
        conditionings=conditionings,
        noiser=noiser,
        sigmas=sigmas,
        stepper=stepper,
        denoising_loop_fn=denoising_loop,
        components=pipeline.pipeline_components,
        dtype=dtype,
        device=model_device,
    )

    video = vae_decode_video(video_state.latent, video_decoder, generator=generator)
    audio = vae_decode_audio(audio_state.latent, audio_decoder, vocoder)

    del prompt_encoding
    del v_context_p
    del a_context_p
    del sigmas
    del conditionings
    del video_state
    del audio_state

    return video, audio


def generate_and_save_best_of_n(
    *,
    prompt: str,
    output_path: Path,
    base_seed: int,
    frame_rate: float,
    generate_candidate: Callable[[int], tuple[Iterator[torch.Tensor], Audio]],
    best_of_n: BestOfNConfig,
    scorer: BonVideoRewardScorer | None,
    aggregator: BonScoreAggregator | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not best_of_n.enabled:
        video, audio = generate_candidate(base_seed)
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=str(output_path),
            video_chunks_number=1,
        )
        return

    if scorer is None:
        raise ValueError("--bon-samples > 1 requires a running BON scorer.")
    if aggregator is None:
        raise ValueError("--bon-samples > 1 requires a BON score aggregator.")

    candidate_dir = output_path.parent / ".bon_candidates" / output_path.stem
    candidate_dir.mkdir(parents=True, exist_ok=True)
    score_records: list[dict[str, object]] = []

    for candidate_idx in range(best_of_n.samples):
        candidate_seed = base_seed + candidate_idx
        candidate_path = candidate_dir / f"{output_path.stem}.cand{candidate_idx:02d}.seed{candidate_seed}{output_path.suffix}"

        video_iter, audio = generate_candidate(candidate_seed)
        video_tensor = materialize_video_tensor(video_iter).to("cpu")
        audio_cpu = audio.to(device="cpu")

        score_values = scorer.score(video_tensor, audio_cpu, prompt, topk_min=best_of_n.topk_min)

        encode_video(
            video=video_tensor,
            fps=frame_rate,
            audio=audio_cpu,
            output_path=str(candidate_path),
            video_chunks_number=1,
        )

        record = {
            "candidate_index": candidate_idx,
            "seed": candidate_seed,
            "candidate_path": str(candidate_path),
            "raw_scores": score_values,
        }
        score_records.append(record)

        del video_tensor
        del audio
        del audio_cpu
        cleanup_memory()

    score_records, aggregation_details = aggregator.aggregate(score_records)
    for record in score_records:
        log_suffix = ""
        align_score = record.get("align_score")
        if best_of_n.align_key and align_score is not None:
            log_suffix = f" {best_of_n.align_key}={align_score:.4f}"
        logging.info(
            "BON candidate %d/%d for %s: seed=%d total=%.4f %s=%.4f%s",
            int(record["candidate_index"]) + 1,
            best_of_n.samples,
            output_path.name,
            int(record["seed"]),
            float(record["total_score"]),
            best_of_n.reward_key,
            float(record["reward_score"]),
            log_suffix,
        )

    best_record = max(score_records, key=lambda record: float(record["total_score"]))
    logging.info(
        "BON selected candidate: idx=%d seed=%d total=%.4f reward=%.4f align=%s",
        int(best_record["candidate_index"]),
        int(best_record["seed"]),
        float(best_record["total_score"]),
        float(best_record["reward_score"]),
        "None" if best_record.get("align_score") is None else f"{float(best_record['align_score']):.4f}",
    )


    metadata_path = output_path.with_suffix(".bon.json")
    metadata = {
        "selected": best_record,
        "scoring": {
            "reward_key": best_of_n.reward_key,
            "align_key": best_of_n.align_key,
            "reward_weight": best_of_n.reward_weight,
            "align_weight": best_of_n.align_weight,
            "topk_min": best_of_n.topk_min,
            "samples": best_of_n.samples,
            "aggregation_method": best_of_n.aggregation_method,
        },
        "aggregation": aggregation_details,
        "candidates": score_records,
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    best_candidate_path = Path(str(best_record["candidate_path"]))
    if best_of_n.keep_candidates:
        shutil.copy2(best_candidate_path, output_path)
    else:
        shutil.move(best_candidate_path, output_path)
        for record in score_records:
            candidate_path = Path(str(record["candidate_path"]))
            if candidate_path.exists():
                candidate_path.unlink()
        try:
            candidate_dir.rmdir()
        except OSError:
            pass

    logging.info("BON selected seed=%s total=%.4f -> %s", best_record["seed"], best_record["total_score"], output_path)


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    Assumes full non distilled model is provided in the checkpoint_path.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        enhance_prompt: bool = False,
        prompt_encoding: EmbeddingsProcessorOutput | None = None,
        negative_prompt_encoding: EmbeddingsProcessorOutput | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        if prompt_encoding is None or negative_prompt_encoding is None:
            ctx_p, ctx_n = encode_prompts(
                [prompt, negative_prompt],
                self.model_ledger,
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
            prompt_encoding = ctx_p
            negative_prompt_encoding = ctx_n
        else:
            ctx_p, ctx_n = prompt_encoding, negative_prompt_encoding
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode image conditionings with the VAE encoder, then free it
        # before loading the transformer to reduce peak VRAM.
        stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        video_encoder = self.model_ledger.video_encoder()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        del video_encoder
        cleanup_memory()

        transformer = self.model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
            )

        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), generator=generator)
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_1_stage_arg_parser(params=params)
    for action in parser._actions:
        if "--prompt" in action.option_strings or "--output-path" in action.option_strings:
            action.required = False
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Path to a prompt file. Supports text lines of VIDEO_FILENAME PROMPT_TEXT or CSV with prompt/name columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated videos when using --prompt-file.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in --output-dir when using --prompt-file.",
    )
    parser.add_argument(
        "--batch-model-mode",
        choices=("reload", "reuse"),
        default="reload",
        help=(
            "Batch prompt-file inference mode: "
            "'reload' matches the original behavior and loads/frees models per prompt, "
            "'reuse' keeps models loaded across the whole batch for faster inference but higher VRAM use."
        ),
    )
    parser.add_argument(
        "--prompt-embeds-path",
        type=str,
        default=None,
        help="Path to a precomputed prompt embedding .pt file for single-prompt inference.",
    )
    parser.add_argument(
        "--negative-prompt-embeds-path",
        type=str,
        default=None,
        help="Path to a precomputed negative prompt embedding .pt file.",
    )
    parser.add_argument(
        "--prompt-embeds-dir",
        type=str,
        default=None,
        help="Directory containing precomputed prompt embeddings for --prompt-file runs. Files are resolved as <video_name_stem>.pt.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split prompt-file items across this many shards. Use with --shard-index.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="0-based shard index to run when using --num-shards.",
    )
    parser.add_argument(
        "--bon-samples",
        type=int,
        default=1,
        help="Generate this many candidates per prompt and keep the highest-scoring one.",
    )
    parser.add_argument(
        "--bon-server-port",
        type=int,
        default=5001,
        help="Local port for the BON VideoReward/VQA server.",
    )
    parser.add_argument(
        "--bon-reward-key",
        type=str,
        default="scores",
        help="Primary score key returned by the BON server.",
    )
    parser.add_argument(
        "--bon-align-key",
        type=str,
        default=None,
        help="Optional secondary score key returned by the BON server, for example JS.",
    )
    parser.add_argument(
        "--bon-reward-weight",
        type=float,
        default=1.0,
        help="Weight applied to --bon-reward-key when combining BON scores.",
    )
    parser.add_argument(
        "--bon-align-weight",
        type=float,
        default=0.0,
        help="Weight applied to --bon-align-key when combining BON scores.",
    )
    parser.add_argument(
        "--bon-topk-min",
        type=float,
        default=0.4,
        help="topk_min forwarded to the BON server for JavisScore calculation.",
    )
    parser.add_argument(
        "--bon-keep-candidates",
        action="store_true",
        help="Keep all encoded BON candidates under .bon_candidates instead of deleting non-selected ones.",
    )
    parser.add_argument(
        "--bon-aggregation-method",
        choices=("weighted", "rank", "minmax", "arw"),
        default="weighted",
        help="How to aggregate BON candidate scores across reward metrics.",
    )
    parser.add_argument(
        "--bon-arw-lr",
        type=float,
        default=0.05,
        help="Learning rate for ARW score aggregation.",
    )
    parser.add_argument(
        "--bon-arw-max-iter",
        type=int,
        default=1,
        help="Inner optimization steps for ARW score aggregation.",
    )
    parser.add_argument(
        "--bon-arw-optimizer",
        type=str,
        default="Adam",
        help="Optimizer for ARW score aggregation.",
    )
    args = parser.parse_args()
    bon_align_key = args.bon_align_key
    if bon_align_key in ("", "None", "none", "null", "NULL"):
        bon_align_key = None
    best_of_n = BestOfNConfig(
        samples=args.bon_samples,
        server_port=args.bon_server_port,
        reward_key=args.bon_reward_key,
        align_key=bon_align_key,
        reward_weight=args.bon_reward_weight,
        align_weight=args.bon_align_weight,
        topk_min=args.bon_topk_min,
        keep_candidates=args.bon_keep_candidates,
        aggregation_method=args.bon_aggregation_method,
        arw_lr=args.bon_arw_lr,
        arw_max_iter=args.bon_arw_max_iter,
        arw_optimizer=args.bon_arw_optimizer,
    )
    scorer = BonVideoRewardScorer(best_of_n.server_port) if best_of_n.enabled else None
    if best_of_n.enabled:
        with torch.inference_mode(False):
            aggregator = BonScoreAggregator(best_of_n)
    else:
        aggregator = None
    if best_of_n.enabled:
        logging.info(
            "Best-of-N enabled: samples=%d reward=%s align=%s aggregation=%s",
            best_of_n.samples,
            best_of_n.reward_key,
            best_of_n.align_key or "<none>",
            best_of_n.aggregation_method,
        )
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_guidance_scale,
        stg_scale=args.video_stg_guidance_scale,
        rescale_scale=args.video_rescale_scale,
        modality_scale=args.a2v_guidance_scale,
        skip_step=args.video_skip_step,
        stg_blocks=args.video_stg_blocks,
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_guidance_scale,
        stg_scale=args.audio_stg_guidance_scale,
        rescale_scale=args.audio_rescale_scale,
        modality_scale=args.v2a_guidance_scale,
        skip_step=args.audio_skip_step,
        stg_blocks=args.audio_stg_blocks,
    )

    if args.prompt_file:
        if not args.output_dir:
            raise ValueError("--output-dir is required when using --prompt-file")
        if args.prompt_embeds_path:
            raise ValueError("--prompt-embeds-path cannot be used with --prompt-file; use --prompt-embeds-dir instead")

        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_embeds_dir = Path(args.prompt_embeds_dir).expanduser().resolve() if args.prompt_embeds_dir else None
        negative_prompt_encoding = (
            load_prompt_encoding(args.negative_prompt_embeds_path, pipeline.device, pipeline.dtype)
            if args.negative_prompt_embeds_path
            else None
        )

        items = load_prompt_items(args.prompt_file)
        items = shard_items(items, args.num_shards, args.shard_index)
        logging.info(
            "Batch sharding: shard %d/%d, %d items selected",
            args.shard_index,
            args.num_shards,
            len(items),
        )
        if args.batch_model_mode == "reuse":
            logging.info("Batch model mode: reuse")

            text_encoder = None
            embeddings_processor = None
            if prompt_embeds_dir is None:
                text_encoder = pipeline.model_ledger.text_encoder()
                embeddings_processor = pipeline.model_ledger.gemma_embeddings_processor()
            video_encoder = pipeline.model_ledger.video_encoder()
            transformer = pipeline.model_ledger.transformer()
            video_decoder = pipeline.model_ledger.video_decoder()
            audio_decoder = pipeline.model_ledger.audio_decoder()
            vocoder = pipeline.model_ledger.vocoder()

            negative_encoding = negative_prompt_encoding
            if negative_encoding is None:
                negative_encoding = encode_prompt_with_models(args.negative_prompt, text_encoder, embeddings_processor)
            v_context_n = negative_encoding.video_encoding
            a_context_n = negative_encoding.audio_encoding

            for idx, (video_name, prompt) in enumerate(items, start=1):
                output_path = output_dir / video_name
                if args.skip_existing and output_path.exists():
                    logging.info("[%d/%d] Skipping existing file: %s", idx, len(items), output_path)
                    continue

                logging.info("[%d/%d] Generating %s", idx, len(items), output_path.name)
                prompt_encoding_path = resolve_prompt_encoding_path(prompt_embeds_dir, video_name)
                if prompt_encoding_path is not None and not prompt_encoding_path.exists():
                    logging.info("[%d/%d] Skipping missing prompt embedding: %s", idx, len(items), prompt_encoding_path)
                    continue
                prompt_encoding = (
                    load_prompt_encoding(prompt_encoding_path, pipeline.device, pipeline.dtype)
                    if prompt_encoding_path is not None
                    else None
                )
                generate_and_save_best_of_n(
                    prompt=prompt,
                    output_path=output_path,
                    base_seed=args.seed,
                    frame_rate=args.frame_rate,
                    generate_candidate=lambda seed: generate_candidate_with_reuse(
                        pipeline=pipeline,
                        prompt=prompt,
                        seed=seed,
                        args=args,
                        text_encoder=text_encoder,
                        embeddings_processor=embeddings_processor,
                        video_encoder=video_encoder,
                        transformer=transformer,
                        video_decoder=video_decoder,
                        audio_decoder=audio_decoder,
                        vocoder=vocoder,
                        v_context_n=v_context_n,
                        a_context_n=a_context_n,
                        video_guider_params=video_guider_params,
                        audio_guider_params=audio_guider_params,
                        prompt_encoding=prompt_encoding,
                    ),
                    best_of_n=best_of_n,
                    scorer=scorer,
                    aggregator=aggregator,
                )

            del negative_encoding
            if text_encoder is not None:
                del text_encoder
            if embeddings_processor is not None:
                del embeddings_processor
            del video_encoder
            del transformer
            del video_decoder
            del audio_decoder
            del vocoder
            cleanup_memory()
        else:
            logging.info("Batch model mode: reload")
            for idx, (video_name, prompt) in enumerate(items, start=1):
                output_path = output_dir / video_name
                if args.skip_existing and output_path.exists():
                    logging.info("[%d/%d] Skipping existing file: %s", idx, len(items), output_path)
                    continue

                logging.info("[%d/%d] Generating %s", idx, len(items), output_path.name)
                prompt_encoding_path = resolve_prompt_encoding_path(prompt_embeds_dir, video_name)
                if prompt_encoding_path is not None and not prompt_encoding_path.exists():
                    logging.info("[%d/%d] Skipping missing prompt embedding: %s", idx, len(items), prompt_encoding_path)
                    continue
                prompt_encoding = (
                    load_prompt_encoding(prompt_encoding_path, pipeline.device, pipeline.dtype)
                    if prompt_encoding_path is not None
                    else None
                )
                generate_and_save_best_of_n(
                    prompt=prompt,
                    output_path=output_path,
                    base_seed=args.seed,
                    frame_rate=args.frame_rate,
                    generate_candidate=lambda seed: pipeline(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        seed=seed,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,
                        frame_rate=args.frame_rate,
                        num_inference_steps=args.num_inference_steps,
                        video_guider_params=video_guider_params,
                        audio_guider_params=audio_guider_params,
                        images=args.images,
                        prompt_encoding=prompt_encoding,
                        negative_prompt_encoding=negative_prompt_encoding,
                    ),
                    best_of_n=best_of_n,
                    scorer=scorer,
                    aggregator=aggregator,
                )
    else:
        if not args.prompt or not args.output_path:
            raise ValueError("Either provide --prompt and --output-path, or use --prompt-file and --output-dir")
        prompt_encoding = (
            load_prompt_encoding(args.prompt_embeds_path, pipeline.device, pipeline.dtype) if args.prompt_embeds_path else None
        )
        negative_prompt_encoding = (
            load_prompt_encoding(args.negative_prompt_embeds_path, pipeline.device, pipeline.dtype)
            if args.negative_prompt_embeds_path
            else None
        )

        generate_and_save_best_of_n(
            prompt=args.prompt,
            output_path=Path(args.output_path),
            base_seed=args.seed,
            frame_rate=args.frame_rate,
            generate_candidate=lambda seed: pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                frame_rate=args.frame_rate,
                num_inference_steps=args.num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=args.images,
                prompt_encoding=prompt_encoding,
                negative_prompt_encoding=negative_prompt_encoding,
            ),
            best_of_n=best_of_n,
            scorer=scorer,
            aggregator=aggregator,
        )


if __name__ == "__main__":
    main()
