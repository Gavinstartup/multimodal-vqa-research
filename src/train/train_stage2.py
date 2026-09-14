"""Stage-2 指令微调：在 Stage-1 projector 权重基础上，用 LLaVA-Instruct-150K 风格的
多轮指令数据继续训练，projector 全量更新 + language_model 用 LoRA 微调。

用法（从项目根目录运行）：
    python -m src.train.train_stage2 \
        --image_dir data/images --annotations data/llava_instruct_150k.json \
        --stage1_checkpoint outputs/stage1_projector/final_model

Stage-1 checkpoint 目录需要包含 config.json + tokenizer + projector.pt
（train_stage1.py 的 final_model/ 输出即可直接用）。
"""

import argparse
import logging
import math
import os
import random

import numpy as np
import torch
from peft import LoraConfig, TaskType
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, get_cosine_schedule_with_warmup

from configs import stage2_config as cfg
from src.model import VQAConfig, VQAForConditionalGeneration
from src.data import VQAInstructDataset, build_collate_fn
from src.utils import resolve_device, resolve_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-2 LoRA instruction fine-tuning")
    parser.add_argument("--vision_model", type=str, default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--stage1_checkpoint", type=str, default=str(cfg.PathSettings.STAGE1_CHECKPOINT))
    parser.add_argument("--image_dir", type=str, default=str(cfg.PathSettings.IMAGE_DIR))
    parser.add_argument("--annotations", type=str, default=str(cfg.PathSettings.ANNOTATIONS_JSON))
    parser.add_argument("--output_dir", type=str, default=str(cfg.PathSettings.OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=cfg.TrainingSettings.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.TrainingSettings.EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.TrainingSettings.LEARNING_RATE)
    parser.add_argument("--projector_lr", type=float, default=cfg.TrainingSettings.PROJECTOR_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.TrainingSettings.WEIGHT_DECAY)
    parser.add_argument("--max_length", type=int, default=cfg.TrainingSettings.MAX_LENGTH)
    parser.add_argument("--grad_accum_steps", type=int, default=cfg.TrainingSettings.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--log_interval", type=int, default=cfg.TrainingSettings.LOG_INTERVAL)
    parser.add_argument("--lora_r", type=int, default=cfg.LoraSettings.R)
    parser.add_argument("--lora_alpha", type=int, default=cfg.LoraSettings.ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=cfg.LoraSettings.DROPOUT)
    parser.add_argument(
        "--save_every_steps", type=int, default=2000,
        help="Save rolling projector_latest.pt + lora_latest/ every N steps within an "
             "epoch, so a long run can resume progress if interrupted. 0 disables it.",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Directory containing a projector_latest.pt/projector_best.pt and a "
             "lora_latest/lora_best adapter dir from an interrupted stage-2 run. "
             "Optimizer/scheduler state and epoch position are not restored.",
    )
    parser.add_argument("--val_split", type=float, default=cfg.TrainingSettings.VAL_SPLIT)
    parser.add_argument("--early_stopping", type=int, default=cfg.TrainingSettings.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--warmup_ratio", type=float, default=cfg.TrainingSettings.WARMUP_RATIO)
    parser.add_argument("--num_workers", type=int, default=cfg.TrainingSettings.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=cfg.TrainingSettings.SEED)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
        help="dtype for the frozen vision tower + LoRA-wrapped LM base weights",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(model, output_dir, tag, extra=None):
    """Saves the (small) trainable state — projector full weights + LoRA adapter —
    under a `tag` (e.g. "latest"/"best"/"final"), skipping the frozen multi-GB base
    vision tower / LM weights entirely."""
    projector_payload = {"projector_state_dict": model.multi_modal_projector.state_dict()}
    if extra:
        projector_payload.update(extra)
    torch.save(projector_payload, os.path.join(output_dir, f"projector_{tag}.pt"))
    model.language_model.save_pretrained(os.path.join(output_dir, f"lora_{tag}"))
    logger.info("Saved %s checkpoint (projector + LoRA adapter) to %s", tag, output_dir)


def run_epoch(
    model, loader, device, optimizer, grad_accum_steps, epoch, epochs, log_interval, writer, train,
    save_every_steps=0, output_dir=None, scheduler=None,
):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    running_loss = 0.0

    if train:
        optimizer.zero_grad()

    desc = f"Epoch {epoch + 1}/{epochs} [{'train' if train else 'val'}]"
    with torch.set_grad_enabled(train):
        with tqdm(loader, desc=desc) as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss

                if train:
                    (loss / grad_accum_steps).backward()
                    if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(loader) - 1:
                        optimizer.step()
                        # Per optimizer step, not per epoch: a single-epoch run gets no
                        # decay at all from an epoch-level scheduler.
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad()

                total_loss += loss.item()
                running_loss += loss.item()
                n_batches += 1

                global_step = epoch * len(loader) + batch_idx

                if train and (batch_idx + 1) % log_interval == 0:
                    avg = running_loss / log_interval
                    pbar.set_postfix(loss=avg)
                    writer.add_scalar("Loss/train", avg, global_step)
                    writer.add_scalar("LR/lora", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("LR/projector", optimizer.param_groups[1]["lr"], global_step)
                    running_loss = 0.0

                if train and save_every_steps and (batch_idx + 1) % save_every_steps == 0:
                    save_checkpoint(model, output_dir, "latest", extra={"epoch": epoch + 1, "step": batch_idx + 1})

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device()
    dtype = resolve_dtype(device, args.dtype)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    logger.info("Loading config + tokenizer from Stage-1 checkpoint %s ...", args.stage1_checkpoint)
    stage1_config = VQAConfig.from_pretrained(args.stage1_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.stage1_checkpoint)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    text_model_name = stage1_config.text_config.name_or_path
    logger.info("Building model from %s + %s ...", args.vision_model, text_model_name)
    model = VQAForConditionalGeneration.from_pretrained_components(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=text_model_name,
        image_token_id=stage1_config.image_token_id,
        torch_dtype=dtype,
        projector_hidden_act=stage1_config.projector_hidden_act,
        vision_feature_layer=stage1_config.vision_feature_layer,
        vision_feature_select_strategy=stage1_config.vision_feature_select_strategy,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.text_config.vocab_size = len(tokenizer)

    projector_path = os.path.join(args.stage1_checkpoint, "projector.pt")
    logger.info("Loading Stage-1 projector weights from %s ...", projector_path)
    projector_state = torch.load(projector_path, map_location="cpu")
    if "projector_state_dict" in projector_state:
        projector_state = projector_state["projector_state_dict"]
    model.multi_modal_projector.load_state_dict(projector_state)

    model.to(device=device, dtype=dtype)
    # Trained in fp32 for the same reason as Stage-1: bf16's 7-bit mantissa would round
    # away small gradient updates to an already-aligned projector.
    model.multi_modal_projector.to(torch.float32)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=cfg.LoraSettings.TARGET_MODULES,
    )
    model.prepare_for_stage2(lora_config)
    model.language_model.print_trainable_parameters()

    if args.resume_from:
        resume_projector = os.path.join(args.resume_from, "projector_latest.pt")
        if not os.path.isfile(resume_projector):
            resume_projector = os.path.join(args.resume_from, "projector_best.pt")
        resume_ckpt = torch.load(resume_projector, map_location=device)
        model.multi_modal_projector.load_state_dict(resume_ckpt.get("projector_state_dict", resume_ckpt))

        resume_lora = os.path.join(args.resume_from, "lora_latest")
        if not os.path.isdir(resume_lora):
            resume_lora = os.path.join(args.resume_from, "lora_best")
        # "default" already exists (get_peft_model just created it), so peft takes the
        # load-weights-only path here and ignores is_trainable — harmless, since that
        # existing adapter is already trainable. Renaming the adapter would break this.
        model.language_model.load_adapter(resume_lora, adapter_name="default", is_trainable=True)
        logger.info("Resumed projector + LoRA weights from %s", args.resume_from)

    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable()
        model.language_model.enable_input_require_grads()
        model.vision_tower.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on vision_tower + language_model")

    num_image_tokens = model.config.num_image_tokens
    logger.info("Each image expands to %d patch tokens", num_image_tokens)

    logger.info("Building dataset from %s / %s ...", args.image_dir, args.annotations)
    full_dataset = VQAInstructDataset(
        image_dir=args.image_dir,
        annotations_path=args.annotations,
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_image_tokens=num_image_tokens,
        image_token="<image>",
        max_length=args.max_length,
    )

    val_size = max(1, int(args.val_split * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logger.info("Train samples: %d, Val samples: %d", train_size, val_size)

    collate_fn = build_collate_fn(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # Separate param groups: LoRA adapters use the LM learning rate, the (still fp32,
    # fully trainable) projector uses its own — Stage-1 already put it in a good
    # basin, so it typically wants a gentler LR than a from-scratch LoRA adapter.
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.language_model.parameters() if p.requires_grad], "lr": args.lr},
            {"params": model.multi_modal_projector.parameters(), "lr": args.projector_lr},
        ],
        weight_decay=args.weight_decay,
    )
    # ceil, matching run_epoch's "step on the last batch even if it doesn't fill an
    # accumulation window" behaviour.
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    logger.info(
        "LR schedule: cosine over %d optimizer steps (%d warmup), lora_lr=%g projector_lr=%g",
        total_steps, warmup_steps, args.lr, args.projector_lr,
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        run_epoch(
            model, train_loader, device, optimizer, args.grad_accum_steps,
            epoch, args.epochs, args.log_interval, writer, train=True,
            save_every_steps=args.save_every_steps, output_dir=args.output_dir, scheduler=scheduler,
        )
        val_loss = run_epoch(
            model, val_loader, device, optimizer, args.grad_accum_steps,
            epoch, args.epochs, args.log_interval, writer, train=False,
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        logger.info("Epoch %d val_loss=%.4f", epoch + 1, val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_checkpoint(model, args.output_dir, "best", extra={"epoch": epoch + 1, "val_loss": best_val_loss})
        else:
            early_stopping_counter += 1
            logger.info(
                "Val loss did not improve. Early stopping counter: %d/%d",
                early_stopping_counter, args.early_stopping,
            )
            if early_stopping_counter >= args.early_stopping:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                break

    best_projector_path = os.path.join(args.output_dir, "projector_best.pt")
    best_lora_path = os.path.join(args.output_dir, "lora_best")
    if os.path.isfile(best_projector_path):
        best_ckpt = torch.load(best_projector_path, map_location=device)
        model.multi_modal_projector.load_state_dict(best_ckpt["projector_state_dict"])
        model.language_model.load_adapter(best_lora_path, adapter_name="default")
        logger.info(
            "Loaded best checkpoint (epoch %s, val_loss=%.4f) for final_model/",
            best_ckpt.get("epoch"), best_ckpt.get("val_loss", float("nan")),
        )

    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.config.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    torch.save(model.multi_modal_projector.state_dict(), os.path.join(final_model_dir, "projector.pt"))
    model.language_model.save_pretrained(final_model_dir)
    logger.info(
        "Training complete. Config + tokenizer + projector.pt + LoRA adapter (best val_loss) saved to %s",
        final_model_dir,
    )


if __name__ == "__main__":
    main()
