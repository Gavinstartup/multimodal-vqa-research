"""Stage-1 projector 对齐预训练。

用法（从项目根目录运行，annotations 可以是 LLaVA-Pretrain 原生 conversations 格式，
也可以是简化的 {image, question, answer} 格式，见 src/data/dataset.py）：
    python -m src.train.train_stage1 \
        --image_dir data/images --annotations data/blip_laion_cc_sbu_558k.json

只有 multi_modal_projector 会被更新，vision_tower 和 language_model 全程冻结。
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor

from configs import stage1_config as cfg
from src.data import VQADataset, build_collate_fn
from src.model import VQAForConditionalGeneration
from src.utils import resolve_device, resolve_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage-1 projector alignment training")
    parser.add_argument("--vision_model", type=str, default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--text_model", type=str, default=cfg.ModelSettings.TEXT_MODEL_NAME)
    parser.add_argument("--image_dir", type=str, default=str(cfg.PathSettings.IMAGE_DIR))
    parser.add_argument("--annotations", type=str, default=str(cfg.PathSettings.ANNOTATIONS_JSON))
    parser.add_argument("--output_dir", type=str, default=str(cfg.PathSettings.OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=cfg.TrainingSettings.BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=cfg.TrainingSettings.EPOCHS)
    parser.add_argument("--lr", type=float, default=cfg.TrainingSettings.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=cfg.TrainingSettings.WEIGHT_DECAY)
    parser.add_argument("--max_length", type=int, default=cfg.TrainingSettings.MAX_LENGTH)
    parser.add_argument("--grad_accum_steps", type=int, default=cfg.TrainingSettings.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--log_interval", type=int, default=cfg.TrainingSettings.LOG_INTERVAL)
    parser.add_argument("--val_split", type=float, default=cfg.TrainingSettings.VAL_SPLIT)
    parser.add_argument("--early_stopping", type=int, default=cfg.TrainingSettings.EARLY_STOPPING_PATIENCE)
    parser.add_argument("--lr_patience", type=int, default=cfg.TrainingSettings.LR_PATIENCE)
    parser.add_argument("--lr_factor", type=float, default=cfg.TrainingSettings.LR_FACTOR)
    parser.add_argument("--num_workers", type=int, default=cfg.TrainingSettings.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=cfg.TrainingSettings.SEED)
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"],
        help="dtype for the frozen vision tower + LM + projector (Qwen3-8B in fp32 needs ~32GB RAM/VRAM)",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", default=True,
        help="Backward still has to flow through the whole frozen 36-layer LM to reach the "
             "projector, so activation memory is what actually determines VRAM headroom here "
             "— this trades ~20-30%% more compute for a large activation-memory cut.",
    )
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_epoch(model, loader, device, optimizer, grad_accum_steps, epoch, epochs, log_interval, writer, train):
    # Safe to put the whole tree (including the frozen vision_tower/language_model) in
    # train() mode: both checkpoints use dropout=0.0, so this has no numerical effect on
    # them, and train() is what lets gradient_checkpointing_enable() actually engage.
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
                    use_cache=False,  # training does one forward pass, no incremental decode
                )
                loss = outputs.loss

                if train:
                    (loss / grad_accum_steps).backward()
                    if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == len(loader) - 1:
                        optimizer.step()
                        optimizer.zero_grad()

                total_loss += loss.item()
                running_loss += loss.item()
                n_batches += 1

                if train and (batch_idx + 1) % log_interval == 0:
                    avg = running_loss / log_interval
                    pbar.set_postfix(loss=avg)
                    global_step = epoch * len(loader) + batch_idx
                    writer.add_scalar("Loss/train", avg, global_step)
                    running_loss = 0.0

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


def save_projector_checkpoint(model, path, extra=None):
    payload = {"projector_state_dict": model.multi_modal_projector.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logger.info("Saved projector checkpoint to %s", path)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device()
    dtype = resolve_dtype(device, args.dtype)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "runs"))

    logger.info("Loading tokenizer and image processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.ModelSettings.IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg.ModelSettings.IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(cfg.ModelSettings.IMAGE_TOKEN)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    logger.info("Building model from %s + %s ...", args.vision_model, args.text_model)
    model = VQAForConditionalGeneration.from_pretrained_components(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=args.text_model,
        image_token_id=image_token_id,
        torch_dtype=dtype,
        projector_hidden_act=cfg.ModelSettings.PROJECTOR_HIDDEN_ACT,
        vision_feature_layer=cfg.ModelSettings.VISION_FEATURE_LAYER,
        vision_feature_select_strategy=cfg.ModelSettings.VISION_FEATURE_SELECT_STRATEGY,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.config.text_config.vocab_size = len(tokenizer)
    model.freeze_for_stage1()
    model.to(device=device, dtype=dtype)
    # The projector is the only trainable module — keep it (and its AdamW optimizer
    # state) in fp32 so gradient updates aren't rounded away by bf16's 7-bit mantissa.
    # The frozen vision_tower/language_model stay in bf16 for memory; the projector's
    # forward() casts their bf16 output back up to fp32 before computing with it.
    model.multi_modal_projector.to(torch.float32)

    if args.gradient_checkpointing:
        # Backward still has to walk the entire frozen 36-layer LM to reach the
        # projector, so activation memory (not weight memory) is the real VRAM
        # bottleneck here — this is what makes batch_size>1 feasible on a single GPU.
        model.language_model.gradient_checkpointing_enable()
        model.vision_tower.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on vision_tower + language_model")

    num_image_tokens = model.config.num_image_tokens
    logger.info("Each image expands to %d patch tokens", num_image_tokens)

    logger.info("Building dataset from %s / %s ...", args.image_dir, args.annotations)
    full_dataset = VQADataset(
        image_dir=args.image_dir,
        annotations_path=args.annotations,
        image_processor=image_processor,
        tokenizer=tokenizer,
        num_image_tokens=num_image_tokens,
        image_token=cfg.ModelSettings.IMAGE_TOKEN,
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

    optimizer = torch.optim.AdamW(
        model.multi_modal_projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        run_epoch(
            model, train_loader, device, optimizer, args.grad_accum_steps,
            epoch, args.epochs, args.log_interval, writer, train=True,
        )
        val_loss = run_epoch(
            model, val_loader, device, optimizer, args.grad_accum_steps,
            epoch, args.epochs, args.log_interval, writer, train=False,
        )
        writer.add_scalar("Loss/val", val_loss, epoch)
        logger.info("Epoch %d val_loss=%.4f", epoch + 1, val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            save_projector_checkpoint(
                model, os.path.join(args.output_dir, "projector_best.pt"),
                extra={"epoch": epoch + 1, "val_loss": best_val_loss},
            )
        else:
            early_stopping_counter += 1
            logger.info(
                "Val loss did not improve. Early stopping counter: %d/%d",
                early_stopping_counter, args.early_stopping,
            )
            if early_stopping_counter >= args.early_stopping:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
                break

    # Whatever's in memory right now is just whichever epoch training happened to stop
    # on — early stopping's *timing* can be a bit off (noisy val_loss, coarse per-epoch
    # checks), so "last epoch" is not necessarily "best epoch". Reload the best-val-loss
    # checkpoint before writing out final_model/, so the deployed artifact is always the
    # best one seen, regardless of exactly when the loop stopped.
    best_ckpt_path = os.path.join(args.output_dir, "projector_best.pt")
    if os.path.isfile(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.multi_modal_projector.load_state_dict(best_ckpt["projector_state_dict"])
        logger.info(
            "Loaded best checkpoint (epoch %s, val_loss=%.4f) for final_model/",
            best_ckpt.get("epoch"), best_ckpt.get("val_loss", float("nan")),
        )

    save_projector_checkpoint(model, os.path.join(args.output_dir, "projector_final.pt"))

    final_model_dir = os.path.join(args.output_dir, "final_model")
    model.config.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    torch.save(model.multi_modal_projector.state_dict(), os.path.join(final_model_dir, "projector.pt"))
    logger.info("Training complete. Config + tokenizer + projector (best val_loss) saved to %s", final_model_dir)


if __name__ == "__main__":
    main()
