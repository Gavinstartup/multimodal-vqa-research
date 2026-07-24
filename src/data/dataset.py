import json
import logging
import os

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VQADataset(Dataset):
    """Image-question-answer triples for Stage-1 projector alignment.

    Accepts two annotation schemas (auto-detected per item):

    1. LLaVA-Pretrain native format (LLaVA-CC3M-595K / blip_laion_cc_sbu_558k) —
       ``{"image": "...", "conversations": [{"from": "human", "value": "...<image>..."},
       {"from": "gpt", "value": "..."}]}``. This is the standard Stage-1 alignment
       dataset and the recommended default; see scripts/prepare_autodl.sh.
    2. A simpler custom schema — ``{"image": "...", "question": "...", "answer": "..."}``.

    Each sample's prompt is ``<image>`` (expanded to ``num_image_tokens`` placeholder
    tokens) + the question; loss is only computed on the answer tokens (question and
    image placeholders are masked with -100 in ``labels``).
    """

    def __init__(
        self,
        image_dir,
        annotations_path,
        image_processor,
        tokenizer,
        num_image_tokens,
        image_token="<image>",
        max_length=1024,
    ):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token = image_token
        self.max_length = max_length

        with open(annotations_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info("Loaded %d image-text pairs from %s", len(self.data), annotations_path)

        self._validate_data()

    @staticmethod
    def _has_valid_schema(item):
        if not isinstance(item, dict) or "image" not in item:
            return False
        if "conversations" in item:
            convs = item["conversations"]
            return (
                isinstance(convs, list)
                and len(convs) >= 2
                and all(isinstance(c, dict) and "from" in c and "value" in c for c in convs[:2])
            )
        return {"question", "answer"} <= item.keys()

    def _validate_data(self):
        if not isinstance(self.data, list):
            raise ValueError("Annotations file must contain a JSON list")

        valid_items = []
        for idx, item in enumerate(self.data):
            if not self._has_valid_schema(item):
                logger.warning("Item %d has an unrecognized/incomplete schema, skipping", idx)
                continue
            if not os.path.exists(os.path.join(self.image_dir, item["image"])):
                logger.warning("Image %s does not exist, skipping", item["image"])
                continue
            valid_items.append(item)

        if len(valid_items) < len(self.data):
            logger.info("Filtered out %d invalid items", len(self.data) - len(valid_items))
        self.data = valid_items

    def _extract_qa(self, item):
        """Returns (question, answer) regardless of which schema this item uses."""
        if "conversations" in item:
            # LLaVA-Pretrain stage-1 data is single-turn: conversations[0] is the human
            # turn (instruction + literal "<image>" marker somewhere in the text),
            # conversations[1] is the gpt turn (the caption). We place the image
            # placeholder block ourselves, so the literal marker is stripped here.
            question = item["conversations"][0]["value"].replace(self.image_token, "").strip()
            answer = item["conversations"][1]["value"].strip()
            return question, answer
        return item["question"].strip(), item["answer"].strip()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        question, answer = self._extract_qa(item)

        image_placeholder = self.image_token * self.num_image_tokens
        prompt = f"{image_placeholder}\n{question}\n"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids

        if len(input_ids) > self.max_length:
            logger.warning(
                "Sample %d (%s) has %d tokens, truncating answer to fit max_length=%d",
                idx, item["image"], len(input_ids), self.max_length,
            )
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }


def build_collate_fn(pad_token_id):
    def collate_fn(batch):
        lengths = [len(item["input_ids"]) for item in batch]
        max_len = max(lengths)

        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id
        )
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)

        # Built from real lengths (not `input_ids != pad_token_id`) since pad_token_id
        # commonly aliases eos_token_id, which also appears legitimately at sequence end.
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, length in enumerate(lengths):
            attention_mask[i, :length] = 1

        pixel_values = torch.stack([item["pixel_values"] for item in batch])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

    return collate_fn


__all__ = ["VQADataset", "build_collate_fn"]
