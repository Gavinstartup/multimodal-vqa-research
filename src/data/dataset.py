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


class VQAInstructDataset(Dataset):
    """Multi-turn image-instruction conversations for Stage-2 LoRA fine-tuning.

    Expects the LLaVA-Instruct-150K native schema — ``{"image": "...", "conversations":
    [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}, ...]}`` — with an
    arbitrary (even) number of alternating human/gpt turns, unlike VQADataset which only
    reads the first pair. The literal ``<image>`` marker is expected somewhere in the
    first human turn only (LLaVA-Instruct convention); it is replaced with the expanded
    placeholder block.

    Turns are rendered as ``USER: ...\\nASSISTANT: ...`` (Vicuna-style, matching the plain
    prompt format already used by generate.py/serve_repl.py) and loss is computed only on
    each ASSISTANT span, so the model learns to answer the specific question asked rather
    than only ever emitting a generic caption for the "USER: <image>\\n..." prefix.
    """

    def __init__(
        self,
        image_dir,
        annotations_path,
        image_processor,
        tokenizer,
        num_image_tokens,
        image_token="<image>",
        max_length=2048,
    ):
        self.image_dir = image_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.num_image_tokens = num_image_tokens
        self.image_token = image_token
        self.max_length = max_length

        if max_length <= num_image_tokens:
            raise ValueError(
                f"max_length={max_length} leaves no room for text after the {num_image_tokens} "
                "image tokens every sample starts with"
            )

        with open(annotations_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info("Loaded %d conversations from %s", len(self.data), annotations_path)

        self._validate_data()

    @staticmethod
    def _has_valid_schema(item):
        if not isinstance(item, dict) or "image" not in item or "conversations" not in item:
            return False
        convs = item["conversations"]
        if not isinstance(convs, list) or len(convs) < 2 or len(convs) % 2 != 0:
            return False
        return all(isinstance(c, dict) and "from" in c and "value" in c for c in convs)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.image_dir, item["image"])

        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]

        convs = item["conversations"]
        image_placeholder = self.image_token * self.num_image_tokens

        input_ids, labels = [], []
        for turn_idx in range(0, len(convs), 2):
            human_text = convs[turn_idx]["value"].strip()
            gpt_text = convs[turn_idx + 1]["value"].strip()

            if turn_idx == 0:
                # Split off the image block so the overflow branch below can trim the
                # question without touching it. The seam sits right after a run of
                # special tokens, so this tokenizes identically to one combined string.
                human_text = human_text.replace(self.image_token, "").strip()
                head_text = f"USER: {image_placeholder}\n"
                body_text = f"{human_text}\nASSISTANT: "
            else:
                head_text = ""
                body_text = f"USER: {human_text}\nASSISTANT: "

            head_ids = self.tokenizer(head_text, add_special_tokens=False)["input_ids"] if head_text else []
            body_ids = self.tokenizer(body_text, add_special_tokens=False)["input_ids"]
            answer_ids = self.tokenizer(gpt_text, add_special_tokens=False)["input_ids"]
            answer_ids = answer_ids + [self.tokenizer.eos_token_id]

            turn_len = len(head_ids) + len(body_ids) + len(answer_ids)

            # Drop whole trailing turns rather than keeping a half one: a truncated tail
            # contributes only -100 labels, and a batch made entirely of such samples
            # gives a NaN loss.
            if input_ids and len(input_ids) + turn_len > self.max_length:
                logger.debug(
                    "Sample %d (%s): dropping turns from #%d on to fit max_length=%d",
                    idx, item["image"], turn_idx // 2 + 1, self.max_length,
                )
                break

            if turn_len > self.max_length:
                # Only reachable on the very first turn (later ones hit the break above).
                # Trim the question, never the answer past its last token — an all -100
                # sample is worse than a clipped question.
                logger.warning(
                    "Sample %d (%s): first turn is %d tokens, trimming the question to fit "
                    "max_length=%d", idx, item["image"], turn_len, self.max_length,
                )
                budget = self.max_length - len(head_ids)
                answer_ids = answer_ids[: max(1, min(len(answer_ids), budget - 1))]
                body_ids = body_ids[: max(1, budget - len(answer_ids))]

            prompt_ids = head_ids + body_ids
            input_ids += prompt_ids + answer_ids
            labels += [-100] * len(prompt_ids) + answer_ids

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


__all__ = ["VQADataset", "VQAInstructDataset", "build_collate_fn"]
