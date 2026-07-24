import json

import torch
from PIL import Image

from src.data import VQADataset, build_collate_fn


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 50 for c in text]}


class FakeImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 8, 8)}


def _write_annotations(tmp_path, items):
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def _make_image(tmp_path, name):
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)
    Image.new("RGB", (8, 8)).save(img_dir / name)
    return img_dir


def test_simple_schema_masks_question_in_labels(tmp_path):
    img_dir = _make_image(tmp_path, "a.jpg")
    ann_path = _write_annotations(tmp_path, [{"image": "a.jpg", "question": "hi", "answer": "ok"}])

    dataset = VQADataset(
        image_dir=str(img_dir),
        annotations_path=str(ann_path),
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
        num_image_tokens=4,
        image_token="<image>",
    )

    assert len(dataset) == 1
    sample = dataset[0]
    input_ids, labels = sample["input_ids"], sample["labels"]

    n_answer_tokens = (labels != -100).sum().item()
    assert n_answer_tokens > 0
    assert torch.equal(labels[-n_answer_tokens:], input_ids[-n_answer_tokens:])
    assert (labels[:-n_answer_tokens] == -100).all()


def test_llava_conversations_schema_is_recognized_and_strips_image_marker(tmp_path):
    img_dir = _make_image(tmp_path, "b.jpg")
    ann_path = _write_annotations(tmp_path, [{
        "image": "b.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\ndescribe this"},
            {"from": "gpt", "value": "a cat"},
        ],
    }])

    dataset = VQADataset(
        image_dir=str(img_dir),
        annotations_path=str(ann_path),
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
        num_image_tokens=4,
        image_token="<image>",
    )

    assert len(dataset) == 1
    question, answer = dataset._extract_qa(dataset.data[0])
    assert "<image>" not in question
    assert question == "describe this"
    assert answer == "a cat"


def test_missing_image_and_malformed_items_are_filtered_out(tmp_path):
    img_dir = _make_image(tmp_path, "c.jpg")
    ann_path = _write_annotations(tmp_path, [
        {"image": "c.jpg", "question": "q", "answer": "a"},
        {"image": "missing.jpg", "question": "q", "answer": "a"},
        {"image": "c.jpg"},
    ])

    dataset = VQADataset(
        image_dir=str(img_dir),
        annotations_path=str(ann_path),
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
        num_image_tokens=4,
    )

    assert len(dataset) == 1


def test_collate_fn_attention_mask_survives_pad_token_aliasing_eos():
    """Regression test: attention_mask must come from real lengths, not
    `input_ids != pad_token_id` — otherwise a legitimate eos token that is reused as
    pad_token gets masked out even when it's part of the real sequence."""
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, -100, 3]),
            "pixel_values": torch.zeros(3, 8, 8),
        },
        {
            "input_ids": torch.tensor([1, 2]),
            "labels": torch.tensor([-100, 2]),
            "pixel_values": torch.zeros(3, 8, 8),
        },
    ]
    collate_fn = build_collate_fn(pad_token_id=2)  # deliberately equal to a real token value

    out = collate_fn(batch)

    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["input_ids"][1].tolist() == [1, 2, 2]  # padded with pad_token_id at the end
    assert out["labels"][1].tolist() == [-100, 2, -100]
    assert out["pixel_values"].shape == (2, 3, 8, 8)
