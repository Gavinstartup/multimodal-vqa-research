import json

import torch
from PIL import Image

import pytest

from src.data import VQADataset, VQAInstructDataset, build_collate_fn


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


def _instruct_dataset(tmp_path, items, max_length=2048, num_image_tokens=4):
    img_dir = _make_image(tmp_path, "d.jpg")
    ann_path = _write_annotations(tmp_path, items)
    return VQAInstructDataset(
        image_dir=str(img_dir),
        annotations_path=str(ann_path),
        image_processor=FakeImageProcessor(),
        tokenizer=FakeTokenizer(),
        num_image_tokens=num_image_tokens,
        image_token="<image>",
        max_length=max_length,
    )


def test_instruct_multi_turn_supervises_every_assistant_span(tmp_path):
    """Every ASSISTANT span contributes loss. An implementation that only read the first
    pair (the way VQADataset does) would silently train on half of a 2-turn sample."""
    dataset = _instruct_dataset(tmp_path, [{
        "image": "d.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nwhat is this"},
            {"from": "gpt", "value": "AAAA"},
            {"from": "human", "value": "how many"},
            {"from": "gpt", "value": "BBB"},
        ],
    }])

    sample = dataset[0]
    supervised = sample["input_ids"][sample["labels"] != -100].tolist()

    a, b, eos = ord("A") % 50, ord("B") % 50, FakeTokenizer.eos_token_id
    assert supervised == [a, a, a, a, eos, b, b, b, eos]


def test_instruct_accepts_mix665k_shape_with_trailing_image_marker(tmp_path):
    """llava_v1_5_mix665k puts <image> at either end of the first human turn and appends a
    format instruction to short-answer items. Neither may leak into the supervised span,
    and the marker must be replaced by the placeholder block rather than kept as text."""
    dataset = _instruct_dataset(tmp_path, [{
        "image": "d.jpg",
        "conversations": [
            {"from": "human", "value": "How many dogs?\nAnswer using a single word.\n<image>"},
            {"from": "gpt", "value": "2"},
        ],
    }])

    assert len(dataset) == 1
    sample = dataset[0]
    supervised = sample["input_ids"][sample["labels"] != -100].tolist()
    assert supervised == [ord("2") % 50, FakeTokenizer.eos_token_id]

    # Marker position must not change the encoding at all: the placeholder block is always
    # emitted first and the literal marker is stripped wherever it sat.
    leading = _instruct_dataset(tmp_path, [{
        "image": "d.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nHow many dogs?\nAnswer using a single word."},
            {"from": "gpt", "value": "2"},
        ],
    }])[0]
    assert sample["input_ids"].tolist() == leading["input_ids"].tolist()
    assert sample["labels"].tolist() == leading["labels"].tolist()


def test_instruct_drops_whole_trailing_turns_instead_of_truncating(tmp_path):
    """A tail-truncated turn contributes only -100 labels, and a batch made entirely of
    those yields a NaN loss. The overflowing turn must disappear, not be cut mid-answer."""
    # FakeTokenizer is char-wise, so one image token costs len("<image>") == 7 "tokens".
    # max_length is picked so turn 1 (38) fits and turn 1 + turn 2 (38 + 61) does not —
    # otherwise the first turn overflows on its own and we would be exercising the
    # question-trimming branch instead of the turn-dropping one.
    dataset = _instruct_dataset(tmp_path, [{
        "image": "d.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nq"},
            {"from": "gpt", "value": "A" * 10},
            {"from": "human", "value": "q2"},
            {"from": "gpt", "value": "B" * 40},
        ],
    }], max_length=50, num_image_tokens=1)

    sample = dataset[0]
    supervised = sample["input_ids"][sample["labels"] != -100].tolist()

    assert len(sample["input_ids"]) <= 50
    assert ord("B") % 50 not in supervised
    assert supervised.count(ord("A") % 50) == 10


def test_instruct_rejects_max_length_that_cannot_fit_the_image_block(tmp_path):
    with pytest.raises(ValueError, match="leaves no room"):
        _instruct_dataset(tmp_path, [{
            "image": "d.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\nq"},
                {"from": "gpt", "value": "a"},
            ],
        }], max_length=4, num_image_tokens=4)
