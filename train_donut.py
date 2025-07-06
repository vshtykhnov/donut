#!/usr/bin/env python
"""Fine-tune Donut on the prepared dataset."""
import json
from pathlib import Path
from typing import Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    DonutProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)

class DonutDataset(Dataset):
    """Reads images and JSON ground truth and returns pixel values + labels."""
    def __init__(
        self,
        images_dir: str | Path,
        gt_dir: str | Path,
        processor: DonutProcessor,
        max_length: int = 512,
    ) -> None:
        self.images = sorted(Path(images_dir).glob("*"))
        self.gts: Dict[str, Path] = {p.stem: p for p in Path(gt_dir).glob("*.json*")}
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image and JSON
        img_path = self.images[idx]
        gt = json.loads(Path(self.gts[img_path.stem]).read_text(encoding="utf-8"))

        # Process image and text together
        image = Image.open(img_path).convert("RGB")
        # Debug prints:
        print(f"Processing {img_path.name}")
        # Processor now handles image only; text handled by tokenizer separately
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        tokenized = self.processor.tokenizer(
            json.dumps(gt, ensure_ascii=False),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.squeeze(0)
        # Debug shapes:
        return {"pixel_values": pixel_values, "labels": labels}


def main() -> None:
    train_images = "donut_dataset/images/train"
    train_gts = "donut_dataset/ground_truth/train"
    val_images = "donut_dataset/images/val"
    val_gts = "donut_dataset/ground_truth/val"
    output_dir = "donut_finetuned"
    model_name = "naver-clova-ix/donut-base"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor and model
    processor = DonutProcessor.from_pretrained(model_name)
    processor.tokenizer.model_max_length = 512

    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device)
    # Freeze visual encoder to avoid catastrophic forgetting on small datasets
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.config.decoder_start_token_id = processor.tokenizer.bos_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.gradient_checkpointing_enable()

    # Prepare datasets
    train_ds = DonutDataset(train_images, train_gts, processor)
    val_ds = DonutDataset(val_images, val_gts, processor)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        generation_max_length=512,
        generation_num_beams=5,
        learning_rate=3e-6,
        num_train_epochs=30,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        max_steps=1000,
    )

    # Data collator masks pad tokens
    pad_token_id = processor.tokenizer.pad_token_id
    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        labels = labels.masked_fill(labels == pad_token_id, -100)
        return {"pixel_values": pixel_values, "labels": labels}

    torch.cuda.empty_cache()

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    print("Scheduler:", training_args.lr_scheduler_type)

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
