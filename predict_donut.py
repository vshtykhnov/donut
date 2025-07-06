#!/usr/bin/env python
"""Run inference with a fine-tuned Donut model."""
import sys
import json
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

def main(model_dir: str, img_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Загрузка процессора и модели
    processor = DonutProcessor.from_pretrained(model_dir, use_fast=True)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device)

    # 2) Читаем и конвертим изображение
    image = Image.open(img_path).convert("RGB")

    # 3) Обрабатываем КАРТИНКУ только через feature_extractor
    enc = processor.feature_extractor(
        images=image,
        return_tensors="pt"
    )
    pixel_values = enc.pixel_values.to(device)

    # 4) Переключаем модель в режим вывода и генерируем без явного BOS
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values=pixel_values,
            max_length=processor.tokenizer.model_max_length,
            num_beams=5,
            early_stopping=True,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
            use_cache=True,
        )

    # 6) Декодируем сначала со всеми спецтоенами
    raw = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(">>> raw with specials:", repr(raw))

    # 7) А потом «чистый» вариант без спецтокенов
    cleaned = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(">>> without specials:", repr(cleaned))

    # 8) Парсим JSON
    try:
        output = json.loads(cleaned)
    except json.JSONDecodeError:
        output = {"error": "failed to parse", "raw": cleaned}

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_donut.py <model_dir> <image_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
