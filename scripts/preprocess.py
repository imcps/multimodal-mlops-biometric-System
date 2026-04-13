import os
import ray
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

RAW_DATA_DIR = "data/raw"
OUTPUT_PARQUET = "data/processed/biometric.parquet"


@ray.remote
def process_image(person_id: str, modality: str, image_path: str) -> dict:
    """
    Validates an image and returns structured metadata.
    """
    with Image.open(image_path) as img:
        img.verify()  # validates file integrity only

    return {
        "person_id": int(person_id),
        "modality": modality,
        "image_path": image_path,
    }


def main():
    ray.init(ignore_reinit_error=True)

    tasks = []

    for person_id in os.listdir(RAW_DATA_DIR):
        person_dir = os.path.join(RAW_DATA_DIR, person_id)
        if not os.path.isdir(person_dir):
            continue

        for modality in ["iris", "fingerprint"]:
            modality_dir = os.path.join(person_dir, modality)
            if not os.path.isdir(modality_dir):
                continue

            for fname in os.listdir(modality_dir):
                image_path = os.path.join(modality_dir, fname)

                tasks.append(
                    process_image.remote(person_id, modality, image_path)
                )

    print(f"🔄 Launching {len(tasks)} preprocessing tasks")

    records = ray.get(tasks)

    table = pa.Table.from_pylist(records)
    os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
    pq.write_table(table, OUTPUT_PARQUET)

    print(f"✅ Parquet metadata written to: {OUTPUT_PARQUET}")
    print(f"✅ Total samples processed: {len(records)}")


if __name__ == "__main__":
    main()