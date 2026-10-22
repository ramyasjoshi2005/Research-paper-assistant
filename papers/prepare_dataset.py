
import json
from pathlib import Path

DATASET_PATH = Path("data/arxiv-metadata.json")

def main():
    if not DATASET_PATH.exists():
        print(f"Dataset not found: {DATASET_PATH.resolve()}")
        print("Check the filename and project folder.")
        return

    print("Reading arXiv dataset...")
    total = 0

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            categories = paper.get("categories", "")
            if any(
                category.startswith(("cs.AI", "cs.LG", "cs.CL", "cs.CV"))
                for category in categories.split()
            ):
                total += 1

                if total <= 5:
                    print("\nPaper:", paper.get("title", "Untitled"))
                    print("ID:", paper.get("id", ""))
                    print("Abstract:", paper.get("abstract", "")[:250])

            if total >= 500:
                break

    print(f"\nFinished! Found {total} relevant papers in the scanned records.")

if __name__ == "__main__":
    main()
