from pathlib import Path

images = set(p.stem for p in Path("src/images/train").glob("*.jpg"))
labels = set(p.stem for p in Path("src/labels/train").glob("*.txt"))

missing = images - labels
if missing:
    print(f"⚠️ {len(missing)} imagens sem anotações!")