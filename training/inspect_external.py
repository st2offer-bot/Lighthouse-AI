from pathlib import Path
from collections import Counter
import json
import zipfile

root = Path.home() / "Downloads"

archives = sorted(
    [
        p for p in root.rglob("*.zip")
        if "banknote" in p.name.lower()
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

for path in archives:
    print("\nARCHIVE:", path.name)

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        annotations = [
            name for name in names
            if name.endswith(".json")
            and "annotation" in name.lower()
        ]

        print("Files:", len(names))
        print("Annotation files:", annotations)

        for name in annotations:
            try:
                data = json.loads(archive.read(name))
            except (ValueError, UnicodeDecodeError):
                continue

            categories = {
                x["id"]: x["name"]
                for x in data.get("categories", [])
            }

            counts = Counter(
                categories.get(x["category_id"], "unknown")
                for x in data.get("annotations", [])
            )

            print("Split:", name)
            print("Images:", len(data.get("images", [])))
            print("Categories:", categories)
            print("Objects:", dict(counts))

print("\nArchives checked:", len(archives))
