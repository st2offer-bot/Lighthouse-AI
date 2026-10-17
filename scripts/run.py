import argparse
import json
from forte_ai import analyze_video
p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("--model", default=None)
p.add_argument("--output", default="result.json")
a = p.parse_args()
result = analyze_video(a.video, model_path=a.model)
with open(a.output, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != "candidates"}, ensure_ascii=False, indent=2))
print(f"Candidate windows: {[(x['timecode_sec'], x['anomaly_score']) for x in result['candidates']]}")
print(f"Saved full result to {a.output}")
