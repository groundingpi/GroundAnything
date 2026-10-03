"""Ground one image using an existing service and save raw output plus visualization."""
import argparse
import json
from pathlib import Path
from grounding_anything import GroundingAnything, visualize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--phrase", required=True)
    p.add_argument("--task", choices=["bbox", "point"], default="bbox")
    p.add_argument("--base-url", default="http://127.0.0.1:8101/v1")
    p.add_argument("--model", default="groundinganything")
    p.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    args = p.parse_args()
    result = GroundingAnything(args.base_url, args.model).predict(args.image, args.phrase, task=args.task)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "result.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n")
    if not result.valid:
        raise SystemExit("Invalid response; raw output saved in result.json: " + str(result.parse_error))
    visualize(args.image, result).save(args.output / "prediction.png")
    print(args.output)


if __name__ == "__main__":
    main()
