#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


STYLE_LABELS = {
    "3d_cgi": "3D CGI",
    "2d_animation": "2D animation",
    "live_action": "live-action",
}

HUMAN_TYPES = {
    "human",
    "person",
    "adult",
    "child",
    "man",
    "woman",
    "boy",
    "girl",
}

UNCLEAR_VALUES = {"", "unclear", "unknown", "n/a", "none", "null"}

ID_PATTERN = re.compile(r"(<C\d{2}>)(?!\()")
TRAILING_PAREN_PATTERN = re.compile(r"\s*\(([^()]*)\)\.?\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--captions-root",
        required=True,
        help="Root directory containing per-chunk JSON caption files.",
    )
    parser.add_argument(
        "--write-sidecar",
        action="store_true",
        help="Write a .merged.txt file next to each JSON.",
    )
    parser.add_argument(
        "--no-update-json",
        action="store_true",
        help="Do not write merged_caption back into the source JSON.",
    )
    return parser.parse_args()


def is_unclear(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text.lower() in UNCLEAR_VALUES


def clean_text(text: object) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def normalize_entity_type(value: object) -> str:
    text = clean_text(value).lower()
    if text in HUMAN_TYPES:
        return "person"
    return text


def pretty_style(value: object) -> str:
    style = clean_text(value).lower()
    if not style or style in UNCLEAR_VALUES:
        return ""
    return STYLE_LABELS.get(style, style.replace("_", " "))


def parse_timecode(value: object) -> Tuple[int, int, int, int]:
    text = clean_text(value)
    if not text:
        return (0, 0, 0, 0)
    parts = text.split(":")
    if len(parts) != 3:
        return (0, 0, 0, 0)
    hour = int(parts[0])
    minute = int(parts[1])
    if "." in parts[2]:
        second_text, frac_text = parts[2].split(".", 1)
        second = int(second_text)
        frac = int((frac_text + "000")[:3])
    else:
        second = int(parts[2])
        frac = 0
    return (hour, minute, second, frac)


def strip_trailing_metadata(caption: str) -> str:
    text = clean_text(caption)
    while True:
        match = TRAILING_PAREN_PATTERN.search(text)
        if not match:
            break
        text = text[: match.start()].rstrip(" ,.")
    return text.rstrip(" .")


def collect_entities(data: Dict) -> Dict[str, Dict]:
    entity_map: Dict[str, Dict] = {}
    for ent in data.get("entities", []):
        ent_id = clean_text(ent.get("id"))
        if ent_id:
            entity_map[ent_id] = dict(ent)
    for seg in data.get("segments", []):
        for ent in seg.get("entities", []):
            ent_id = clean_text(ent.get("id"))
            if ent_id and ent_id not in entity_map:
                entity_map[ent_id] = dict(ent)
    return entity_map


def describe_entity(entity: Dict) -> str:
    if not entity:
        return ""
    parts: List[str] = []

    ent_type = normalize_entity_type(entity.get("type"))
    if ent_type and ent_type not in UNCLEAR_VALUES:
        parts.append(ent_type)

    gender = clean_text(entity.get("gender_presentation")).lower()
    if gender and gender not in UNCLEAR_VALUES:
        parts.append(gender)

    ethnicity = clean_text(entity.get("ethnicity_or_race"))
    if ethnicity and ethnicity.lower() not in UNCLEAR_VALUES:
        parts.append(ethnicity)

    appearance = clean_text(entity.get("appearance")).strip(" ,.;")
    if appearance and appearance.lower() not in UNCLEAR_VALUES:
        parts.append(appearance)

    deduped: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(part)
    return ", ".join(deduped)


def remove_redundant_entity_appearance(
    caption: str,
    segment_entities: Dict[str, Dict],
    entity_map: Dict[str, Dict],
) -> str:
    cleaned = caption
    seen_ids = []
    for match in ID_PATTERN.finditer(caption):
        ent_id = match.group(1).strip("<>")
        if ent_id not in seen_ids:
            seen_ids.append(ent_id)

    for ent_id in seen_ids:
        ent = segment_entities.get(ent_id) or entity_map.get(ent_id)
        if not ent:
            continue
        desc = describe_entity(ent)
        appearance = clean_text(ent.get("appearance")).strip(" ,.;")
        if not desc or not appearance or appearance.lower() in UNCLEAR_VALUES:
            continue
        token = f"<{ent_id}>({desc})"
        pattern = re.compile(
            rf"{re.escape(token)}\s+{re.escape(appearance)}(?=[\s,.;]|$)",
            re.IGNORECASE,
        )
        cleaned = pattern.sub(token, cleaned)
    return cleaned


def enrich_caption(caption: str, segment: Dict, entity_map: Dict[str, Dict]) -> str:
    present_ids = [clean_text(x) for x in segment.get("present_ids", []) if clean_text(x)]
    segment_entities = {
        clean_text(ent.get("id")): ent
        for ent in segment.get("entities", [])
        if clean_text(ent.get("id"))
    }
    if not present_ids and not segment_entities:
        return caption

    def repl(match: re.Match) -> str:
        token = match.group(1)
        ent_id = token.strip("<>")
        ent = segment_entities.get(ent_id) or entity_map.get(ent_id)
        desc = describe_entity(ent or {})
        if not desc:
            return token
        return f"{token}({desc})"

    expanded = ID_PATTERN.sub(repl, caption)
    return remove_redundant_entity_appearance(expanded, segment_entities, entity_map)


def choose_style(segments: Iterable[Dict]) -> str:
    ordered: List[str] = []
    counts: Counter = Counter()
    for seg in segments:
        style = clean_text(seg.get("visual_style")).lower()
        if not style or style in UNCLEAR_VALUES:
            continue
        counts[style] += 1
        if style not in ordered:
            ordered.append(style)
    if not counts:
        return ""
    best_count = max(counts.values())
    for style in ordered:
        if counts[style] == best_count:
            return pretty_style(style)
    return pretty_style(ordered[0])


def build_merged_caption(data: Dict) -> str:
    segments = list(data.get("segments", []))
    if not segments:
        return ""

    segments.sort(key=lambda seg: (parse_timecode(seg.get("start_time")), parse_timecode(seg.get("end_time"))))
    style = choose_style(segments)
    prefix = f"In this {style} sequence, " if style else "In this sequence, "
    entity_map = collect_entities(data)

    parts: List[str] = []
    for idx, seg in enumerate(segments, start=1):
        base_caption = strip_trailing_metadata(clean_text(seg.get("caption")))
        if not base_caption:
            continue
        merged_caption = enrich_caption(base_caption, seg, entity_map)
        parts.append(f"{idx}. {merged_caption}")

    if not parts:
        return prefix.rstrip()
    return prefix + ", ".join(parts) + "."


def iter_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".json.bak"):
            continue
        yield path


def main() -> None:
    args = parse_args()
    captions_root = Path(args.captions_root)
    json_paths = list(iter_json_files(captions_root))

    updated = 0
    for json_path in json_paths:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        merged_caption = build_merged_caption(data)

        if not args.no_update_json:
            data["merged_caption"] = merged_caption
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.write_sidecar:
            sidecar_path = json_path.with_suffix(".merged.txt")
            sidecar_path.write_text(merged_caption + "\n", encoding="utf-8")

        updated += 1
        if updated % 200 == 0:
            print(f"[INFO] processed={updated}/{len(json_paths)}")

    print(f"[DONE] processed={updated}")


if __name__ == "__main__":
    main()
