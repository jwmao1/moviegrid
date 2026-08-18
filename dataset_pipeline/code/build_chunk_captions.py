#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


GENDER_MAP = {"male": "man", "female": "woman"}


def clean(value):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or text.lower() == "unclear":
        return None
    return text


def norm(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_style_tuple(caption):
    return re.sub(r"\s*\([^()]*\)\.?\s*$", "", caption).strip()


def entity_style(data):
    styles = []
    for ent in data.get("entities", []):
        if not isinstance(ent, dict):
            continue
        style = clean(ent.get("style"))
        if style and style not in styles:
            styles.append(style)
    if styles:
        return styles[0]
    for seg in data.get("segments", []):
        if not isinstance(seg, dict):
            continue
        style = clean(seg.get("visual_style"))
        if style:
            return style
    return None


def entity_base(ent):
    gender = clean(ent.get("gender_presentation"))
    typ = clean(ent.get("type"))
    if gender:
        return GENDER_MAP.get(gender.lower(), gender)
    if typ and typ != "human":
        return typ
    return None


def build_label(ent):
    race = clean(ent.get("ethnicity_or_race"))
    app = clean(ent.get("appearance"))
    base = entity_base(ent)

    if app:
        app_text = app
        if base and not norm(app).startswith(norm(base)):
            app_text = f"{base} {app}"
        if race and base:
            return f"{race} {base}, {app_text}"
        if race:
            return f"{race}, {app_text}"
        return app_text

    if race and base:
        return f"{race} {base}"
    return race or base or clean(ent.get("type")) or ""


def duplicate_entity_phrases(ent):
    phrases = []
    app = clean(ent.get("appearance"))
    base = entity_base(ent)
    if not app:
        return phrases

    phrases.append(app)
    if " and " in app:
        phrases.append(app.split(" and ", 1)[0].strip())
    if app.lower().startswith("wearing "):
        phrases.append(app[len("wearing ") :].strip())
        if " and " in app:
            first_half = app.split(" and ", 1)[0].strip()
            if first_half.lower().startswith("wearing "):
                phrases.append(first_half[len("wearing ") :].strip())
    if base:
        phrases.append(f"{base} {app}")
        if app.lower().startswith("wearing "):
            phrases.append(f"{base} {app[len('wearing '):].strip()}")

    deduped = []
    seen = set()
    for phrase in phrases:
        phrase_n = norm(phrase)
        if phrase_n and phrase_n not in seen:
            deduped.append(phrase)
            seen.add(phrase_n)
    return deduped


def phrase_prefix_patterns(phrase, min_words=3):
    words = re.findall(r"[A-Za-z0-9']+", phrase)
    patterns = []
    for count in range(len(words), min_words - 1, -1):
        prefix = words[:count]
        sep = r"(?:[\s,./-]+)"
        pattern = r"\b" + sep.join(re.escape(word) + r"\b" for word in prefix)
        patterns.append(pattern)
    return patterns


def cleanup_text(text):
    text = strip_style_tuple(text)
    text = re.sub(r"\bwith is\b", "is", text, flags=re.I)
    text = re.sub(r"\bwith are\b", "are", text, flags=re.I)
    text = re.sub(r"\bmustache hand reaching\b", "mustache is reaching", text, flags=re.I)
    text = re.sub(r"\bwearing ([^,.;]+?) and light is\b", r"wearing \1 is", text, flags=re.I)
    text = re.sub(r"\bin suit reaching\b", "reaching", text, flags=re.I)
    text = re.sub(r"\bin suit is\b", "is", text, flags=re.I)
    text = re.sub(r"\bis wearing helmet and holding\b", "is holding", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    if text and not text.endswith("."):
        text += "."
    return text


def apply_entities(caption, entities):
    text = strip_style_tuple(caption)
    valid = []
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        ent_id = clean(ent.get("id"))
        if not ent_id:
            continue
        label = build_label(ent)
        valid.append((ent_id, ent, label))
        if label:
            text = re.sub(fr"<{re.escape(ent_id)}>", f"<{ent_id}>({label})", text)

    for ent_id, ent, label in valid:
        if not label:
            continue
        for phrase in duplicate_entity_phrases(ent):
            text = re.sub(
                fr"(<{re.escape(ent_id)}>\([^)]*\))\s+{re.escape(phrase)}\b[ ,]*",
                r"\1 ",
                text,
                flags=re.I,
            )
            for prefix_pattern in phrase_prefix_patterns(phrase):
                text = re.sub(
                    fr"(<{re.escape(ent_id)}>\([^)]*\))\s+{prefix_pattern}[ ,]*",
                    r"\1 ",
                    text,
                    flags=re.I,
                )

    return cleanup_text(text)


def build_chunk_text(data):
    parts = []
    for idx, seg in enumerate(data.get("segments", []), 1):
        if not isinstance(seg, dict):
            continue
        caption = clean(seg.get("caption")) or ""
        if not caption:
            continue
        entities = seg.get("entities", [])
        merged = apply_entities(caption, entities) if entities else cleanup_text(caption)
        parts.append(f"{idx}. {merged}")

    style = entity_style(data)
    prefix = f"In a {style} style, the sequence begins with " if style else "The sequence begins with "
    return prefix + " ".join(parts)


def write_caption(json_path, out_path):
    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    text = build_chunk_text(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def iter_json_files(src_root):
    for case_dir in sorted(src_root.iterdir()):
        if not case_dir.is_dir():
            continue
        vlm_dir = case_dir / "vlm"
        if not vlm_dir.is_dir():
            continue
        for json_path in sorted(vlm_dir.glob("*.json")):
            yield case_dir.name, json_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--dst-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=[])
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    only = set(args.only)
    processed = 0

    for case_id, json_path in iter_json_files(src_root):
        stem = json_path.stem
        if only and stem not in only:
            continue
        out_path = dst_root / case_id / f"{stem}.caption.txt"
        write_caption(json_path, out_path)
        processed += 1
        if args.limit and processed >= args.limit:
            break
        if processed % 5000 == 0:
            print(f"processed={processed}", flush=True)

    print(f"final_processed={processed}")


if __name__ == "__main__":
    main()
