import os
import glob
import json
import math
import re
import traceback
from collections import Counter

import torch
try:
    import av
except ImportError:
    av = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct").strip()

DATASET_ROOT = os.environ.get("DATASET_ROOT", "").strip()
VIDEO_GLOB = os.environ.get("VIDEO_GLOB", "").strip()
if not VIDEO_GLOB and DATASET_ROOT:
    VIDEO_GLOB = os.path.join(DATASET_ROOT, "*", "*.mp4")
VIDEO_LIST_FILE = os.environ.get("VIDEO_LIST_FILE", "").strip()
OUT_ROOT = os.environ.get("OUT_ROOT", "").strip()
GRID_ROOT = os.environ.get("GRID_ROOT", "").strip()
GRID_DIR_NAME = os.environ.get("GRID_DIR_NAME", "16grid").strip() or "16grid"
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
OVERWRITE_EXISTING = os.environ.get("OVERWRITE_EXISTING", "").strip().lower() in {"1", "true", "yes", "y", "on"}

# video constraints
FPS = 0.5
MAX_PIXELS = 256 * 32 * 32
MIN_PIXELS = 4 * 32 * 32
TOTAL_PIXELS = 20480 * 32 * 32

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))
RETRY_MAX_NEW_TOKENS = int(os.environ.get("RETRY_MAX_NEW_TOKENS", "2048"))
DO_SAMPLE = False
MAX_ENTITY_CATALOG = int(os.environ.get("MAX_ENTITY_CATALOG", "8"))
RETRY_MAX_ENTITY_CATALOG = int(os.environ.get("RETRY_MAX_ENTITY_CATALOG", "4"))
COMPACT_SCHEMA = os.environ.get("COMPACT_SCHEMA", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
REQUIRE_NONEMPTY_ENTITIES = os.environ.get("REQUIRE_NONEMPTY_ENTITIES", "").strip().lower() in {"1", "true", "yes", "y", "on"}

PRIMARY_ENTITY_TYPES = {"human", "animal", "creature", "vehicle"}
BACKGROUND_ENTITY_KEYWORDS = {
    "tree", "trees", "grass", "bush", "bushes", "leaf", "leaves", "flower", "flowers",
    "sky", "cloud", "clouds", "sun", "moon", "mountain", "mountains", "water", "river",
    "ocean", "beach", "road", "sidewalk", "wall", "floor", "ceiling", "room background",
    "building facade", "generic building", "furniture", "audience", "crowd",
}
SMALL_PROP_KEYWORDS = {
    "bottle", "bottles", "cup", "cups", "crate", "basket", "bag", "money bag", "coin slot",
    "slot", "plant", "potted plant", "sign", "stand", "machine", "base", "lemons",
}
FORBIDDEN_APPEARANCE_PATTERNS = (
    r",\s*being held\b.*",
    r",\s*hanging from\b.*",
    r",\s*mounted on\b.*",
    r",\s*next to\b.*",
    r",\s*near\b.*",
    r",\s*inside\b.*",
)

if COMPACT_SCHEMA:
    PROMPT_TEXT = f"""
Split the video into 10-second segments. Return STRICT JSON ONLY.

1) Identify AT MOST {MAX_ENTITY_CATALOG} foreground, story-relevant main subjects across the whole video. Assign stable IDs C01, C02, ...
Important:
- Persistence alone is NOT enough.
- An entity must be a foreground subject that a human would likely mention as the subject of a caption.
- If many visually similar people or animals appear as a crowd, herd, flock, pack, squad, audience, or group, collapse them into ONE group entity instead of assigning one ID per member.
- Do NOT assign separate IDs to interchangeable near-identical members unless the video consistently distinguishes them as different recurring individuals.
- If there is one clearly tracked individual plus many similar background members, keep the tracked individual and optionally one group entity for the rest.
- Prioritize, in order:
  1. main humans / characters / animals
  2. foreground vehicles / creatures / objects that move, are being handled, or stay camera-centered
  3. a specific building / landmark / prop ONLY if the whole video is mainly about that thing and there is no stronger foreground subject
- Prefer fewer entities to wrong entities.
- Hard limit: never output more than {MAX_ENTITY_CATALOG} entities. If more candidates exist, merge or drop lower-priority ones instead of creating more IDs.
- Merge visually identical recurring entities. Do NOT create duplicate IDs for the same recurring person/object.
- Focus on main subjects only. Ignore static background scenery, generic environment, audience extras, and tiny incidental objects even if they persist.
- Never use these as entities unless the video is explicitly about that thing and there is no stronger foreground subject:
  tree, grass, bush, leaves, flowers, sky, clouds, sun, moon, mountain, water, river, ocean, beach, road, sidewalk, wall, floor, ceiling, generic building facade, room background, audience extras, generic furniture.
For each entity include:
- id
- style
- type
- gender_presentation
- ethnicity_or_race (ONLY if type is "human"): e.g., "White", "Black", "Asian", "Latino/Hispanic", "Middle Eastern", "Indigenous", "mixed", or "unclear".
- appearance: ONLY static, identity-like visual attributes (NO actions / states / relations).
  Allowed: species/type, color, clothing, hairstyle, accessories, distinctive marks.
  Forbidden (do NOT include in appearance): any actions/states/relations, e.g.
  "being held", "sleeping", "sitting", "standing", "walking", "talking", "driving",
  "next to", "with", "in a car", "near the truck".
  If needed, those belong to per-segment action/scene, NOT appearance.

2) Produce a coarse timeline of ALL 10-second segments with:
- start_time, end_time
- present_ids: list of entity IDs from the entity catalog that are visible in that segment

Rules:
- Only visible facts. No movie names/source/backstory.
- If unsure, use "unclear".
- Keep the output compact.
- If humans are absent, use only clear foreground animals, vehicles, props, or creatures as entities.
- If the video is mostly scenery or background environment with no clear foreground subject, "entities" may be [].
- Do NOT force background objects into the entity catalog just to avoid emptiness.
- If a segment has no catalog entity visible, use "present_ids": [].
- DO NOT repeat the full entity objects inside each segment.

Output format (STRICT JSON ONLY, no extra text):
{{
  "entities": [
    {{
      "id": "C01",
      "style": "realistic/cartoon/anime/3d_cgi/unclear",
      "type": "human/animal/object/creature/unclear",
      "gender_presentation": "male/female/nonbinary/unclear",
      "ethnicity_or_race": "only for human; otherwise omit or use unclear",
      "appearance": "short static visual cues only"
    }}
  ],
  "segments": [
    {{
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "present_ids": ["C01", "C02"]
    }}
  ]
}}
""".strip()
else:
    PROMPT_TEXT = r"""
Split the video into several segments (each 10 seconds). Return STRICT JSON ONLY.

1) Identify foreground, story-relevant main subjects across the video. Assign stable IDs C01, C02, ...
Important:
- Persistence alone is NOT enough.
- An entity must be a foreground subject that a human would likely mention as the subject of a caption.
- If many visually similar people or animals appear as a crowd, herd, flock, pack, squad, audience, or group, collapse them into ONE group entity instead of assigning one ID per member.
- Do NOT assign separate IDs to interchangeable near-identical members unless the video consistently distinguishes them as different recurring individuals.
- Prioritize main humans / characters / animals, then foreground vehicles / handled objects / creatures.
- Prefer fewer entities to wrong entities.
- Never exceed the entity limit; merge or drop lower-priority candidates instead of creating more IDs.
- Ignore static background scenery and generic environment even if they recur.
- Never use generic background as entities unless the video is mainly about that thing:
  tree, grass, sky, clouds, road, wall, floor, ceiling, generic building facade, room background, audience extras, generic furniture.
For each entity include:
- id
- style
- type
- gender_presentation
- ethnicity_or_race (ONLY if type is "human"): e.g., "White", "Black", "Asian", "Latino/Hispanic", "Middle Eastern", "Indigenous", "mixed", or "unclear".
- appearance: ONLY static, identity-like visual attributes (NO actions / states / relations).
  Allowed: species/type, color, clothing, hairstyle, accessories, distinctive marks.
  Forbidden (do NOT include in appearance): any actions/states/relations, e.g.
  "being held", "sleeping", "sitting", "standing", "walking", "talking", "driving",
  "next to", "with", "in a car", "near the truck".
  If needed, those belong to per-segment action/scene, NOT appearance.

2) Produce a coarse timeline of all segments with:
- start_time, end_time
- present_ids: list of entity IDs present

Rules:
- Only visible facts. No movie names/source/backstory.
- If unsure, use "unclear".
- If there is no clear foreground subject, "entities" may be [].

Output format (STRICT JSON ONLY, no extra text):
[
  {
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "entities": [
      {
        "id": "C01",
        "appearance": "short observable cues (e.g., man/cat, red hoodie, glasses, backpack / orange fur, collar)"
      }
    ]
  }
]
""".strip()

STRICT_RETRY_PROMPT_TEXT = f"""
Split the video into 10-second segments. Return STRICT JSON ONLY.

Keep the response SHORT and COMPLETE.

1) Identify AT MOST {RETRY_MAX_ENTITY_CATALOG} foreground, story-relevant subjects across the whole video. Assign stable IDs C01, C02, ...
Critical rules:
- NEVER enumerate many similar people/vehicles one by one.
- If multiple visually similar people appear together, collapse them into ONE group entity.
- If multiple visually similar vehicles appear together, collapse them into ONE group entity.
- Prefer 0-2 entities over duplicate entities.
- Only keep a separate individual if the video clearly tracks that individual as distinct from the rest.
- Ignore background scenery and tiny incidental props.

For each entity include only:
- id
- style
- type
- gender_presentation
- ethnicity_or_race (only for human; otherwise omit or use "unclear")
- appearance: short static visual cues only

2) Produce ALL 10-second segments with:
- start_time
- end_time
- present_ids

Rules:
- If you cannot determine a stable entity catalog, use "entities": [] but STILL return the full segment timeline.
- If a segment has no catalog entity visible, use "present_ids": [].
- Output STRICT JSON ONLY with this exact schema:
{{
  "entities": [
    {{
      "id": "C01",
      "style": "realistic/cartoon/anime/3d_cgi/unclear",
      "type": "human/animal/object/creature/unclear",
      "gender_presentation": "male/female/nonbinary/unclear",
      "ethnicity_or_race": "only for human; otherwise omit or use unclear",
      "appearance": "short static visual cues only"
    }}
  ],
  "segments": [
    {{
      "start_time": "HH:MM:SS",
      "end_time": "HH:MM:SS",
      "present_ids": ["C01"]
    }}
  ]
}}
""".strip()


# -----------------------------
# Helpers
# -----------------------------
def build_messages(video_path: str, prompt_text: str = PROMPT_TEXT):
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": FPS,
                    "max_pixels": MAX_PIXELS,
                    "min_pixels": MIN_PIXELS,
                    "total_pixels": TOTAL_PIXELS,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def ensure_parent(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)


def extract_complete_array_objects(text: str, key: str):
    key_pos = text.find(f'"{key}"')
    if key_pos < 0:
        return []

    lb = text.find("[", key_pos)
    if lb < 0:
        return []

    i = lb + 1
    n = len(text)
    objs = []
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            break

        start = i
        depth = 0
        in_str = False
        escaped = False
        while i < n:
            ch = text[i]
            if in_str:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        objs.append(text[start:i])
                        break
            i += 1
        else:
            break
    return objs


def recover_truncated_compact_wrapper(text: str):
    if '"entities"' not in text or '"segments"' not in text:
        return None

    entity_objs = extract_complete_array_objects(text, "entities")
    segment_objs = extract_complete_array_objects(text, "segments")
    if not entity_objs or not segment_objs:
        return None

    entities = []
    for raw in entity_objs:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("id"):
            entities.append(obj)

    valid_ids = {str(ent.get("id")).strip() for ent in entities if isinstance(ent, dict)}
    segments = []
    for raw in segment_objs:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "start_time" not in obj or "end_time" not in obj:
            continue
        present_ids = obj.get("present_ids", [])
        if not isinstance(present_ids, list):
            present_ids = []
        obj["present_ids"] = [pid for pid in present_ids if pid in valid_ids]
        segments.append(obj)

    if not entities or not segments:
        return None
    return {"entities": entities, "segments": segments}


def safe_json_loads(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in (r"\{.*\}", r"\[.*\]"):
        m = re.search(pattern, text, flags=re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            continue

    recovered = recover_truncated_compact_wrapper(text)
    if recovered is not None:
        return recovered
    raise json.JSONDecodeError("unable to parse json", text, 0)


def clean_entity_appearance(text):
    appearance = " ".join(str(text or "").split()).strip(" ,;")
    if not appearance:
        return "unclear"
    for pattern in FORBIDDEN_APPEARANCE_PATTERNS:
        appearance = re.sub(pattern, "", appearance, flags=re.IGNORECASE).strip(" ,;")
    return appearance or "unclear"


def format_hhmmss(seconds):
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_video_duration_seconds(video_path):
    if av is None:
        return None
    try:
        container = av.open(video_path)
        try:
            if container.duration is not None:
                return float(container.duration / 1_000_000)
            if container.streams.video:
                stream = container.streams.video[0]
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
        finally:
            container.close()
    except Exception:
        return None
    return None


def build_empty_timeline_from_vlm(video_path):
    chunk_stem = os.path.splitext(os.path.basename(video_path))[0]
    vlm_dir = os.path.dirname(video_path)
    case_dir = os.path.dirname(vlm_dir)
    case_id = os.path.basename(case_dir)

    candidate_dirs = [
        os.path.join(case_dir, chunk_stem),
        os.path.join(case_dir, "vlm", chunk_stem),
    ]

    dataset_parent = os.path.dirname(os.path.abspath(DATASET_ROOT))
    candidate_dirs.append(os.path.join(dataset_parent, GRID_DIR_NAME, case_id, chunk_stem))
    if GRID_ROOT:
        candidate_dirs.append(os.path.join(GRID_ROOT, case_id, chunk_stem))

    clip_paths = []
    for candidate in candidate_dirs:
        paths = sorted(glob.glob(os.path.join(candidate, "*.mp4")))
        if paths:
            clip_paths = paths
            break
    segment_count = len(clip_paths)
    if segment_count == 0:
        duration = get_video_duration_seconds(video_path)
        if duration is None or duration <= 0:
            return None
        segment_count = max(1, int(math.ceil(duration / 10.0)))

    segments = []
    for idx in range(segment_count):
        start = idx * 10
        end = (idx + 1) * 10
        segments.append(
            {
                "start_time": format_hhmmss(start),
                "end_time": format_hhmmss(end),
                "present_ids": [],
            }
        )
    return {"entities": [], "segments": segments}


def entity_priority(entity_type):
    if entity_type == "human":
        return 0
    if entity_type in {"animal", "creature"}:
        return 1
    if entity_type == "vehicle":
        return 2
    if entity_type == "object":
        return 3
    return 4


def should_drop_background_entity(entity_type, appearance_lower):
    if entity_type in PRIMARY_ENTITY_TYPES:
        return False
    return any(keyword in appearance_lower for keyword in BACKGROUND_ENTITY_KEYWORDS)


def collapse_duplicate_entities(entities, segments):
    canonical_for = {}
    canonical_entities = []
    seen_keys = {}

    for ent in entities:
        entity_id = str(ent.get("id", "")).strip()
        if not entity_id:
            continue
        key = (
            str(ent.get("style", "unclear")).strip().lower(),
            str(ent.get("type", "unclear")).strip().lower(),
            str(ent.get("gender_presentation", "unclear")).strip().lower(),
            str(ent.get("ethnicity_or_race", "unclear")).strip().lower(),
            str(ent.get("appearance", "unclear")).strip().lower(),
        )
        canonical = seen_keys.get(key)
        if canonical is None:
            seen_keys[key] = entity_id
            canonical_for[entity_id] = entity_id
            canonical_entities.append(ent)
        else:
            canonical_for[entity_id] = canonical

    remapped_segments = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        remapped = []
        seen = set()
        for pid in seg.get("present_ids", []):
            pid = str(pid).strip()
            mapped = canonical_for.get(pid)
            if mapped and mapped not in seen:
                remapped.append(mapped)
                seen.add(mapped)
        cloned = dict(seg)
        cloned["present_ids"] = remapped
        remapped_segments.append(cloned)

    return canonical_entities, remapped_segments


def filter_and_reindex_compact_data(data):
    if not isinstance(data, dict):
        return data

    entities = data.get("entities")
    segments = data.get("segments")
    if not isinstance(entities, list) or not isinstance(segments, list):
        return data

    segment_counts = Counter()
    total_segments = len(segments)
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        seen = set()
        for pid in seg.get("present_ids", []):
            pid = str(pid).strip()
            if pid and pid not in seen:
                segment_counts[pid] += 1
                seen.add(pid)

    cleaned_entities = []
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        entity_id = str(ent.get("id", "")).strip()
        if not entity_id:
            continue
        entity_type = str(ent.get("type", "unclear")).strip().lower() or "unclear"
        appearance = clean_entity_appearance(ent.get("appearance"))
        appearance_lower = appearance.lower()
        if should_drop_background_entity(entity_type, appearance_lower):
            continue
        cloned = dict(ent)
        cloned["type"] = entity_type
        cloned["appearance"] = appearance
        cleaned_entities.append(cloned)

    if not cleaned_entities:
        data["entities"] = []
        for seg in segments:
            if isinstance(seg, dict):
                seg["present_ids"] = []
        return data

    cleaned_entities, segments = collapse_duplicate_entities(cleaned_entities, segments)

    has_primary = any(str(ent.get("type", "unclear")).strip().lower() in PRIMARY_ENTITY_TYPES for ent in cleaned_entities)
    kept_entities = []
    for ent in cleaned_entities:
        entity_id = str(ent.get("id")).strip()
        entity_type = str(ent.get("type", "unclear")).strip().lower()
        appearance_lower = str(ent.get("appearance", "")).lower()
        count = segment_counts.get(entity_id, 0)
        ratio = (count / total_segments) if total_segments else 0.0

        if count == 0:
            continue
        if has_primary and entity_type not in PRIMARY_ENTITY_TYPES:
            if count <= 1:
                continue
            if any(keyword in appearance_lower for keyword in SMALL_PROP_KEYWORDS) and ratio < 0.75:
                continue
            if ratio < 0.60:
                continue
        kept_entities.append(ent)

    if not kept_entities:
        kept_entities = [ent for ent in cleaned_entities if str(ent.get("type", "unclear")).strip().lower() in PRIMARY_ENTITY_TYPES]

    if has_primary:
        primary_entities = [ent for ent in kept_entities if str(ent.get("type", "unclear")).strip().lower() in PRIMARY_ENTITY_TYPES]
        support_entities = [ent for ent in kept_entities if str(ent.get("type", "unclear")).strip().lower() not in PRIMARY_ENTITY_TYPES]
        support_entities.sort(
            key=lambda ent: (
                -segment_counts.get(str(ent.get("id")).strip(), 0),
                entity_priority(str(ent.get("type", "unclear")).strip().lower()),
                str(ent.get("id")).strip(),
            )
        )
        kept_entities = primary_entities + support_entities[:1]

    kept_entities.sort(
        key=lambda ent: (
            entity_priority(str(ent.get("type", "unclear")).strip().lower()),
            -segment_counts.get(str(ent.get("id")).strip(), 0),
            str(ent.get("id")).strip(),
        )
    )
    kept_entities = kept_entities[:MAX_ENTITY_CATALOG]

    id_map = {}
    reindexed_entities = []
    for idx, ent in enumerate(kept_entities, 1):
        old_id = str(ent.get("id")).strip()
        new_id = f"C{idx:02d}"
        id_map[old_id] = new_id
        cloned = dict(ent)
        cloned["id"] = new_id
        reindexed_entities.append(cloned)

    reindexed_segments = []
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        new_ids = []
        seen = set()
        for pid in seg.get("present_ids", []):
            pid = str(pid).strip()
            mapped = id_map.get(pid)
            if mapped and mapped not in seen:
                new_ids.append(mapped)
                seen.add(mapped)
        cloned = dict(seg)
        cloned["present_ids"] = new_ids
        reindexed_segments.append(cloned)

    return {"entities": reindexed_entities, "segments": reindexed_segments}


def load_video_paths():
    if VIDEO_LIST_FILE:
        with open(VIDEO_LIST_FILE, "r", encoding="utf-8") as f:
            paths = [line.strip() for line in f if line.strip()]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"missing videos in VIDEO_LIST_FILE: {missing[:5]}")
        return sorted(dict.fromkeys(paths))
    return sorted(glob.glob(VIDEO_GLOB))


def rel_to_out(video_path: str, in_root=DATASET_ROOT, out_root=OUT_ROOT):
    """
    把 DATASET_ROOT/A/B.mp4
    映射到 OUT_ROOT/A/B.json (同目录结构)
    """
    rel = os.path.relpath(video_path, in_root)
    rel_no_ext = os.path.splitext(rel)[0]
    out_json = os.path.join(out_root, rel_no_ext + ".json")
    out_raw = os.path.join(out_root, rel_no_ext + ".raw.txt")
    out_err = os.path.join(out_root, rel_no_ext + ".error.txt")
    return out_json, out_raw, out_err


@torch.inference_mode()
def run_one(model, processor, video_path: str, prompt_text: str = PROMPT_TEXT, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = build_messages(video_path, prompt_text=prompt_text)

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,          # Qwen3-VL关键
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,              # 已在 qwen_vl_utils resize
        **video_kwargs,
    ).to(model.device)

    inputs.pop("token_type_ids", None)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=DO_SAMPLE,
    )
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    out_text = processor.batch_decode(gen_only, skip_special_tokens=True)[0]
    return out_text


def parse_and_normalize_output(video_path: str, out_text: str):
    data = safe_json_loads(out_text)
    notes = []

    if COMPACT_SCHEMA and isinstance(data, dict):
        before_entities = len(data.get("entities", [])) if isinstance(data.get("entities"), list) else 0
        if isinstance(data.get("segments"), list) and not data["segments"]:
            fallback = build_empty_timeline_from_vlm(video_path)
            if fallback is not None:
                data = fallback
                notes.append("fallback empty timeline")
        data = filter_and_reindex_compact_data(data)
        after_entities = len(data.get("entities", [])) if isinstance(data.get("entities"), list) else 0
        if before_entities != after_entities:
            notes.append(f"filtered entities: {before_entities} -> {after_entities}")

    if (
        COMPACT_SCHEMA
        and isinstance(data, dict)
        and isinstance(data.get("segments"), list)
        and not data["segments"]
    ):
        raise json.JSONDecodeError("empty compact segments", out_text, 0)
    if (
        COMPACT_SCHEMA
        and REQUIRE_NONEMPTY_ENTITIES
        and isinstance(data, dict)
        and isinstance(data.get("segments"), list)
        and not data.get("entities")
    ):
        raise json.JSONDecodeError("empty compact entities", out_text, 0)
    return data, notes


def main():
    if not DATASET_ROOT:
        raise ValueError("DATASET_ROOT is required")
    if not OUT_ROOT:
        raise ValueError("OUT_ROOT is required")
    if not VIDEO_LIST_FILE and not VIDEO_GLOB:
        raise ValueError("Set VIDEO_LIST_FILE or VIDEO_GLOB")
    if SHARD_COUNT < 1:
        raise ValueError(f"SHARD_COUNT must be >= 1, got {SHARD_COUNT}")
    if not 0 <= SHARD_INDEX < SHARD_COUNT:
        raise ValueError(f"SHARD_INDEX must satisfy 0 <= index < count, got index={SHARD_INDEX}, count={SHARD_COUNT}")

    # Load model once
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    all_paths = load_video_paths()
    paths = [vp for idx, vp in enumerate(all_paths) if idx % SHARD_COUNT == SHARD_INDEX]
    print(f"[INFO] dataset_root={DATASET_ROOT}")
    print(f"[INFO] out_root={OUT_ROOT}")
    print(f"[INFO] shard={SHARD_INDEX}/{SHARD_COUNT}")
    if VIDEO_LIST_FILE:
        print(f"[INFO] video_list_file={VIDEO_LIST_FILE}")
    else:
        print(f"[INFO] video_glob={VIDEO_GLOB}")
    print(f"[INFO] overwrite_existing={OVERWRITE_EXISTING}")
    print(f"[INFO] compact_schema={COMPACT_SCHEMA}")
    print("Found videos:", len(all_paths))
    print("Videos in this shard:", len(paths))
    if not paths:
        return

    ok = 0
    fail = 0
    skipped = 0

    for i, vp in enumerate(paths, 1):
        out_json, out_raw, out_err = rel_to_out(vp)

        if os.path.exists(out_json) and not OVERWRITE_EXISTING:
            print(f"[{i}/{len(paths)}] SKIP (exists): {vp} -> {out_json}")
            skipped += 1
            continue

        ensure_parent(out_json)
        ensure_parent(out_raw)
        ensure_parent(out_err)

        print(f"[{i}/{len(paths)}] RUN: {vp}")
        try:
            out_text = run_one(model, processor, vp)

            # 尝试严格 JSON
            try:
                data, notes = parse_and_normalize_output(vp, out_text)
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                for note in notes:
                    print(f"  -> {note}")
                for stale in (out_raw, out_err):
                    if os.path.exists(stale):
                        os.remove(stale)
                ok += 1
                print(f"  -> OK: {out_json}")
            except json.JSONDecodeError:
                print("  -> retry with stricter compact prompt")
                retry_text = run_one(
                    model,
                    processor,
                    vp,
                    prompt_text=STRICT_RETRY_PROMPT_TEXT,
                    max_new_tokens=RETRY_MAX_NEW_TOKENS,
                )
                try:
                    data, notes = parse_and_normalize_output(vp, retry_text)
                    with open(out_json, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    for note in notes:
                        print(f"  -> {note}")
                    for stale in (out_raw, out_err):
                        if os.path.exists(stale):
                            os.remove(stale)
                    ok += 1
                    print(f"  -> OK after retry: {out_json}")
                except json.JSONDecodeError:
                    with open(out_raw, "w", encoding="utf-8") as f:
                        f.write(retry_text)
                    fail += 1
                    print(f"  -> NOT JSON after retry, saved raw: {out_raw}")

        except Exception as e:
            fail += 1
            with open(out_err, "w", encoding="utf-8") as f:
                f.write(f"Video: {vp}\n\n")
                f.write("Exception:\n")
                f.write(str(e) + "\n\n")
                f.write("Traceback:\n")
                f.write(traceback.format_exc())
            print(f"  -> FAIL, saved error: {out_err}")

    print(f"Done. ok={ok}, fail={fail}, skipped={skipped}, total={len(paths)}")
    print("Outputs under:", OUT_ROOT)


if __name__ == "__main__":
    main()
