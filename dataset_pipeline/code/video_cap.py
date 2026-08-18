import os
import re
import json
import glob
import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info, extract_vision_info, fetch_video, fetch_image

_PVI_PARAMS = set()
try:
    _PVI_PARAMS = set(inspect.signature(process_vision_info).parameters)
except (TypeError, ValueError):
    _PVI_PARAMS = set()

_PVI_HAS_IMAGE_PATCH_SIZE = "image_patch_size" in _PVI_PARAMS
_PVI_HAS_RETURN_VIDEO_KWARGS = "return_video_kwargs" in _PVI_PARAMS
_PVI_HAS_RETURN_VIDEO_METADATA = "return_video_metadata" in _PVI_PARAMS


# ----------------------------
# 1) 配置：只改这里
# ----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct").strip()

# 数据根目录
DATASET_ROOT = os.environ.get("DATASET_ROOT", "").strip()
CAPTIONS_ROOT = os.environ.get("CAPTIONS_ROOT", "").strip()

# 可选：只处理指定 case_id / chunk 名称（None 表示全部）
# 例：CASE_FILTER = {"99"} 或 CHUNK_FILTER = {"99_chunk0004"}
CASE_FILTER = None
CHUNK_FILTER = None
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "1"))
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))

# 出错是否继续下一个 chunk
CONTINUE_ON_ERROR = True

# 如果你的 clips 可能缺号/乱序，建议 True：按目录实际 mp4 的数字名排序映射到 seg_list
USE_SORTED_CLIPS_MAPPING = True

# 子视频输入约束（与你原来一致）
VIDEO_CFG = dict(
    fps=0.5,
    max_pixels=256 * 32 * 32,
    min_pixels=4 * 32 * 32,
    total_pixels=20480 * 32 * 32,
)

STYLE_CHOICES = [
    "realistic",
    "cinematic",
    "comic",
    "cartoon",
    "anime",
    "3d_cgi",
    "stop_motion",
    "pixel_art",
    "game_render",
    "unclear",
]

# ----------------------------
# 2) 长度控制：改这里就能“精炼每段话”
# ----------------------------
MAX_WORDS_SCENE = int(os.environ.get("MAX_WORDS_SCENE", "12"))
MAX_WORDS_ACTION = int(os.environ.get("MAX_WORDS_ACTION", "10"))
MAX_WORDS_CAPTION = int(os.environ.get("MAX_WORDS_CAPTION", "40"))
MAX_WORDS_APPEAR = int(os.environ.get("MAX_WORDS_APPEAR", "10"))
MAX_PROMPT_ENTITIES = int(os.environ.get("MAX_PROMPT_ENTITIES", "12"))
MAX_CAPTION_ENTITY_IDS = int(os.environ.get("MAX_CAPTION_ENTITY_IDS", "3"))
MAX_WORDS_CAPTION_APPEAR = int(os.environ.get("MAX_WORDS_CAPTION_APPEAR", "5"))
MAX_CHUNK_ENTITY_CATALOG = int(os.environ.get("MAX_CHUNK_ENTITY_CATALOG", "8"))
MIN_RECUR_ENTITY_SEGMENTS = int(os.environ.get("MIN_RECUR_ENTITY_SEGMENTS", "2"))

APPEARANCE_ACTION_WORDS = {
    "being",
    "driving",
    "falling",
    "flying",
    "gesturing",
    "holding",
    "jumping",
    "kneeling",
    "looking",
    "lying",
    "moving",
    "performing",
    "playing",
    "pointing",
    "posing",
    "reacting",
    "riding",
    "running",
    "sitting",
    "sleeping",
    "smiling",
    "speaking",
    "standing",
    "swimming",
    "talking",
    "walking",
    "waving",
}

APPEARANCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "near",
    "of",
    "on",
    "the",
    "to",
    "with",
    "wearing",
}


# ----------------------------
# 3) 工具函数
# ----------------------------
def safe_json_obj(text: str) -> Optional[Any]:
    """尽量把模型输出解析成 JSON dict（容错处理 ```json ...``` 或夹杂文本）"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def trim_tail_punct(s: str) -> str:
    return re.sub(r"[,;:，\.。]+$", "", (s or "").strip())


def truncate_words(s: str, max_words: int) -> str:
    """按空格分词截断（英文稳定；中文也能兜底）"""
    s = re.sub(r"\s+", " ", (s or "").strip())
    if max_words <= 0 or not s:
        return s
    ws = s.split(" ")
    if len(ws) <= max_words:
        return s
    return " ".join(ws[:max_words]).strip()


def _recover_json_array(text: str) -> Optional[List[Any]]:
    """尝试从截断的 JSON 数组中尽量恢复有效对象列表。"""
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n and text[idx].isspace():
        idx += 1
    if idx >= n or text[idx] != "[":
        return None
    idx += 1

    items: List[Any] = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        if text[idx] == "]":
            return items
        if text[idx] == ",":
            idx += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, idx)
        except json.JSONDecodeError:
            break
        items.append(obj)
        idx = end

    return items if items else None


def _recover_partial_first_segment(text: str) -> Optional[List[Dict[str, Any]]]:
    """从截断的 raw.txt 中尽量恢复第一个 segment。"""
    start_time = re.search(r'"start_time"\s*:\s*"([^"]*)"', text)
    end_time = re.search(r'"end_time"\s*:\s*"([^"]*)"', text)
    entities_key = re.search(r'"entities"\s*:\s*\[', text)

    entities: List[Dict[str, Any]] = []
    if entities_key:
        decoder = json.JSONDecoder()
        idx = entities_key.end()
        n = len(text)
        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n or text[idx] == "]":
                break
            if text[idx] == ",":
                idx += 1
                continue
            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                entities.append(obj)
            idx = end

    if start_time is None and end_time is None and not entities:
        return None

    seg: Dict[str, Any] = {
        "start_time": start_time.group(1) if start_time else "00:00:00",
        "end_time": end_time.group(1) if end_time else "00:00:10",
        "entities": entities,
    }
    return [seg]


def hydrate_segments_from_catalog(data: Dict[str, Any]) -> Dict[str, Any]:
    catalog = data.get("entities")
    segs = data.get("segments")
    if not isinstance(catalog, list) or not isinstance(segs, list):
        return data

    entity_map = {}
    for ent in catalog:
        if not isinstance(ent, dict):
            continue
        ent_id = str(ent.get("id", "")).strip()
        if not ent_id:
            continue
        entity_map[ent_id] = ent

    for seg in segs:
        if not isinstance(seg, dict):
            continue
        present_ids = seg.get("present_ids", [])
        if not isinstance(present_ids, list):
            present_ids = []
        if isinstance(seg.get("entities"), list) and seg.get("entities"):
            continue
        seg["entities"] = [dict(entity_map[eid]) for eid in present_ids if eid in entity_map]
    return data


def load_segments_from_file(path: str) -> Tuple[Any, List[Dict[str, Any]], bool]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        recovered = _recover_json_array(text)
        if recovered is not None:
            print(f"[WARN] recovered truncated json list: {path} items={len(recovered)}")
            return recovered, recovered, False

        partial = _recover_partial_first_segment(text)
        if partial is None:
            raise
        entity_count = len(partial[0].get("entities", []))
        print(f"[WARN] recovered partial first segment: {path} entities={entity_count}")
        return partial, partial, False

    if isinstance(data, dict) and "segments" in data and isinstance(data["segments"], list):
        data = hydrate_segments_from_catalog(data)
        return data, data["segments"], True
    if isinstance(data, list):
        return data, data, False
    raise ValueError("orig_json must be a list, or a dict with key 'segments' (list).")


def normalize_style(label: str) -> str:
    label = (label or "").strip()
    if label in STYLE_CHOICES:
        return label
    l = label.lower()
    if "real" in l or "live" in l:
        return "realistic"
    if "cinema" in l:
        return "cinematic"
    if "comic" in l:
        return "comic"
    if "anime" in l:
        return "anime"
    if "cartoon" in l:
        return "cartoon"
    if "cgi" in l or "3d" in l:
        return "3d_cgi"
    if "stop" in l:
        return "stop_motion"
    if "pixel" in l:
        return "pixel_art"
    if "game" in l:
        return "game_render"
    return "unclear"


def clean_action(action: str) -> str:
    """action 清洗成“无主语”的动词短语，尽量短"""
    a = (action or "").strip()
    a = re.sub(r"\bC\d+\b", "", a).strip()
    a = re.sub(r"^(?:and|then)\s+", "", a, flags=re.IGNORECASE).strip()
    a = re.sub(
        r"^(?:[Tt]he\s+)?(?:person|people|man|men|woman|women|boy|boys|girl|girls|someone|"
        r"student|students|character|characters|child|children|kid|kids|minion|minions|"
        r"performer|performers|player|players|worker|workers|figure|figures|animal|animals)\s+",
        "",
        a,
    ).strip()
    a = re.sub(r"\s+", " ", a).strip()
    a = truncate_words(a, MAX_WORDS_ACTION)
    a = trim_tail_punct(a)
    return a if a else "unclear"


def clean_scene(scene: str) -> str:
    """scene 只保留地点/物体，不要出现 Cxx；并截断"""
    s = (scene or "").strip()
    s = re.sub(r"\bC\d+\b", "", s).strip()
    s = re.sub(r"\s+", " ", s).strip()
    s = truncate_words(s, MAX_WORDS_SCENE)
    s = trim_tail_punct(s)
    return s if s else "unclear"


def clean_appearance(app: str) -> str:
    """appearance 只要静态外观，且强制短"""
    a = (app or "").strip()
    a = re.sub(r"\b(is|are|being)\b.*$", "", a, flags=re.IGNORECASE).strip()
    a = re.sub(r"\s+", " ", a).strip()
    a = truncate_words(a, MAX_WORDS_APPEAR)
    a = trim_tail_punct(a)
    return a if a else "unclear"


def compact_entities(entities: List[Dict[str, Any]], max_entities: Optional[int] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for ent in entities or []:
        if not isinstance(ent, dict):
            continue
        appearance = str(ent.get("appearance", "")).strip().lower()
        ent_type = str(ent.get("type", "")).strip().lower()
        ent_id = str(ent.get("id", "")).strip()
        key = (appearance or ent_id, ent_type)
        if key in seen:
            continue
        seen.add(key)
        out.append(ent)
        if max_entities is not None and len(out) >= max_entities:
            break
    return out


def compact_entities_for_prompt(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return compact_entities(entities, MAX_PROMPT_ENTITIES)


def _normalized_text(val: Any) -> str:
    return re.sub(r"\s+", " ", str(val or "").strip())


def _normalized_entity_record(ent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(ent, dict):
        return None

    ent_type = _normalized_text(ent.get("type", "unclear")).lower() or "unclear"
    appearance_raw = _normalized_text(ent.get("appearance", "unclear"))
    appearance_raw = re.sub(
        r"\b(?:being|driving|falling|flying|gesturing|holding|jumping|kneeling|looking|lying|moving|"
        r"performing|playing|pointing|posing|reacting|riding|running|sitting|sleeping|smiling|"
        r"speaking|standing|swimming|talking|walking|waving)\b.*$",
        "",
        appearance_raw,
        flags=re.IGNORECASE,
    ).strip(" ,;:")
    appearance = clean_appearance(appearance_raw)
    style = normalize_style(_normalized_text(ent.get("style", "unclear")) or "unclear")
    gender = _normalized_text(ent.get("gender_presentation", "unclear")).lower() or "unclear"

    if appearance == "unclear" and ent_type == "unclear":
        return None

    rec: Dict[str, Any] = {
        "style": style,
        "type": ent_type,
        "gender_presentation": gender,
        "appearance": appearance,
    }
    if ent_type == "human":
        rec["ethnicity_or_race"] = _normalized_text(ent.get("ethnicity_or_race", "unclear")) or "unclear"
    return rec


def _appearance_identity_key(appearance: str) -> Tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", appearance.lower())
    kept = [
        w
        for w in words
        if w not in APPEARANCE_STOPWORDS and w not in APPEARANCE_ACTION_WORDS
    ]
    return tuple(sorted(dict.fromkeys(kept)))


def normalize_visible_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for ent in entities or []:
        rec = _normalized_entity_record(ent)
        if rec is None:
            continue
        key = (
            rec.get("appearance", "unclear"),
            rec.get("type", "unclear"),
            rec.get("style", "unclear"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def _legacy_entity_signature(ent: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    rec = _normalized_entity_record(ent)
    if rec is None:
        return None
    return (
        "|".join(_appearance_identity_key(rec.get("appearance", "unclear"))) or rec.get("appearance", "unclear"),
        rec.get("type", "unclear"),
        rec.get("style", "unclear"),
    )


def _legacy_signatures_match(lhs: Tuple[str, str, str], rhs: Tuple[str, str, str]) -> bool:
    if lhs[1:] != rhs[1:]:
        return False

    left_tokens = {tok for tok in lhs[0].split("|") if tok}
    right_tokens = {tok for tok in rhs[0].split("|") if tok}
    if not left_tokens or not right_tokens:
        return lhs[0] == rhs[0]
    if left_tokens == right_tokens:
        return True

    inter = left_tokens & right_tokens
    if len(inter) >= 3 and (left_tokens <= right_tokens or right_tokens <= left_tokens):
        return True

    union = left_tokens | right_tokens
    if union and len(inter) >= 3 and (len(inter) / len(union)) >= 0.75:
        return True
    return False


def remap_chunk_entities(seg_list: List[Dict[str, Any]], segments_obj: Any, wrapper: bool) -> List[Dict[str, Any]]:
    if not seg_list:
        if isinstance(segments_obj, dict):
            segments_obj["entities"] = []
            segments_obj["segments"] = seg_list
        return []

    if (
        wrapper
        and isinstance(segments_obj, dict)
        and isinstance(segments_obj.get("entities"), list)
        and segments_obj.get("entities")
    ):
        catalog = []
        entity_map = {}
        for ent in segments_obj.get("entities", []):
            if not isinstance(ent, dict):
                continue
            ent_id = _normalized_text(ent.get("id"))
            rec = _normalized_entity_record(ent)
            if not ent_id or rec is None:
                continue
            entity_map[ent_id] = {**rec, "id": ent_id}
            catalog.append(ent_id)

        counts: Dict[str, int] = {eid: 0 for eid in catalog}
        first_seen: Dict[str, int] = {}
        for seg_idx, seg in enumerate(seg_list):
            if not isinstance(seg, dict):
                continue
            raw_local = seg.get("entities", [])
            if not isinstance(raw_local, list):
                raw_local = []
            seg["_local_entities"] = normalize_visible_entities(raw_local)
            present_ids = seg.get("present_ids", [])
            if not isinstance(present_ids, list):
                present_ids = []
            if not present_ids:
                present_ids = [ent.get("id") for ent in seg.get("entities", []) if isinstance(ent, dict)]
            seen = []
            for ent_id in present_ids:
                ent_id = _normalized_text(ent_id)
                if ent_id and ent_id in entity_map and ent_id not in seen:
                    seen.append(ent_id)
            for ent_id in seen:
                counts[ent_id] += 1
                first_seen.setdefault(ent_id, seg_idx)

        kept_ids = [
            ent_id
            for ent_id in catalog
            if counts.get(ent_id, 0) >= MIN_RECUR_ENTITY_SEGMENTS
        ]
        kept_ids.sort(key=lambda ent_id: (-counts.get(ent_id, 0), first_seen.get(ent_id, 10**9), ent_id))
        if MAX_CHUNK_ENTITY_CATALOG > 0:
            kept_ids = kept_ids[:MAX_CHUNK_ENTITY_CATALOG]
        kept_set = set(kept_ids)

        segments_obj["entities"] = [dict(entity_map[ent_id]) for ent_id in kept_ids]
        for seg in seg_list:
            if not isinstance(seg, dict):
                continue
            present_ids = seg.get("present_ids", [])
            if not isinstance(present_ids, list):
                present_ids = []
            if not present_ids:
                present_ids = [ent.get("id") for ent in seg.get("entities", []) if isinstance(ent, dict)]
            filtered_ids = []
            for ent_id in present_ids:
                ent_id = _normalized_text(ent_id)
                if ent_id in kept_set and ent_id not in filtered_ids:
                    filtered_ids.append(ent_id)
            seg["present_ids"] = filtered_ids
            seg["entities"] = [dict(entity_map[ent_id]) for ent_id in filtered_ids]
        segments_obj["segments"] = seg_list
        return segments_obj["entities"]

    stats: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    segment_sigs: List[List[Tuple[str, str, str]]] = []
    for seg_idx, seg in enumerate(seg_list):
        seen = []
        raw_entities = seg.get("entities", []) if isinstance(seg, dict) else []
        if not isinstance(raw_entities, list):
            raw_entities = []
        if isinstance(seg, dict):
            seg["_local_entities"] = normalize_visible_entities(raw_entities)
        for ent in raw_entities:
            sig = _legacy_entity_signature(ent)
            if sig is None:
                continue
            matched_sig = None
            for existing_sig in stats:
                if _legacy_signatures_match(existing_sig, sig):
                    matched_sig = existing_sig
                    break
            if matched_sig is not None:
                sig = matched_sig
            if sig in seen:
                continue
            seen.append(sig)
            rec = _normalized_entity_record(ent)
            if rec is None:
                continue
            if sig not in stats:
                stats[sig] = {"count": 0, "first_seen": seg_idx, "record": rec}
            else:
                prev = stats[sig]["record"]
                if len(rec.get("appearance", "")) < len(prev.get("appearance", "")):
                    stats[sig]["record"] = rec
            stats[sig]["count"] += 1
            stats[sig]["first_seen"] = min(stats[sig]["first_seen"], seg_idx)
        segment_sigs.append(seen)

    kept_sigs = [
        sig
        for sig, meta in stats.items()
        if meta.get("count", 0) >= MIN_RECUR_ENTITY_SEGMENTS
    ]
    kept_sigs.sort(key=lambda sig: (-stats[sig]["count"], stats[sig]["first_seen"], sig))
    if MAX_CHUNK_ENTITY_CATALOG > 0:
        kept_sigs = kept_sigs[:MAX_CHUNK_ENTITY_CATALOG]

    sig_to_entity: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    chunk_catalog = []
    for idx, sig in enumerate(kept_sigs, 1):
        ent = dict(stats[sig]["record"])
        ent["id"] = f"C{idx:02d}"
        sig_to_entity[sig] = ent
        chunk_catalog.append(ent)

    for seg, sigs in zip(seg_list, segment_sigs):
        if not isinstance(seg, dict):
            continue
        seg_entities = [dict(sig_to_entity[sig]) for sig in sigs if sig in sig_to_entity]
        seg["entities"] = seg_entities
        if "present_ids" in seg or seg_entities:
            seg["present_ids"] = [ent["id"] for ent in seg_entities]

    if isinstance(segments_obj, dict):
        segments_obj["entities"] = chunk_catalog
        segments_obj["segments"] = seg_list
    return chunk_catalog


def compact_video_style(video_style: Optional[Dict[str, str]]) -> str:
    """把 shot/motion/light/tone 压成很短的 tag 串"""
    if not isinstance(video_style, dict):
        return ""
    shot = (video_style.get("shot_type") or "").strip()
    motion = (video_style.get("camera_motion") or "").strip()
    lighting = (video_style.get("lighting") or "").strip()
    tone = (video_style.get("tone") or "").strip()

    def _shrink(x: str) -> str:
        x = re.sub(r"\s+", " ", x).strip()
        return truncate_words(x, 2)

    tags = [_shrink(t) for t in [shot, motion, lighting, tone] if t and t != "unclear"]
    return ", ".join([t for t in tags if t])


def uppercase_first_char(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s[:1].upper() + s[1:]


def compose_no_entity_core(action: str, scene: str) -> str:
    parts = []
    if action and action != "unclear":
        parts.append(action)
    if scene and scene != "unclear":
        if parts:
            parts.append(f"in {scene}")
        else:
            parts.append(scene)
    core = " ".join(parts).strip()
    return uppercase_first_char(core or "Unclear scene")


def join_with_and(parts: List[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def compose_entity_subject(entities: List[Dict[str, Any]]) -> Tuple[str, str]:
    compacted = compact_entities(entities, None)
    total = len(compacted)
    if total == 0:
        return "", ""

    if total == 1:
        ent = compacted[0]
        cid = ent.get("id", "C??")
        app = truncate_words(clean_appearance(str(ent.get("appearance", "unclear"))), MAX_WORDS_CAPTION_APPEAR)
        app = trim_tail_punct(app)
        label = f"<{cid}> {app}" if app and app != "unclear" else f"<{cid}>"
        return label, "is"

    shown = compacted[:MAX_CAPTION_ENTITY_IDS]
    labels = [f"<{ent.get('id', 'C??')}>" for ent in shown]
    remaining = total - len(shown)
    if remaining > 0:
        labels.append(f"{remaining} more entities")
    return join_with_and(labels), "are"


def describe_plain_entity(ent: Dict[str, Any]) -> str:
    rec = _normalized_entity_record(ent) or {}
    ent_type = _normalized_text(rec.get("type", "object")).lower() or "object"
    appearance = _normalized_text(rec.get("appearance", "unclear"))

    if appearance == "unclear":
        return ent_type if ent_type != "unclear" else "object"

    human_like = {
        "human",
        "adult",
        "child",
        "man",
        "woman",
        "person",
        "boy",
        "girl",
    }
    if ent_type in human_like:
        if appearance.lower().startswith("wearing "):
            return f"{ent_type} {appearance}"
        if ent_type in appearance.lower():
            return appearance
        if " with " in appearance:
            return f"{ent_type} {appearance}"
        return f"{ent_type} with {appearance}"

    if ent_type not in {"unclear", "object", "creature"}:
        if ent_type in appearance.lower():
            return appearance
        m = re.match(r"^(.*?)(?:,?\s*with\s+(.*))?$", appearance)
        prefix = (m.group(1) if m else appearance).strip(" ,")
        suffix = (m.group(2) if m and m.group(2) else "").strip(" ,")
        if prefix and suffix:
            return f"{prefix} {ent_type} with {suffix}"
        if prefix:
            return f"{prefix} {ent_type}"
        return ent_type

    return appearance


def compose_plain_subject(entities: List[Dict[str, Any]]) -> Tuple[str, str]:
    normalized = normalize_visible_entities(entities)
    total = len(normalized)
    if total == 0:
        return "", ""

    labels = [describe_plain_entity(ent) for ent in normalized[:MAX_CAPTION_ENTITY_IDS]]
    remaining = total - len(labels)
    if remaining > 0:
        labels.append(f"{remaining} other entities")
    return join_with_and(labels), ("is" if total == 1 else "are")


def choose_entity_connector(action: str, default_verb: str) -> str:
    first = (action or "").strip().split(" ", 1)[0].lower()
    if not first or default_verb == "":
        return ""
    if first.endswith("ing") or first.endswith("ed") or first in {"being"}:
        return default_verb
    return ""


def compose_caption(
    entities: List[Dict[str, Any]],
    scene: str,
    action: str,
    visual_style: str,
    video_style: Optional[Dict[str, str]] = None,
    fallback_entities: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    一句话 caption：
    <C01> ... and <C02> ... are <action> in <scene> (style, shot, motion, light, tone).
    """
    who, verb = compose_entity_subject(entities)
    if not who and fallback_entities:
        who, verb = compose_plain_subject(fallback_entities)

    scene = clean_scene(scene)
    action = clean_action(action)
    visual_style = normalize_style(visual_style)

    vs_tag = compact_video_style(video_style)
    tag = f"{visual_style}" + (f", {vs_tag}" if vs_tag else "")

    if who:
        connector = choose_entity_connector(action, verb)
        if connector:
            core = f"{who} {connector} {action} in {scene}"
        else:
            core = f"{who} {action} in {scene}"
        core = re.sub(r"\s+", " ", core).strip()
        core = uppercase_first_char(core)
    else:
        core = compose_no_entity_core(action, scene)

    tag_words = len(tag.split()) if tag else 0
    max_core_words = max(8, MAX_WORDS_CAPTION - tag_words)
    core = truncate_words(core, max_core_words)

    cap = f"{core} ({tag})."
    cap = re.sub(r"\s+", " ", cap).replace("..", ".").strip()
    if not cap.endswith("."):
        cap = cap.rstrip(",;") + "."
    return cap


def build_prompt(entities: List[Dict[str, Any]]) -> str:
    entity_lines = []
    for idx, e in enumerate(entities or [], 1):
        entity_id = e.get("id", "") or f"E{idx:02d}"
        entity_lines.append(f"- {entity_id}: appearance={e.get('appearance','unclear')}")
    entity_block = "\n".join(entity_lines) if entity_lines else "- (none)"

    choices = ", ".join(STYLE_CHOICES)

    return f"""
You are annotating a short video segment for dataset captions.

Known entities (fixed; DO NOT restate them in action/scene):
{entity_block}

Return STRICT JSON ONLY:
{{
  "scene": "<= {MAX_WORDS_SCENE} words, noun phrase only, NO actions, NO IDs",
  "action": "<= {MAX_WORDS_ACTION} words, verb phrase only, NO subject, NO IDs",
  "visual_style": "one label from [{choices}]",
  "video_style": {{
    "shot_type": "one of: close-up/medium/long/over-the-shoulder/unclear",
    "camera_motion": "one of: static/pan/tilt/handheld/zoom/dolly/unclear",
    "lighting": "one of: daylight/indoor_warm/night/low_key/backlit/unclear",
    "tone": "one of: cinematic/home-video/dramatic/comedic/neutral/unclear"
  }}
}}

Rules:
- scene: location/setting/objects only (NO verbs; NO entity IDs).
- action: start with a verb phrase (no subject), e.g. "being held in snow", "sleeping on cushion".
- Be concise. No repetition. Only visible facts. No backstory. If unsure use "unclear".
""".strip()


def _list_sorted_clips(dir_path: str) -> List[str]:
    """返回按数字文件名排序的 mp4 列表：0000.mp4, 0001.mp4..."""
    mp4s = glob.glob(os.path.join(dir_path, "*.mp4"))
    items = []
    for p in mp4s:
        stem = os.path.splitext(os.path.basename(p))[0]
        if re.fullmatch(r"\d+", stem):
            items.append((int(stem), p))
        else:
            # 非纯数字名放后面（很少见）
            items.append((10**18, p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def find_clip_by_index(clips_dir: str, seg_idx: int) -> Optional[str]:
    """严格按 seg_idx -> 0000.mp4 的规则找"""
    p = os.path.join(clips_dir, f"{seg_idx:04d}.mp4")
    if os.path.isfile(p):
        return p
    p2 = os.path.join(clips_dir, f"{seg_idx}.mp4")
    if os.path.isfile(p2):
        return p2
    ms = sorted(glob.glob(os.path.join(clips_dir, f"{seg_idx:04d}*.mp4")))
    return ms[0] if ms else None


def process_vision_info_compat(messages, patch_size: Optional[int] = None):
    if _PVI_HAS_IMAGE_PATCH_SIZE:
        kwargs = {}
        if patch_size is not None:
            kwargs["image_patch_size"] = patch_size
        if _PVI_HAS_RETURN_VIDEO_KWARGS:
            kwargs["return_video_kwargs"] = True
        if _PVI_HAS_RETURN_VIDEO_METADATA:
            kwargs["return_video_metadata"] = True

        res = process_vision_info(messages, **kwargs)
        if isinstance(res, tuple) and len(res) == 3:
            images, videos, video_kwargs = res
        elif isinstance(res, tuple) and len(res) == 2:
            images, videos = res
            video_kwargs = {}
        else:
            raise TypeError("process_vision_info returned an unexpected value")
        return images, videos, (video_kwargs or {})

    if patch_size is None:
        patch_size = 16

    vision_infos = extract_vision_info(messages)
    image_inputs = []
    video_inputs = []
    video_sample_fps_list = []
    video_metadata_list = []
    for vision_info in vision_infos:
        if "image" in vision_info or "image_url" in vision_info:
            image_inputs.append(fetch_image(vision_info, size_factor=patch_size))
        elif "video" in vision_info:
            video_input, sample_fps = fetch_video(
                vision_info,
                image_factor=patch_size,
                return_video_sample_fps=True,
            )
            video_inputs.append(video_input)
            video_sample_fps_list.append(sample_fps)
            if isinstance(video_input, list):
                num_frames = len(video_input)
            else:
                num_frames = int(getattr(video_input, "shape", [0])[0])
            if num_frames > 0:
                video_metadata_list.append(
                    {
                        "total_num_frames": num_frames,
                        "fps": float(sample_fps) if sample_fps is not None else None,
                        "frames_indices": list(range(num_frames)),
                    }
                )
        else:
            raise ValueError("image, image_url or video should in content.")

    if not image_inputs:
        image_inputs = None
    if not video_inputs:
        video_inputs = None
    video_kwargs = {"fps": video_sample_fps_list} if video_sample_fps_list else {}
    if video_metadata_list:
        video_kwargs["video_metadata"] = video_metadata_list if len(video_metadata_list) > 1 else video_metadata_list[0]
    return image_inputs, video_inputs, video_kwargs


@torch.no_grad()
def infer_scene_action_style(
    model,
    processor,
    video_path: str,
    entities: List[Dict[str, Any]],
) -> Tuple[str, str, str, Dict[str, str], str]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, **VIDEO_CFG},
            {"type": "text", "text": build_prompt(entities)},
        ],
    }]

    # HF chat template：add_generation_prompt 用于追加“开始生成”的提示 token（按 tokenizer 模板决定）
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    patch_size = None
    if hasattr(processor, "video_processor") and hasattr(processor.video_processor, "patch_size"):
        patch_size = processor.video_processor.patch_size
    elif hasattr(processor, "image_processor") and hasattr(processor.image_processor, "patch_size"):
        patch_size = processor.image_processor.patch_size

    images, videos, video_kwargs = process_vision_info_compat(messages, patch_size=patch_size)
    if isinstance(video_kwargs, dict) and "fps" in video_kwargs:
        fps_val = video_kwargs.get("fps")
        if isinstance(fps_val, list):
            if len(fps_val) == 1:
                video_kwargs["fps"] = fps_val[0]
            elif len(fps_val) == 0:
                video_kwargs.pop("fps", None)
    video_metadatas = None
    if _PVI_HAS_RETURN_VIDEO_METADATA and videos is not None:
        try:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        except ValueError:
            video_metadatas = None

    inputs_kwargs = dict(
        text=text,
        images=images,
        videos=videos,
        return_tensors="pt",
        do_resize=False,
    )
    if isinstance(video_kwargs, dict):
        inputs_kwargs.update(video_kwargs)
    inputs_kwargs.setdefault("do_sample_frames", False)
    if not inputs_kwargs.get("do_sample_frames"):
        inputs_kwargs.pop("fps", None)
    if video_metadatas is not None:
        inputs_kwargs["video_metadata"] = video_metadatas

    inputs = processor(**inputs_kwargs).to(model.device)

    inputs.pop("token_type_ids", None)

    out_ids = model.generate(**inputs, max_new_tokens=192, do_sample=False)
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(gen_only, skip_special_tokens=True)[0].strip()

    obj = safe_json_obj(raw)
    if not isinstance(obj, dict):
        return (
            "unclear",
            "unclear",
            "unclear",
            {"shot_type": "unclear", "camera_motion": "unclear", "lighting": "unclear", "tone": "unclear"},
            raw,
        )

    scene = clean_scene(str(obj.get("scene", "unclear")))
    action = clean_action(str(obj.get("action", "unclear")))
    visual_style = normalize_style(str(obj.get("visual_style", "unclear")))

    vs = obj.get("video_style", {})
    if not isinstance(vs, dict):
        vs = {}
    video_style = {
        "shot_type": str(vs.get("shot_type", "unclear")).strip(),
        "camera_motion": str(vs.get("camera_motion", "unclear")).strip(),
        "lighting": str(vs.get("lighting", "unclear")).strip(),
        "tone": str(vs.get("tone", "unclear")).strip(),
    }

    return scene, action, visual_style, video_style, raw


# ----------------------------
# 4) 主流程
# ----------------------------
def _normalize_filter(val) -> Optional[set]:
    if val is None:
        return None
    if isinstance(val, (list, tuple, set)):
        return {str(v) for v in val}
    return {str(val)}


def _normalize_filter_env(name: str) -> Optional[set]:
    items = set()

    raw = os.environ.get(name, "").strip()
    if raw:
        items.update(item.strip() for item in raw.split(",") if item.strip())

    file_path = os.environ.get(f"{name}_FILE", "").strip()
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            items.update(line.strip() for line in f if line.strip())

    return items or None


def _case_sort_key(name: str) -> Tuple[int, int, str]:
    if name.isdigit():
        return (0, int(name), "")
    return (1, 0, name)


def _chunk_sort_key(name: str) -> Tuple[int, int, str]:
    m = re.search(r"chunk(\d+)$", name)
    if m:
        return (0, int(m.group(1)), "")
    return (1, 0, name)


def expand_segments_to_clip_count(seg_list: List[Dict[str, Any]], clip_count: int) -> List[Dict[str, Any]]:
    if clip_count <= 0 or not seg_list or len(seg_list) == clip_count:
        return seg_list

    expanded: List[Dict[str, Any]] = []
    src_count = len(seg_list)
    for idx in range(clip_count):
        src_idx = min((idx * src_count) // clip_count, src_count - 1)
        expanded.append(copy.deepcopy(seg_list[src_idx]))
    return expanded


def iter_chunk_tasks(
    dataset_root: str,
    captions_root: str,
    case_filter: Optional[set],
    chunk_filter: Optional[set],
):
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset_root not found: {dataset_root}")
    if not os.path.isdir(captions_root):
        raise FileNotFoundError(f"captions_root not found: {captions_root}")

    case_ids = [
        d
        for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d)) and not d.startswith(".")
    ]
    case_ids.sort(key=_case_sort_key)

    for case_id in case_ids:
        if case_filter and case_id not in case_filter:
            continue
        case_dir = os.path.join(dataset_root, case_id)
        vlm_dir = os.path.join(case_dir, "vlm")
        if not os.path.isdir(vlm_dir):
            print(f"[WARN] missing vlm dir: {vlm_dir}")
            continue

        caption_case_dir = os.path.join(captions_root, case_id, "vlm")
        if not os.path.isdir(caption_case_dir):
            print(f"[WARN] missing captions dir: {caption_case_dir}")
            continue

        chunk_dirs = [
            d
            for d in os.listdir(vlm_dir)
            if os.path.isdir(os.path.join(vlm_dir, d)) and not d.startswith(".")
        ]
        chunk_dirs.sort(key=_chunk_sort_key)

        for chunk_name in chunk_dirs:
            if chunk_filter and chunk_name not in chunk_filter:
                continue
            clips_dir = os.path.join(vlm_dir, chunk_name)
            orig_json_path = os.path.join(caption_case_dir, f"{chunk_name}.json")
            if not os.path.isfile(orig_json_path):
                alt_raw = os.path.join(caption_case_dir, f"{chunk_name}.raw.txt")
                alt_txt = os.path.join(caption_case_dir, f"{chunk_name}.txt")
                if os.path.isfile(alt_raw):
                    orig_json_path = alt_raw
                elif os.path.isfile(alt_txt):
                    orig_json_path = alt_txt
                else:
                    print(f"[WARN] missing caption json/txt: {orig_json_path}")
                    continue
            yield case_id, chunk_name, clips_dir, orig_json_path


def process_one_chunk(model, processor, clips_dir: str, orig_json_path: str):
    if not os.path.isfile(orig_json_path):
        raise FileNotFoundError(f"orig_json_path not found: {orig_json_path}")
    if not os.path.isdir(clips_dir):
        raise ValueError(f"clips_dir is not a directory: {clips_dir}")

    segments, seg_list, wrapper = load_segments_from_file(orig_json_path)

    # 备份
    bak_path = orig_json_path + ".bak"
    if not os.path.exists(bak_path):
        with open(bak_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        print(f"[INFO] backup saved: {bak_path}")

    # 准备 clip 映射
    sorted_clips = _list_sorted_clips(clips_dir) if USE_SORTED_CLIPS_MAPPING else []
    if USE_SORTED_CLIPS_MAPPING:
        print(f"[INFO] clips_dir={clips_dir}")
        print(f"[INFO] clips found={len(sorted_clips)} (sorted by numeric filename)")
    else:
        print(f"[INFO] clips_dir={clips_dir}")
        print("[INFO] mapping mode: strict index -> {i:04d}.mp4")

    clip_count = len(sorted_clips) if USE_SORTED_CLIPS_MAPPING else len(_list_sorted_clips(clips_dir))
    if len(seg_list) < clip_count and seg_list:
        kind = "compact" if wrapper else "list"
        print(
            f"[WARN] {kind} segments shorter than clips: segments={len(seg_list)} clips={clip_count}; "
            "expanding by proportional repeat"
        )
        seg_list = expand_segments_to_clip_count(seg_list, clip_count)
        if isinstance(segments, dict):
            segments["segments"] = seg_list
        else:
            segments = seg_list
    elif len(seg_list) < clip_count:
        print(f"[WARN] segments shorter than clips: segments={len(seg_list)} clips={clip_count}; padding placeholders")
        seg_list.extend({"entities": []} for _ in range(clip_count - len(seg_list)))
        if isinstance(segments, list):
            segments = seg_list
    elif len(seg_list) > clip_count:
        kind = "compact" if wrapper else "list"
        print(f"[WARN] {kind} segments longer than clips: segments={len(seg_list)} clips={clip_count}; trimming tail")
        seg_list = seg_list[:clip_count]
        if isinstance(segments, dict):
            segments["segments"] = seg_list
        else:
            segments = seg_list

    if os.path.exists(bak_path):
        try:
            _, backup_seg_list, _ = load_segments_from_file(bak_path)
            if len(backup_seg_list) < clip_count and backup_seg_list:
                backup_seg_list = expand_segments_to_clip_count(backup_seg_list, clip_count)
            elif len(backup_seg_list) > clip_count:
                backup_seg_list = backup_seg_list[:clip_count]

            restored = 0
            for seg, bak_seg in zip(seg_list, backup_seg_list):
                if not isinstance(seg, dict) or not isinstance(bak_seg, dict):
                    continue
                cur_entities = seg.get("entities", [])
                bak_entities = bak_seg.get("entities", [])
                if (not isinstance(cur_entities, list) or not cur_entities) and isinstance(bak_entities, list) and bak_entities:
                    seg["entities"] = copy.deepcopy(bak_entities)
                    restored += 1
            if restored:
                print(f"[INFO] restored local entities from backup for {restored} segments")
        except Exception as exc:
            print(f"[WARN] failed to load backup entities from {bak_path}: {exc}")

    chunk_catalog = remap_chunk_entities(seg_list, segments, wrapper)
    print(f"[INFO] recurring chunk entities={len(chunk_catalog)}")
    print(f"[INFO] segments={len(seg_list)}")

    updated = 0
    missing = 0

    for i, seg in enumerate(seg_list):
        if not isinstance(seg, dict):
            print(f"[WARN] seg{i:04d} is not a dict, skipped")
            continue
        if USE_SORTED_CLIPS_MAPPING:
            clip_path = sorted_clips[i] if i < len(sorted_clips) else None
        else:
            clip_path = find_clip_by_index(clips_dir, i)

        if clip_path is None or not os.path.isfile(clip_path):
            missing += 1
            if USE_SORTED_CLIPS_MAPPING:
                print(f"[MISS] seg{i:04d} no clip at sorted index {i}")
            else:
                print(f"[MISS] seg{i:04d} no clip matched: {i:04d}.mp4")
            continue

        local_entities = seg.pop("_local_entities", [])
        if not isinstance(local_entities, list):
            local_entities = []
        entities = seg.get("entities", [])
        if not isinstance(entities, list):
            entities = []
        prompt_entities = entities if entities else local_entities
        prompt_entities = compact_entities_for_prompt(prompt_entities)

        scene, action, visual_style, video_style, raw = infer_scene_action_style(
            model, processor, clip_path, prompt_entities
        )

        seg["scene"] = scene
        seg["action"] = action
        seg["visual_style"] = visual_style
        seg["video_style"] = video_style
        seg["caption"] = compose_caption(
            entities,
            scene,
            action,
            visual_style,
            video_style=video_style,
            fallback_entities=local_entities,
        )
        seg["qwen_raw"] = raw
        seg["clip_path"] = clip_path  # 额外记录一下来源 clip，排查更方便

        updated += 1
        print(f"[OK] seg{i:04d} file={os.path.basename(clip_path)} style={visual_style} -> {seg['caption']}")

    with open(orig_json_path, "w", encoding="utf-8") as f:
        json.dump(segments if wrapper else seg_list, f, ensure_ascii=False, indent=2)

    print(f"[DONE] updated={updated}, missing_clips={missing}, wrote_back={orig_json_path}")


def main():
    if not DATASET_ROOT:
        raise ValueError("DATASET_ROOT is required")
    if not CAPTIONS_ROOT:
        raise ValueError("CAPTIONS_ROOT is required")
    if SHARD_COUNT < 1:
        raise ValueError(f"SHARD_COUNT must be >= 1, got {SHARD_COUNT}")
    if not 0 <= SHARD_INDEX < SHARD_COUNT:
        raise ValueError(f"SHARD_INDEX must satisfy 0 <= index < count, got index={SHARD_INDEX}, count={SHARD_COUNT}")

    case_filter = _normalize_filter_env("CASE_FILTER")
    if case_filter is None:
        case_filter = _normalize_filter(CASE_FILTER)
    chunk_filter = _normalize_filter_env("CHUNK_FILTER")
    if chunk_filter is None:
        chunk_filter = _normalize_filter(CHUNK_FILTER)

    all_tasks = list(iter_chunk_tasks(DATASET_ROOT, CAPTIONS_ROOT, case_filter, chunk_filter))
    tasks = [task for idx, task in enumerate(all_tasks) if idx % SHARD_COUNT == SHARD_INDEX]
    if not tasks:
        print("[INFO] no tasks found.")
        return

    print(f"[INFO] dataset_root={DATASET_ROOT}")
    print(f"[INFO] captions_root={CAPTIONS_ROOT}")
    print(f"[INFO] shard={SHARD_INDEX}/{SHARD_COUNT}")
    print(f"[INFO] total tasks={len(all_tasks)}")
    print(f"[INFO] shard tasks={len(tasks)}")
    print("[INFO] loading model...")
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("[INFO] model loaded.")

    for idx, (case_id, chunk_name, clips_dir, orig_json_path) in enumerate(tasks, 1):
        print(f"\n[TASK {idx}/{len(tasks)}] case={case_id} chunk={chunk_name}")
        try:
            process_one_chunk(model, processor, clips_dir, orig_json_path)
        except Exception as e:
            print(f"[ERROR] case={case_id} chunk={chunk_name} failed: {e}")
            if not CONTINUE_ON_ERROR:
                raise


if __name__ == "__main__":
    main()
