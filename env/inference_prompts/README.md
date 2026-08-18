# Inference Prompt Library

This directory stores separate inference cases for StoryPack-16 and StoryPack-64:

- `16grid/`: six short prompts with eight numbered shots each.
- `64grid/`: six long prompts with 18–22 numbered shots each.

Pass one of these prompt files to the portable inference wrapper:

```bash
WAN_REPO=Wan2.2 \
WAN_MODEL_DIR=checkpoints/Wan2.2-TI2V-5B \
bash scripts/infer.sh \
  checkpoints/StoryPack/storypack-16 \
  env/inference_prompts/16grid/stopmotion_knitted_toys_clock_short.txt \
  outputs/stopmotion
```

StoryPack-16 cases:

- `3dcgi_boy_robot_cat_fair_short`
- `anime_rooftop_lanterns_short`
- `cinematic_hospital_airport_short`
- `cinematic_prison_office_short`
- `realistic_river_farmland_aerial_short`
- `stopmotion_knitted_toys_clock_short`

StoryPack-64 cases:

- `cinematic_prison_office`
- `3dcgi_boy_robot_cat_moonlit_fair`
- `stopmotion_knitted_penguin_mouse_clocktower`
- `cinematic_neural_courier_hospital_airport`
- `realistic_river_farmland_terracotta_aerial`
- `anime_neon_fireworks_school_escape`

List available prompts with:

```bash
bash scripts/list_prompts.sh
```
