"""Add an explicit weights_state field to each model record.

Detecting "in repo / external / none" from free-text prose is fragile once the
text is translated. This sets the field authoritatively, right after the
"id" key, so the UI never has to guess.
"""

import json
import collections

STATE = {
    "nnunet": "external",
    "nnunet-bra21-kaist": "yes",
    "monai-brats-bundle": "external",
    "monai-swinunetr-brats": "yes",
    "brainsegfounder": "yes",
    "neuroxai-models": "yes",
    "gradcam-lib": "no",
    "captum-lib": "no",
    "lgg-unet": "yes",
    "brisc-baselines": "no",
    "brisc-swin-hafunet": "no",
    "medsam": "yes",
    "medsam2": "yes",
    "aimi-brain-mr": "yes",
    "aimi-lung-ct": "yes",
    "aimi-other": "yes",
    "transunet": "yes",
    "unetr": "yes",
    "segformer": "yes",
    "imagenet-backbones": "yes",
    "convolutional-transformer-hybrid": "no",
    "sam-med3d": "no",
    "oncopilot": "no",
    "unet-variants": "no",
    "neuroxai-lib": "yes",
    "shap-lib": "no",
    "tcav-lib": "no",
    "scorecam-lib": "no",
}

PATH = "tools/chunks/01-models.json"

with open(PATH, encoding="utf-8-sig") as f:
    models = json.load(f, object_pairs_hook=collections.OrderedDict)

missing = [m["id"] for m in models if m["id"] not in STATE]
if missing:
    raise SystemExit("No weights_state defined for: " + ", ".join(missing))

out = []
for m in models:
    new = collections.OrderedDict()
    for k, v in m.items():
        new[k] = v
        if k == "id":
            new["weights_state"] = STATE[m["id"]]
    out.append(new)

with open(PATH, "w", encoding="utf-8", newline="") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
    f.write("\n")

counts = collections.Counter(STATE.values())
print("weights_state written for", len(out), "models:", dict(counts))
