# 🩺 Vascular Labeling Guidelines (Schema v2025.03)

This guide describes how to use the CVAT labeling schema `cvat_labels_array_final_conduits_2025.03.json` for structured annotation of vascular imaging, including vessels, bones, and devices.

---

## 🔖 Label Categories

| Label       | Description                                        |
|-------------|----------------------------------------------------|
| `vessel`    | All native arteries, AVFs, and bypass grafts       |
| `bone`      | Bones appearing in angiographic or X-ray images    |
| `device`    | Implanted or procedural tools (stents, grafts, etc)|

---

## 🩺 Vessel Label Attributes

| Attribute            | Type      | Description |
|----------------------|-----------|-------------|
| `vessel_id`          | select    | Name of the vessel or structure (e.g. `SFA`, `AVF`, `bypass`) |
| `side`               | select    | Laterality: `L`, `R`, `B` (bilateral), or `N/A` |
| `segment`            | select    | Optional: `proximal`, `mid`, `distal`, `AK`, `BK` |
| `aortic_zone`        | select    | SVS zones 0–11 (only use if vessel is the aorta) |
| `anomaly_type`       | select    | `none`, `stenosis`, `aneurysm`, `dissection`, `occlusion` |
| `modality`           | select    | Imaging modality: `DSA` (default), `CO2-DSA`, `CT`, `MRA`, `X-ray` |
| `avf_type`           | select    | Optional for `AVF`: `brachiocephalic`, `radiocephalic`, etc. |
| `conduit`            | select    | For `AVF` or `bypass`: `vein`, `PTFE graft`, `Dacron`, `cryo`, `unknown` |
| `bypass_type`        | select    | For `bypass`: `fem-pop`, `aorto-bifem`, `ax-bifem`, etc. |
| `annotator`          | select    | Defaults to `human`; can be `DrSAM`, `AI` |
| `label_schema_version` | select | Fixed: `2025.03` |

---

## 🦴 Bone Label Attributes

| Attribute  | Description |
|------------|-------------|
| `bone_id`  | Bone name: `femur`, `pelvis`, `tibia`, `fibula`, `radius`, `ulna`, `humerus`, `calcaneus` |
| `side`     | Laterality |

---

## ⚙️ Device Label Attributes

| Attribute         | Description |
|------------------|-------------|
| `device_type`     | Type of device (e.g., `stent`, `catheter`, `EVAR`, `balloon`, etc.) |
| `side`            | Device laterality if applicable |
| `aortic_zone`     | SVS zone if relevant |
| `modality`        | Imaging modality |
| `deployment_zone` | Optional zone text (e.g., `2–4`, `bifurcation`) |

---

## 💡 Labeling Tips

- Only apply `aortic_zone` when `vessel_id = aorta` or device spans zones.
- Use `bypass_type` and `conduit` only when `vessel_id = bypass`.
- AVFs should be labeled as `vessel_id = AVF` and may include `avf_type` and `conduit`.
- You do **not** need to fill every attribute — leave irrelevant fields blank.

---

## 📤 Export & Usage

- Export labels in **COCO 1.0** format from CVAT.
- All attributes are embedded in the `annotations[*].attributes` block.
- Metadata supports downstream parsing for segmentation, anomaly classification, and clinical filtering.