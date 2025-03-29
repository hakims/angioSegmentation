# 🩺 Vascular Labeling Guidelines (Schema 2025.03)

This guide outlines how to label vascular structures in angiographic images using the CVAT schema `cvat_vascular_schema_2025.03.json`.

---

## 📦 Categories

| Name    | Description                             |
|---------|-----------------------------------------|
| vessel  | Arteries, AVFs, and surgical bypasses   |
| bone    | Bony structures (femur, pelvis, etc.)   |
| device  | Stents, wires, sheaths, etc.            |

---

## 🔖 Key Attributes for `vessel`

| Attribute        | Description |
|------------------|-------------|
| `vessel_id`      | Canonical structure name (e.g., SFA, pop, AVF, bypass) |
| `side`           | "L", "R", "B", or "N/A" for laterality |
| `segment`        | Optional: "proximal", "mid", "distal", "AK", "BK" |
| `aortic_zone`    | 0–11 (only for aorta and devices) |
| `anomaly_type`   | "none", "stenosis", "aneurysm", "dissection", "occlusion" |
| `modality`       | Imaging type — default is "DSA" |
| `presence_only`  | Check if label is non-segmented flag |
| `annotator`      | User/model label source |
| `label_schema_version` | Fixed: "2025.03" |

---

## 🩺 Special Cases

### ✅ AVFs
Use `vessel_id: "AVF"` and optionally specify:
- `avf_type`: "brachiocephalic", "radiocephalic", "brachiobasilic"
- `conduit`: "vein", "PTFE graft", etc.

### ✅ Bypasses
Use `vessel_id: "bypass"` and set:
- `bypass_type`: "fem-pop", "ax-bifem", etc.
- `conduit`: "vein", "PTFE graft", "Dacron"

---

## 💡 Example Annotations

### Healthy CFA
```json
{ "vessel_id": "CFA", "side": "R", "anomaly_type": "none" }
```

### Distal SFA stenosis
```json
{ "vessel_id": "SFA", "side": "R", "segment": "distal", "anomaly_type": "stenosis" }
```

### SFA-to-pop bypass
```json
{ "vessel_id": "bypass", "bypass_type": "fem-pop", "conduit": "PTFE graft" }
```

### AVF with aneurysm
```json
{ "vessel_id": "AVF", "side": "L", "anomaly_type": "aneurysm", "avf_type": "radiocephalic" }
```

---

## 🧪 Tips

- Always assign `vessel_id`, `side`, and `anomaly_type`.
- Use `segment`, `aortic_zone`, and `conduit` **only when relevant**.
- Set `presence_only: true` if no segmentation is drawn.