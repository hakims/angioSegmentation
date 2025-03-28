import zipfile
from pathlib import Path
from datetime import datetime

# Exact top-level folders to include
INCLUDE_DIRS = {
    "bin", "preprocessing", "segmentation", "utils"
}

# Specific subfolders from dr_sam_core
DR_SAM_SUBDIRS = {
    "dr_sam_core/anomality_detection",
    "dr_sam_core/segmentation"
}

# Common excludes
EXCLUDES = {"__pycache__", ".ipynb_checkpoints"}
EXCLUDE_EXTS = {".log", ".tmp", ".bak"}

def should_include(path: Path, root: Path):
    rel_path = path.relative_to(root)
    parts = rel_path.parts

    # Top-level folders (bin, segmentation, etc.)
    if parts[0] in INCLUDE_DIRS:
        return not any(p in EXCLUDES for p in parts) and path.suffix not in EXCLUDE_EXTS

    # Specific subfolders from dr_sam_core
    for allowed in DR_SAM_SUBDIRS:
        if str(rel_path).startswith(allowed):
            return not any(p in EXCLUDES for p in parts) and path.suffix not in EXCLUDE_EXTS

    return False

def zip_project(output_name=None):
    project_root = Path(__file__).resolve().parent
    date_str = datetime.now().strftime("%Y-%m-%d")
    archive_name = output_name or f"AngioMLM_bundle_{date_str}.zip"
    archive_path = project_root / archive_name

    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in project_root.rglob("*"):
            if file.is_file() and should_include(file, project_root):
                zipf.write(file, arcname=file.relative_to(project_root))

    print(f"✅ Project packaged to: {archive_path}")

if __name__ == "__main__":
    zip_project()
