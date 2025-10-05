import os
import json
import zipfile

def build_image_lookup(zip_path):
    """
    Builds a mapping from image filename (without subdir) to full path in the zip.
    Returns a dict: {basename: full_path_in_zip}
    """
    lookup = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.is_dir():
                continue
            basename = os.path.basename(file_info.filename)
            lookup[basename] = file_info.filename
    return lookup

def fix_eval_image_paths_with_zip(jsonl_path, zip_path, output_jsonl_path):
    """
    Fixes image paths in a Web2Code eval JSONL file by searching the zip for the correct path.
    """
    image_lookup = build_image_lookup(zip_path)
    fixed_count = 0
    missing_count = 0
    with open(jsonl_path, 'r', encoding='utf-8') as infile, \
         open(output_jsonl_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if not line.strip():
                continue
            entry = json.loads(line)
            image_name = entry.get("image", None)
            if image_name:
                fixed_path = image_lookup.get(os.path.basename(image_name))
                if fixed_path:
                    if entry["image"] != fixed_path:
                        entry["image"] = fixed_path
                        fixed_count += 1
                else:
                    missing_count += 1
            outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"Fixed {fixed_count} image paths.")
    if missing_count > 0:
        print(f"Warning: {missing_count} images could not be found in the zip.")

if __name__ == "__main__":
    """
    Run this script to fix image paths in the Web2Code eval JSONL file.
    """
    input_jsonl = os.path.join("Web2Code", "Web2Code_eval.jsonl")
    output_jsonl = os.path.join("Web2Code", "Web2Code_eval_fixed.jsonl")
    zip_path = os.path.join("Web2Code", "Web2Code_image_eval.zip")

    fix_eval_image_paths_with_zip(input_jsonl, zip_path, output_jsonl)