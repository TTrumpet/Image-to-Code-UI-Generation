import os
import json
import zipfile
from PIL import Image

def print_zip_structure(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        folders = set()
        for file_info in zip_ref.infolist():
            path = file_info.filename
            # Remove trailing slash for directories
            if path.endswith('/'):
                folders.add(path.rstrip('/'))
            else:
                # Add parent directories of files
                parent = os.path.dirname(path)
                if parent:
                    folders.add(parent)
        for folder in sorted(folders):
            print(folder)

class Web2CodeDataLoader:
    def __init__(self, json_path, zip_path, extract_dir="extracted_images"):
        self.json_path = json_path
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.data = self._load_json()
        self._extract_images()  # Only extract, do not load into memory

    def _load_json(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_images(self):
        # Extract images if not already extracted
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir)
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)

    def get_data(self):
        return self.data

    def get_image(self, image_name):
        image_path = os.path.join(self.extract_dir, image_name)
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            raise FileNotFoundError(f"Image {image_name} not found in extracted directory.")

    def get_entry_with_image(self, idx):
        entry = self.data[idx]
        image_name = entry.get("image", None)
        pil_image = None
        if image_name:
            try:
                pil_image = self.get_image(image_name)
            except Exception as e:
                print(f"Could not open image {image_name}: {e}")
        return entry, pil_image

class Web2CodeEvalDataLoader:
    def __init__(self, jsonl_path, zip_path, extract_dir="extracted_images_eval"):
        self.jsonl_path = jsonl_path
        self.zip_path = zip_path
        self.extract_dir = extract_dir
        self.data = self._load_jsonl()
        self._extract_images()

    def _load_jsonl(self):
        data = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def _extract_images(self):
        if not os.path.exists(self.extract_dir):
            os.makedirs(self.extract_dir)
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)

    def get_data(self):
        return self.data

    def get_image(self, image_name):
        image_path = os.path.join(self.extract_dir, image_name)
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            raise FileNotFoundError(f"Image {image_name} not found in extracted directory.")

    def get_entry_with_image(self, idx):
        entry = self.data[idx]
        image_name = entry.get("image", None)
        pil_image = None
        if image_name:
            try:
                pil_image = self.get_image(image_name)
            except Exception as e:
                print(f"Could not open image {image_name}: {e}")
        return entry, pil_image

def dummy_model(conversation, image):
    # Placeholder for your model logic
    print(f"Model received conversation: {conversation[:60]}...")  # Print first 60 chars
    print(f"Model received image: {type(image)}")

def check_missing_images(data, extract_dir, image_key="image"):
    """
    Checks for missing images referenced in the data.
    Returns a list of (index, image_name) tuples for missing images.
    """
    missing_images = []
    for idx, entry in enumerate(data):
        image_name = entry.get(image_key, None)
        if image_name:
            image_path = os.path.join(extract_dir, image_name)
            if not os.path.exists(image_path):
                missing_images.append((idx, image_name))
    return missing_images

if __name__ == "__main__":
    # Load Web2Code train set
    zip_path = os.path.join("Web2Code", "Web2Code_image.zip")
    data_path = os.path.join("Web2Code", "Web2Code.json")
    loader = Web2CodeDataLoader(data_path, zip_path)
    data = loader.get_data()
    print(f"Loaded {len(data)} entries from JSON.")

    # Check for missing images in train set
    missing_images = check_missing_images(data, loader.extract_dir)
    if missing_images:
        print(f"\nMissing images ({len(missing_images)}):")
        for idx, image_name in missing_images:
            print(f"Entry {idx}: {image_name}")
    else:
        print("\nAll referenced images are present.")
    
    # Print first 2 entries to test
    for i in range(2):  
        entry, pil_image = loader.get_entry_with_image(i)
        print(json.dumps({k: v for k, v in entry.items() if k != "pil_image"}, indent=2))
        print(f"pil_image type: {type(pil_image)}")

    # Iterate through all data and pass to dummy model
    for idx in range(len(data)):
        entry, pil_image = loader.get_entry_with_image(idx)
        conversation = entry.get("conversation", "")
        # Pass conversation and image to your model (placeholder)
        dummy_model(conversation, pil_image)

    '''
    # Load Web2Code eval data
    eval_zip_path = os.path.join("Web2Code", "Web2Code_image_eval.zip")
    eval_jsonl_path = os.path.join("Web2Code", "Web2Code_eval_fixed.jsonl")
    eval_loader = Web2CodeEvalDataLoader(eval_jsonl_path, eval_zip_path)
    eval_data = eval_loader.get_data()
    print(f"Loaded {len(eval_data)} eval entries from JSONL.")

    # Check for missing images in eval set
    missing_eval_images = check_missing_images(eval_data, eval_loader.extract_dir)
    if missing_eval_images:
        print(f"\nMissing eval images ({len(missing_eval_images)}):")
        #for idx, image_name in missing_eval_images:
        #    print(f"Entry {idx}: {image_name}")
    else:
        print("\nAll referenced eval images are present.")

    # Print first 2 entries to test
    for i in range(2):
        entry, pil_image = eval_loader.get_entry_with_image(i)
        print(json.dumps(entry, indent=2))
        print(f"pil_image type: {type(pil_image)}")
    '''