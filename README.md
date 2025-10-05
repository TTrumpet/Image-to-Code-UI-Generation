# Image-to-Code-UI-Generation
Multimodal AI Agent for Image-to-Code UI Generation - CAP 6411 Computer Vision Systems Group Project

# Setup
Download Web2Code dataset manually.
```
conda create -n Image2Code python=3.10
pip install -r requirements.txt
```
# Training
See dataloader.py, Web2CodeDataLoader. This function will iterate through the json data and zipped images.

# Evaluation
Run fix_eval_image_paths.py to generate Web2Code_eval_fixed.jsonl.
See dataloader.py, Web2CodeEvalDataLoader. This function will iterate through the jsonl data and zipped images.