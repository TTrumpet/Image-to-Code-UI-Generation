import os
import re
import json
import base64
import mimetypes
import argparse
import pathlib
import io
import requests
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple, Annotated # <-- MODIFIED
from dataclasses import dataclass, field
from io import BytesIO
from PIL import Image
from uuid import uuid4

# --- LangGraph / LangChain ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- OpenAI / Azure ---
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- HF Transformers & Diffusers (Local VLM) ---
#import torch
#from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, pipeline, BitsAndBytesConfig
#from diffusers import DiffusionPipeline

# --- Playwright (for screenshots) ---
from playwright.sync_api import sync_playwright

# --- Load Environment Variables ---
load_dotenv()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 1: UNIFIED MODEL MANAGER (Singleton)
#
# Manages loading all API clients.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class ModelManager:
    """Manages loading all API clients at startup."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Check if already initialized
        if hasattr(self, 'azure_client'):
            return 
            
        print("Initializing and loading all API clients...")
        
        # 1. Configure Azure Client
        self.AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "")
        self.AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
        if not self.AZURE_API_KEY or not self.AZURE_ENDPOINT:
            print("Warning: AZURE_OPENAI_API_KEY or ENDPOINT_URL not set.")
        else:
            self.azure_client = AzureOpenAI(
                azure_endpoint=self.AZURE_ENDPOINT,
                api_key=self.AZURE_API_KEY,
                api_version="2024-10-21"
            )
            print("AzureOpenAI client loaded.")

        print("All API clients loaded and ready.")

    def get_azure_client(self) -> AzureOpenAI:
        if not hasattr(self, 'azure_client'):
            raise RuntimeError("Azure client not initialized. Set AZURE_OPENAI_API_KEY and ENDPOINT_URL.")
        return self.azure_client

    # --- VLM Chat (now routes to GPT-4 Vision) ---
    def chat_vlm(self, messages, temperature=0.2, max_new_tokens=2048):
        print("--- Routing chat_vlm to Azure GPT-4 Vision ---")
        client = self.get_azure_client()
        # Convert Qwen message format to Azure format
        azure_messages = []
        for msg in messages:
            role = msg['role']
            content_parts = []
            # msg['content'] is a list of dicts
            for part in msg['content']:
                if part['type'] == 'text':
                    content_parts.append({"type": "text", "text": part['text']})
                elif part['type'] == 'image':
                    # Qwen sends b64 data, Azure wants data:mime;base64,...
                    b64_data = part['image']
                    if not b64_data.startswith('data:'):
                        b64_data = f"data:image/png;base64,{b64_data}"
                    content_parts.append({"type": "image_url", "image_url": {"url": b64_data, "detail": "high"}})
            azure_messages.append({"role": role, "content": content_parts})
        
        resp = client.chat.completions.create(
            model="gpt-4.1-mini", # Use the default vision model from your script
            messages=azure_messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # --- LLM Chat (now routes to GPT-4) ---
    def chat_llm(self, prompt: str):
        print("--- Routing chat_llm to Azure GPT-4 ---")
        client = self.get_azure_client()
        messages = [{"role": "user", "content": prompt}]
        resp = client.chat.completions.create(
            model="gpt-4.1-mini", # Use the default text model from your script
            messages=messages,
            temperature=0.1,
            max_tokens=2048,
        )
        return (resp.choices[0].message.content or "").strip()

    # --- Generator (now routes to free NanoBanana/SD-Model-API) ---
    def generate_image(self, prompt: str) -> Image.Image:
        print(f"Generating image with sd-model-api (NanoBanana) for prompt: '{prompt}'")
        # This is a free, keyless SDXL API endpoint
        url = "https://sd-model-api.mfr.io/v1/generation/text-to-image"
        # Using the "nanobanana/photofinish-xl" model for good quality
        payload = {"prompt": prompt, "model": "nanobanana/photofinish-xl"}
        
        try:
            response = requests.post(url, json=payload, timeout=60) # 60 second timeout
            response.raise_for_status()
            
            # The API returns a list of base64 images
            data = response.json()
            b64_image = data['images'][0]
            
            # Decode the base64 string
            image_data = base64.b64decode(b64_image)
            image_stream = io.BytesIO(image_data)
            
            # Open the image with PIL
            img = Image.open(image_stream)
            return img
        except Exception as e:
            print(f"Error calling image generation API: {e}")
            # Return a dummy placeholder image on failure
            img = Image.new('RGB', (512, 512), color = (128, 128, 128))
            return img
    
    # --- Azure Chat (Unchanged) ---
    def chat_complete_azure(self, deployment: str, messages: List[Dict[str, Any]],
                            temperature: float, max_tokens: int) -> str:
        client = self.get_azure_client()
        resp = client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

# --- Initialize models ONCE ---
models = ModelManager()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 2: ASSET-GENERATION TOOL
#
# This graph is now a single-node pipeline.
# It no longer searches Pexels. It only generates and resizes images.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def b64img(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class AssetGraphState(TypedDict):
    """State for the asset-finding subgraph."""
    instructions: str
    bounding_box: Tuple[int, int]
    # These are no longer used but kept for schema compatibility
    search_query: Optional[str]
    found_image_url: Optional[str]
    # This is the only output
    final_asset_path: Optional[str]

def asset_generate_and_resize_node(state: AssetGraphState) -> dict:
    print("---(Asset Tool) NODE: Generate & Resize Image---")
    prompt = state["instructions"]
    bbox = state.get("bounding_box")
    
    # 1. Generate the image
    generated_image = models.generate_image(prompt)
    
    # 2. Resize the image if bounding_box is provided
    if bbox and isinstance(bbox, (tuple, list)) and len(bbox) == 2:
        try:
            # Use thumbnail to resize *while maintaining aspect ratio*
            generated_image.thumbnail(bbox)
            print(f"Resized image to fit within {bbox}")
        except Exception as e:
            print(f"Warning: Could not resize image. Error: {e}")

    # 3. Save the final image
    output_dir = pathlib.Path("Outputs/Assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"generated_{uuid4()}.png"
    full_save_path = output_dir / filename
    generated_image.save(full_save_path)
    print(f"Image generated and saved to {full_save_path}")
    
    # 4. Create the relative path for HTML
    html_path = pathlib.Path("Assets") / filename
    final_asset_path = str(html_path.as_posix())
    
    return {"final_asset_path": final_asset_path}

def build_asset_graph():
    workflow = StateGraph(AssetGraphState)
    # This is now a single-node graph
    workflow.add_node("generate_image", asset_generate_and_resize_node)
    workflow.set_entry_point("generate_image")
    workflow.add_edge("generate_image", END)
    return workflow.compile()

# --- Compile the graph ---
asset_agent_app = build_asset_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 3: CODE EDITOR TOOL
#
# This is the self-contained graph for editing HTML.
# It will be used as a tool by the "Brain".
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class CodeEditorState(TypedDict):
    html_code: str
    user_request: str
    messages: list[str]

EDITOR_SYSTEM_PROMPT = """
You are an expert senior web developer specializing in HTML, CSS, and JavaScript.
Your task is to take an existing HTML file, a user's request for changes, and to output the *new, complete, and updated HTML file*.

**CRITICAL RULES:**
1.  **Output ONLY the Code:** Your entire response MUST be *only* the raw, updated HTML code.
2.  **No Conversation:** Do NOT include "Here is the updated code:", "I have made the following changes:", or any other explanatory text, comments, or markdown formatting.
3.  **Return the Full File:** Always return the complete HTML file, from `<!DOCTYPE html>` to `</html>`, incorporating the requested changes. Do not return just a snippet.
"""

def _clean_llm_output(code: str) -> str:
    """Removes common markdown formatting."""
    code = code.strip()
    if code.startswith("```html"):
        code = code[7:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def _call_gpt_editor(html_code: str, user_request: str) -> str:
    """Uses Azure (GPT) model."""
    user_prompt = f"**User Request:**\n{user_request}\n\n**Original HTML Code:**\n```html\n{html_code}\n```\n\n**Your updated HTML Code:**"
    try:
        # --- MODIFIED: Use the ModelManager's chat_complete_azure method ---
        messages = [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        edited_code = models.chat_complete_azure(
            deployment="gpt-4.1-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=8192
        )
        return _clean_llm_output(edited_code)
    except Exception as e:
        print(f"Error calling Azure API: {e}")
        return f"\n{html_code}"

def _call_qwen_editor(html_code: str, user_request: str) -> str:
    """Uses Local Qwen VLM."""
    user_prompt = f"**User Request:**\n{user_request}\n\n**Original HTML Code:**\n```html\n{html_code}\n```\n\n**Your updated HTML Code:**"
    messages = [
        {"role": "system", "content": [{"type": "text", "text": EDITOR_SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
    ]
    try:
        edited_code = models.chat_vlm(messages, temperature=0.0, max_new_tokens=8192)
        return _clean_llm_output(edited_code)
    except Exception as e:
        print(f"Error calling local Qwen VLM: {e}")
        return f"\n{html_code}"

def node_edit_code(state: CodeEditorState) -> dict:
    print("---(Edit Tool) NODE: Edit Code---")
    html_code, user_request = state['html_code'], state['user_request']
    messages = state.get('messages', [])
    
    if not user_request:
        return {"messages": messages + ["No user request provided. Skipping edit."]}
        
    try:
        print(f"--- Routing editor to Azure GPT ---")
        new_html_code = _call_gpt_editor(html_code, user_request)
        
        msg = f"Code edit complete using Azure GPT."
        print(msg)
        return {"html_code": new_html_code, "user_request": "", "messages": messages + [msg]}
    except Exception as e:
        error_msg = f"Error in code editing node: {e}"
        print(error_msg)
        return {"html_code": html_code, "messages": messages + [error_msg]}

def build_edit_graph():
    workflow = StateGraph(CodeEditorState)
    workflow.add_node("edit_code", node_edit_code)
    workflow.set_entry_point("edit_code")
    workflow.add_edge("edit_code", END)
    return workflow.compile(checkpointer=MemorySaver())

# --- Compile the graph ---
edit_agent_app = build_edit_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 4: AZURE VLM PIPELINE
#
# This pipeline is reordered:
# 1. (Qwen) Plan assets (names, descriptions, sizes).
# 2. (SD Mini) Generate all assets.
# 3. (GPT-4) Generate code, using the asset names as placeholders.
# 4. (Python) Patch the HTML to replace names with final paths.
# 5. (GPT-4) Score & Refine.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

## --- Helpers ---
_SCORE_KEYS = ["aesthetics","completeness","layout_fidelity","text_legibility","visual_balance"]

def _section(text: str, name: str) -> str:
    pat = rf"{name}:\s*\n(.*?)(?=\n[A-Z_]+:\s*\n|\Z)"
    m = re.search(pat, text, flags=re.S)
    return m.group(1).strip() if m else ""

def _score_val(block: str, key: str, default: int = 0) -> int:
    m = re.search(rf"{key}\s*:\s*(-?\d+)", block, flags=re.I)
    try:
        return int(m.group(1)) if m else default
    except:
        return default
        
def encode_image_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def extract_html(text: str) -> str:
    m = re.search(r"```html(.*?)```", text, flags=re.S|re.I)
    if m:
        return m.group(1).strip()
    i = text.lower().find("<html")
    return text[i:].strip() if i != -1 else text.strip()

def _patch_html(html: str, asset_paths: Dict[str, str], messages: List[str]) -> Tuple[str, List[str]]:
    """
    Helper to patch HTML with asset paths.
    Replaces src="asset_name" with src="Assets/path/to/asset.png"
    """
    if not asset_paths:
        messages.append("Patching: No assets to patch.")
        return html, messages

    for asset_name, new_path in asset_paths.items():
        
        # --- 1. Patch <img> tags ---
        # Regex: src="asset_name" (with optional quotes)
        img_regex = re.compile(rf'src=(["\']?){re.escape(asset_name)}\1', re.I)
        replace_with = f'src="{new_path}"'
        
        new_html, count = img_regex.subn(replace_with, html)
        if count > 0:
            html = new_html
            messages.append(f"Patched <img> tag for {asset_name} -> {new_path}")

        # --- 2. Patch CSS background-image ---
        # Regex: url("asset_name") or url('asset_name') or url(asset_name)
        css_regex = re.compile(rf'url\((["\']?){re.escape(asset_name)}\1\)', re.I)
        css_replace_with = f'url("{new_path}")'
        
        new_html, count = css_regex.subn(css_replace_with, html)
        if count > 0:
            html = new_html
            messages.append(f"Patched CSS background-image for {asset_name} -> {new_path}")
    
    return html, messages

### --- Prompts ---
RELDESC_SYSTEM = "You are a meticulous UI analyst who describes layouts as a single dense paragraph of relative relationships."
RELDESC_PROMPT = """
From the provided wireframe image, produce ONE detailed paragraph (no bullets, no lists, no headings, no JSON, no code)
that states the RELATIVE layout and styling of the page so a code generator can rebuild it.

Requirements for the paragraph:
- Mention the overall background color and the dominant text color (infer reasonable #HEX).
- Describe the NAV BAR first: its position (top row), left/center/right alignment of brand/logo and items, the item order,
  which (if any) is emphasized/active, approximate pill/underline treatment, and colors for default vs active.
- Describe the CONTENT in reading order by rows:
  * For each row, say HOW MANY items appear side-by-side and their approximate relative widths (e.g., equal thirds, 1/3–2/3).
  * For “cards” or boxes: state the title text (or short descriptor), the presence of body text, any CTA (button/link) labels,
    button shape (rounded/square), fill/outline, and inferred #HEX colors for surface, border, and CTA.
  * Call out approximate spacing (e.g., tight/medium/roomy), gutters/gaps (px if you can), and typical radii (px).
- Describe the FOOTER last: alignment (center/left/right), text size relative to body, and background/text colors.
- Include one sentence on typography: font family category (system sans/serif), approximate base size (px), and headings hierarchy.
- Keep everything in ONE paragraph. Do not use line breaks except normal wrapping.

Return ONLY that single paragraph.
"""
BRIEF_SYSTEM = "You are a senior product designer who converts wireframes into precise UI design briefs."
BRIEF_PROMPT = """
Using the RELATIVE LAYL'OUT DESCRIPTION (authoritative for relative structure) and the wireframe image,
write a **UI DESIGN BRIEF** using EXACTLY these section headings, in this order. Be concise but specific. Infer reasonable hex colors.
If there is any conflict, prefer the wireframe image but keep structure consistent with the relative description.

### UI_SUMMARY
One paragraph that states page purpose and the major regions.

### COLOR_PALETTE
List 6–10 colors as `name: #HEX` including background, surface/card, text, muted text, primary,
secondary/accent, link, button-default, button-active.

### TYPOGRAPHY
Font family (system stack), base font-size (px), title sizes (h1/h2/h3 in px), and weight rules.
Line-heights.

### LAYOUT
Container max-width (px), global padding (px), section gaps (px), and the overall structure
(header/nav, content rows/columns, footer). State **how many items appear side-by-side** in each row
and at which breakpoint they stack.

### NAVBAR
Exact order of items, which one is ACTIVE, and pill styling (padding, radius, default vs active
background/text colors).

### CARDS
For each card in the content row: title text, body summary, CTA label and style (button/link),
card padding, radius, shadow, spacing between title/body/cta.

### RESPONSIVE_RULES
Breakpoints (sm/md/lg in px) and what changes at each (column stack rules, spacing adjustments).

### SPACING_AND_BORDERS
Numbers (px) for margins, gaps, radii used across elements.

Output ONLY the brief text with those headings (no code fences, no JSON).
"""

ASSET_NAMING_SYSTEM = "You are a UI/UX asset planner. Your job is to analyze a wireframe and a UI brief to identify all necessary image assets."
ASSET_NAMING_PROMPT = """
Analyze the wireframe image and the UI DESIGN BRIEF.
Your task is to identify every single component that needs a real image asset (e.g., logos, hero images, card images, user avatars).

Respond with ONLY a JSON list of objects, one for each asset needed.
If no assets are needed, return an empty list [].

JSON Schema:
[
  {
    "name": "<unique_snake_case_name_for_this_asset_e.g., hero_image, product_ringly_go>",
    "description": "<detailed description for an image generation model, e.g., 'a photo of a modern office building', 'a geometric logo in blue and white'>",
    "width": <int_width_in_px_e.g., 1200>,
    "height": <int_height_in_px_e.g., 400>
  }
]

UI DESIGN BRIEF:
"""

CODE_SYSTEM = "You are a meticulous frontend engineer who writes clean, modern, responsive HTML+CSS."
CODE_PROMPT = """
You will be given a **RELATIVE LAYOUT DESCRIPTION**, a **UI DESIGN BRIEF**, and a **LIST OF PRE-GENERATED ASSETS**.

Your task is to generate a SINGLE, self-contained HTML document.

**CRITICAL ASSET RULE:**
You MUST use the pre-generated assets. Their names and descriptions are provided below.
When you need an image, you MUST use its exact **name** (e.g., "hero_image") as the placeholder in the `src` attribute or `url()` function.

Example:
- For an `<img>` tag: `<img src="hero_image" alt="...">`
- For a CSS background: `background-image: url("hero_image");`

DO NOT use `src="placeholder"`, `data-asset-id`, or any external URLs like 'https://...'.

**PRE-GENERATED ASSETS:**
{asset_info_string}

**UI DESIGN BRIEF:**
{brief}

**RELATIVE LAYOUT DESCRIPTION:**
{rel_desc}

---
Output ONLY valid HTML starting with <html> and ending with </html>.
"""

SCORING_RUBRIC = r"""
You are an experienced front-end engineer. Compare two images: (A) the original wireframe, and (B) the generated HTML rendering,
and read the HTML/CSS code used for (B).

Return a PLAIN-TEXT report with the following sections EXACTLY in this order
(no JSON, no code fences around the whole report):

SCORES:
aesthetics: <0-10>
completeness: <0-10>
layout_fidelity: <0-10>   # be harsh; row alignment, relative sizes and positions must match A
text_legibility: <0-10>
visual_balance: <0-10>
aggregate: <float>        # mean of the five scores

ASSET_FEEDBACK:
1.  You **MUST** use the exact asset `name` from the "Valid Asset Names" list provided below.
2.  If an asset is good, DO NOT add it to the list.
3.  If all assets are good, return an empty list `[]`.

**Valid Asset Names:**
{asset_names_list}

[
  { "name": "<asset_name_from_plan>", "new_description": "<a new, improved description for the image generator>" }
]

ISSUES_TOP3:
- <short, specific issue 1>
- <issue 2>
- <issue 3>

LAYOUT_DIFFS:
- component: <nav|grid|card[1]|card[2]|footer>
  a_bbox_pct: [x,y,w,h]   # approx percentages (0–100) of page width/height in A
  b_bbox_pct: [x,y,w,h]   # same for B
  fix: <one sentence with exact px/cols/gaps>

CSS_PATCH:
```css
/* <= 40 lines, use existing selectors where possible; use px and hex colors */
.selector { property: value; }
/* ... */
```
HTML_EDITS:
- <one edit per line; selector + action, e.g., "add-class .card --class=wide":
- <allowed actions: move-before, move-after, insert-before, insert-after, set-attr, replace-text, add-class, remove-class>

REGENERATE_PROMPT: 
<1–4 lines with exact grid, gaps (px), radii (px), hex colors, and font sizes to rebuild if needed>

FEEDBACK: 
<one dense paragraph prioritizing layout_fidelity with exact px/cols/gaps/hex values>
"""
REFINE_SYSTEM = "You are a senior frontend engineer who strictly applies critique to improve HTML/CSS while matching the wireframe."
REFINE_PROMPT = """
You are given:
(A) the original wireframe image
The CURRENT HTML (single-file) that produced (B) the rendering
A critique ("feedback") produced by a rubric-based comparison of A vs B

Task:
Produce a NEW single-file HTML that addresses EVERY feedback point while staying faithful to A.
Fix layout fidelity (columns, spacing, alignment), completeness (ensure all components in A exist), typography/contrast for legibility, and overall aesthetics and balance.
Keep it self-contained (inline <style>; no external CSS/JS).
Output ONLY valid HTML starting with <html> and ending with </html>.
"""

@dataclass
class CodeRefineState:
    # CLI inputs
    image_path: Annotated[str, lambda a, b: a] # <-- MODIFIED
    out_rel_desc: Annotated[str, lambda a, b: a] # <-- MODIFIED
    out_brief: Annotated[str, lambda a, b: a] # <-- MODIFIED
    out_html: str
    vision_deployment: Annotated[str, lambda a, b: a] # <-- MODIFIED
    text_deployment: Annotated[str, lambda a, b: a] # <-- MODIFIED
    reldesc_tokens: Annotated[int, lambda a, b: a] # <-- MODIFIED
    brief_tokens: Annotated[int, lambda a, b: a] # <-- MODIFIED
    code_tokens: Annotated[int, lambda a, b: a] # <-- MODIFIED
    judge_tokens: Annotated[int, lambda a, b: a] # <-- MODIFIED
    temp: Annotated[float, lambda a, b: a] # <-- MODIFIED
    refine_max_iters: Annotated[int, lambda a, b: a] # <-- MODIFIED
    refine_threshold: Annotated[int, lambda a, b: a] # <-- MODIFIED
    shot_width: Annotated[int, lambda a, b: a] # <-- MODIFIED
    shot_height: Annotated[int, lambda a, b: a] # <-- MODIFIED
    
    # find_assets: bool = False  <-- REMOVED. We will always run the planner.
    
    # Runtime state
    image_data_url: Annotated[Optional[str], lambda a, b: a] = None # <-- MODIFIED
    rel_desc: Annotated[Optional[str], lambda a, b: a] = None # <-- MODIFIED
    brief: Annotated[Optional[str], lambda a, b: a] = None # <-- MODIFIED
    html: Optional[str] = None
    current_iteration: int = 0
    scores: Optional[Dict[str, Any]] = None
    stop_refinement: bool = False
    
    # Asset fields
    # asset_plan holds the JSON from Qwen
    asset_plan: Annotated[List[Dict[str, Any]], lambda a, b: a] = field(default_factory=list) # <-- MODIFIED
    # asset_paths maps the name to the final generated path
    asset_paths: Dict[str, str] = field(default_factory=dict)
    
    messages: List[str] = field(default_factory=list)

def parse_text_report(report: str) -> Dict[str, Any]:
    sb = _section(report, "SCORES")
    scores = {k: _score_val(sb, k, 0) for k in _SCORE_KEYS}
    m_agg = re.search(r"aggregate\s*:\s*([0-9]+(?:\.[0-9]+)?)", sb, flags=re.I)
    aggregate = float(m_agg.group(1)) if m_agg else sum(scores.values())/5.0
    css_patch = ""
    css_match = re.search(r"CSS_PATCH:\s*```css\s+(.*?)\s+```", report, flags=re.S|re.I)
    if css_match:
        css_patch = css_match.group(1).strip()
    
    asset_feedback = []
    asset_match = re.search(r"ASSET_FEEDBACK:\s*(\[.*?\])", report, flags=re.S|re.I)
    if asset_match:
        try:
            asset_feedback = json.loads(asset_match.group(1))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse ASSET_FEEDBACK JSON: {asset_match.group(1)}")
            pass # Keep asset_feedback as empty list

    html_edits = _section(report, "HTML_EDITS")
    regenerate_prompt = _section(report, "REGENERATE_PROMPT")
    feedback = _section(report, "FEEDBACK")
    issues = _section(report, "ISSUES_TOP3")
    layout_diffs = _section(report, "LAYOUT_DIFFS")

    return {
        "scores": scores,
        "aggregate": aggregate,
        "asset_feedback": asset_feedback, # <-- NEW
        "css_patch": css_patch,
        "html_edits": html_edits,
        "regenerate_prompt": regenerate_prompt,
        "feedback": feedback,
        "issues_top3": issues,
        "layout_diffs": layout_diffs,
        "raw": report,
    }

def refine_with_feedback(vision_deployment: str, wireframe_image: str, current_html: str, feedback: str,
                         css_patch: str = "", html_edits: str = "", regenerate_prompt: str = "",
                         temperature: float = 0.12, max_tokens: int = 2200) -> str:
    data_a = encode_image_to_data_url(wireframe_image)
    refine_instructions = f"{REFINE_PROMPT.strip()}\n\nAPPLY THESE PATCHES EXACTLY:..."
    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_a}},
            {"type": "text", "text": refine_instructions + "\n\nCURRENT_HTML:\n```html\n" + current_html + "\n```"}
        ]},
    ]
    out = models.chat_complete_azure(vision_deployment, messages, temperature, max_tokens)
    html = extract_html(out)
    if "<html" not in html.lower():
        html = f"<!DOCTYPE html>\n<html>\n<head><meta charset='utf-8'><title>Refined</title></head>\n<body>\n{html}\n</body>\n</html>"
    return html

def node_stage0(state: CodeRefineState) -> CodeRefineState:
    state.image_data_url = encode_image_to_data_url(state.image_path)
    messages = [
        {"role": "system", "content": RELDESC_SYSTEM},
        {"role": "user", "content": [
            {"type":"image_url", "image_url":{"url":state.image_data_url}},
            {"type":"text", "text": RELDESC_PROMPT.strip()},
        ]},
    ]
    state.rel_desc = models.chat_complete_azure(state.vision_deployment, messages, state.temp, state.reldesc_tokens)
    pathlib.Path(state.out_rel_desc).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_rel_desc, "w", encoding="utf-8") as f: f.write(state.rel_desc.strip())
    state.messages.append("Stage0: Generated relative layout description.")
    return state

def node_stage1(state: CodeRefineState) -> CodeRefineState:
    messages = [
        {"role": "system", "content": BRIEF_SYSTEM},
        {"role": "user", "content": [
            {"type":"image_url", "image_url":{"url":state.image_data_url}},
            {"type":"text", "text": BRIEF_PROMPT.strip() + "\n\nRELATIVE LAYOUT DESCRIPTION:\n" + state.rel_desc.strip()},
        ]},
    ]
    state.brief = models.chat_complete_azure(state.vision_deployment, messages, state.temp, state.brief_tokens)
    pathlib.Path(state.out_brief).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_brief, "w", encoding="utf-8") as f: f.write(state.brief)
    state.messages.append("Stage1: Generated UI design brief.")
    return state

def node_plan_assets_gpt(state: CodeRefineState) -> CodeRefineState: # <-- RENAMED
    print("---(Azure VLM) NODE: Planning Assets (GPT)---") # <-- RENAMED
    messages = [
        {"role": "system", "content": [{"type": "text", "text": ASSET_NAMING_SYSTEM}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this wireframe image and brief."},
            {"type": "image", "image": state.image_data_url.split(",")[1]}, # Send b64 data
            {"type": "text", "text": ASSET_NAMING_PROMPT + state.brief}
        ]}
    ]
    resp = models.chat_vlm(messages, temperature=0.0, max_new_tokens=1024) # This now calls GPT-4V
    
    try:
        plan = json.loads(resp[resp.find("["):resp.rfind("]")+1])
        state.asset_plan = plan
        state.messages.append(f"Stage1.5: Planned {len(plan)} assets (GPT).") # <-- RENAMED
        print(f"Asset plan: {plan}")
    except Exception as e:
        print(f"Error parsing asset plan: {e}\nResponse was: {resp}")
        state.asset_plan = []
        state.messages.append("Stage1.5: Error parsing asset plan (GPT).") # <-- RENAMED
    return state

def node_generate_assets(state: CodeRefineState) -> CodeRefineState:
    print("---(Azure VLM) NODE: Generating Assets---")
    if not state.asset_plan:
        state.messages.append("Stage1.6: No assets to generate.")
        return state
        
    current_asset_paths = {}
    for asset_request in state.asset_plan:
        name = asset_request.get('name')
        desc = asset_request.get('description')
        width = asset_request.get('width', 512)
        height = asset_request.get('height', 512)

        if not all([name, desc]): continue
        
        print(f"-> Generating asset for '{name}': {desc} (Size: {width}x{height})")
        
        # Call the asset_agent_app (which is now just the generator)
        result = asset_agent_app.invoke({
            "instructions": desc, 
            "bounding_box": (width, height)
        })
        
        if final_path := result.get("final_asset_path"):
            current_asset_paths[name] = final_path # Map the *name* to the path
            msg = f"Asset resolved for {name}: {final_path}"
            state.messages.append(msg); print(f"    ✅ {msg}")
        else:
            msg = f"Asset process failed for {name}."
            state.messages.append(msg); print(f"    ❌ {msg}")
            
    state.asset_paths = current_asset_paths
    return state

def node_stage2(state: CodeRefineState) -> CodeRefineState:
    
    # Create the asset info string for the prompt
    asset_info_string = "I have pre-generated the following images for you:\n"
    if not state.asset_plan:
        asset_info_string = "No image assets were found or generated. Do not add any `<img>` tags unless they are placeholders."
    else:
        for asset in state.asset_plan:
            asset_info_string += f"- Name: `{asset['name']}` (Description: {asset['description']})\n"

    # Format the prompt
    prompt = CODE_PROMPT.format(
        asset_info_string=asset_info_string,
        brief=state.brief.strip(),
        rel_desc=state.rel_desc.strip()
    )
    
    messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": [{"type":"text", "text": prompt}]},
    ]
    
    raw = models.chat_complete_azure(state.text_deployment, messages, state.temp, state.code_tokens)
    state.html = extract_html(raw)
    
    state.messages.append("Stage2: Generated HTML (with asset names as placeholders).")
    return state

def node_patch_html_with_assets(state: CodeRefineState) -> CodeRefineState:
    """
    Replaces the asset names (e.g., src="hero_image") with the final paths.
    """
    print("---(Azure VLM) NODE: Patching HTML with generated asset paths---")
    
    if not state.asset_paths: # <-- MODIFIED: Simplified check
        state.messages.append("Stage2.7: No assets to patch.")
    else:
        # Use the helper to patch
        state.html, state.messages = _patch_html(state.html, state.asset_paths, state.messages)

    # Save the (potentially) patched HTML
    pathlib.Path(state.out_html).parent.mkdir(parents=True, exist_ok=True)
    with open(state.out_html, "w", encoding="utf-8") as f: f.write(state.html)
    state.messages.append(f"Stage2.7: Saved HTML -> {state.out_html}")
    return state

def node_save_html_pre_score(state: CodeRefineState) -> CodeRefineState:
    """
    Saves the HTML file to the out_html path.
    This is used when the asset patching flow is skipped.
    """
    print("---(Azure VLM) NODE: Saving HTML before scoring---")
    try:
        pathlib.Path(state.out_html).parent.mkdir(parents=True, exist_ok=True)
        with open(state.out_html, "w", encoding="utf-8") as f:
            f.write(state.html)
        state.messages.append(f"Saved HTML for scoring -> {state.out_html}")
    except Exception as e:
        state.messages.append(f"Error saving HTML: {e}")
        print(f"Error saving HTML: {e}")
    return state

def node_stage3_score(state: CodeRefineState) -> CodeRefineState:
    html_path = pathlib.Path(state.out_html)
    shot_png_path = html_path.with_name(html_path.stem + f"_iter{state.current_iteration}.png")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": state.shot_width, "height": state.shot_height}, device_scale_factor=2.0)
        page = ctx.new_page()
        page.goto(pathlib.Path(state.out_html).resolve().as_uri())
        page.wait_for_load_state("networkidle")
        page.screenshot(path=shot_png_path, full_page=True)
        ctx.close()
        browser.close()
    
    data_a = encode_image_to_data_url(state.image_path)
    data_b = encode_image_to_data_url(shot_png_path)

    # *** NEW: Generate the asset name list for the prompt ***
    asset_names = [asset.get("name") for asset in state.asset_plan if asset.get("name")]
    if not asset_names:
        asset_names_list_str = "- No assets were planned."
    else:
        asset_names_list_str = "\n".join([f"- `{name}`" for name in asset_names])
    
    # *** Inject the list into the rubric ***
    scoring_rubric_prompt = SCORING_RUBRIC.format(asset_names_list=asset_names_list_str)
    
    messages = [
        {"role": "system", "content": "Return the specified PLAIN-TEXT report exactly as instructed."},
        {"role": "user", "content": [
            {"type": "text", "text": scoring_rubric_prompt}, # <-- Use the formatted prompt
            {"type": "image_url", "image_url":{"url": data_a}},
            {"type": "image_url", "image_url":{"url": data_b}},
            {"type": "text", "text": "HTML/CSS code used to produce image (B):\n" + state.html}
        ]},
    ]
    resp = models.chat_complete_azure(state.vision_deployment, messages, 0.0, state.judge_tokens)
    state.scores = parse_text_report(resp) # This will now parse the correct names
    state.messages.append(f"Stage3: Scoring done (Iter {state.current_iteration}).")
    
    min_score = min(int(state.scores["scores"][k]) for k in _SCORE_KEYS)
    if min_score >= state.refine_threshold:
        state.stop_refinement = True
    return state

def node_refine_loop(state: CodeRefineState) -> CodeRefineState:
    if state.stop_refinement or state.current_iteration >= state.refine_max_iters:
        state.messages.append("Refinement loop ended.")
        return state
    
    state.current_iteration += 1

    assets_to_fix = state.scores.get("asset_feedback", [])
    # Check if there are assets to fix *and* assets were planned
    if assets_to_fix and state.asset_plan:
        state.messages.append(f"Refining {len(assets_to_fix)} assets...")
        print(f"---(Azure VLM) REFINING ASSETS: {assets_to_fix} ---")
        
        # Get the asset plan to find the right bounding box
        asset_plan_map = {item.get("name"): item for item in state.asset_plan}
        
        for asset_request in assets_to_fix:
            name = asset_request.get("name")
            new_desc = asset_request.get("new_description")
            if not all([name, new_desc]):
                continue

            # Get original planned size, or default
            plan_item = asset_plan_map.get(name, {})
            width = plan_item.get("width", 512)
            height = plan_item.get("height", 512)
            
            print(f"-> Re-generating asset for '{name}': {new_desc} (Size: {width}x{height})")
            
            # Call the asset_agent_app (which is now just the generator)
            result = asset_agent_app.invoke({
                "instructions": new_desc, 
                "bounding_box": (width, height)
            })
            
            if final_path := result.get("final_asset_path"):
                state.asset_paths[name] = final_path
                msg = f"Asset REFINED for {name}: {final_path}"
                state.messages.append(msg); print(f"    ✅ {msg}")
            else:
                msg = f"Asset refine process failed for {name}."
                state.messages.append(msg); print(f"    ❌ {msg}")
    
    state.html = refine_with_feedback(
        vision_deployment=state.vision_deployment,
        wireframe_image=state.image_path,
        current_html=state.html,
        feedback=state.scores.get("feedback",""),
        css_patch=state.scores.get("css_patch",""),
        html_edits=state.scores.get("html_edits",""),
        regenerate_prompt=state.scores.get("regenerate_prompt",""),
        temperature=state.temp,
        max_tokens=state.code_tokens
    )
    
    if state.asset_paths:
        state.messages.append(f"Re-patching assets for iteration {state.current_iteration}...")
        state.html, state.messages = _patch_html(state.html, state.asset_paths, state.messages)
    
    versioned_path = pathlib.Path(state.out_html).with_name(pathlib.Path(state.out_html).stem + f"_v{state.current_iteration}" + pathlib.Path(state.out_html).suffix)
    with open(versioned_path, "w", encoding="utf-8") as f: f.write(state.html)
    state.out_html = str(versioned_path) 
    state.messages.append(f"Saved refined HTML v{state.current_iteration} -> {versioned_path}")
    return state

def decide_next(state: CodeRefineState) -> str:
    if not state.stop_refinement and state.current_iteration < state.refine_max_iters:
        return "refine_loop"
    return "end"

# REMOVED: def route_to_asset_planning(state: CodeRefineState) -> str:

def build_azure_vlm_graph():
    workflow = StateGraph(CodeRefineState)
    workflow.add_node("stage0", node_stage0)
    workflow.add_node("stage1", node_stage1)
    workflow.add_node("plan_assets_gpt", node_plan_assets_gpt)
    workflow.add_node("generate_assets", node_generate_assets)
    workflow.add_node("stage2", node_stage2)
    workflow.add_node("patch_html", node_patch_html_with_assets)
    workflow.add_node("stage3_score", node_stage3_score)
    workflow.add_node("refine_loop", node_refine_loop)

    workflow.set_entry_point("stage0")
    workflow.add_edge("stage0", "stage1")
    
    # --- MODIFIED: Removed conditional routing ---
    # We now always run the planner, which is simpler
    workflow.add_edge("stage1", "plan_assets_gpt")
    
    # --- MODIFIED: Parallel Fan-out / Fan-in ---
    # plan_assets_gpt forks to two parallel branches
    workflow.add_edge("plan_assets_gpt", "generate_assets") # Branch A
    workflow.add_edge("plan_assets_gpt", "stage2")          # Branch B
    
    # patch_html will only run after *both* branches are complete
    workflow.add_edge("generate_assets", "patch_html") # Join A
    workflow.add_edge("stage2", "patch_html")          # Join B
    
    # --- Continue to scoring ---
    workflow.add_edge("patch_html", "stage3_score")
    
    # Original refinement loop
    workflow.add_edge("stage3_score", "refine_loop")
    workflow.add_conditional_edges("refine_loop", decide_next, {"refine_loop": "stage3_score", "end": END})
    
    return workflow.compile(checkpointer=MemorySaver())

# --- Compile the graph ---
azure_vlm_app = build_azure_vlm_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 5: MAIN "BRAIN" AGENT (Command Center)
#
# This new agent uses the local Qwen VLM to decide which
# pipeline to run using standard conditional routing.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class BrainState(TypedDict):
    messages: List[Dict[str, Any]]
    cli_args: argparse.Namespace
    
    # New fields for routing
    next_task: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None
    task_result: Optional[str] = None


# --- Pipeline Functions (With Optional Output Paths) ---

def helper_run_azure_vlm_pipeline(image_path: str, 
                                  out_html_path: str, out_brief_path: str, out_reldesc_path: str) -> str:
    """
    Use this tool to generate a new HTML webpage from a wireframe image.
    This runs the full Azure VLM pipeline.
    
    Args:
        image_path (str): The file path to the input wireframe image.
        out_html_path (str): The file path to save the final HTML.
        out_brief_path (str): The file path to save the UI brief.
        out_reldesc_path (str): The file path to save the relative description.
    """
    print(f"--- BRAIN: Invoking Azure VLM Pipeline for {image_path} ---")
    try:
        # --- Hardcode model choices ---
        vision_deployment = "gpt-4.1-mini"
        text_deployment = "gpt-4.1-mini"
        
        pathlib.Path(out_html_path).parent.mkdir(parents=True, exist_ok=True)
        
        state = CodeRefineState(
            image_path=image_path,
            out_rel_desc=out_reldesc_path,
            out_brief=out_brief_path,
            out_html=out_html_path,
            vision_deployment=vision_deployment, 
            text_deployment=text_deployment,
            # find_assets=find_assets, <-- REMOVED
            # Hardcode pipeline defaults
            reldesc_tokens=700,
            brief_tokens=1100,
            code_tokens=2200,
            judge_tokens=900,
            temp=0.12,
            refine_max_iters=3,
            refine_threshold=8,
            shot_width=1536,
            shot_height=900
        )

        run_id = f"wireframe-{uuid4()}"
        config = {"configurable": {"thread_id": run_id}}
        result = azure_vlm_app.invoke(state, config=config)
        
        final_path = result.get('out_html', out_html_path)
        return json.dumps({
            "status": "success",
            "message": "Azure VLM pipeline completed.",
            "final_html_path": final_path,
            "messages": result.get("messages", [])
        })
    except Exception as e:
        print(f"Error in Azure VLM helper: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def helper_run_code_editor(html_path: str, edit_request: str, output_path: str) -> str:
    """
    Use this tool to edit an existing HTML file based on a user's text request.
    Args:
        html_path (str): The file path to the HTML file you want to edit.
        edit_request (str): The user's instruction (e.g., "Make the h1 tag blue").
        output_path (str): The file path to save the *new* edited HTML.
    """
    print(f"--- BRAIN: Invoking Code Editor for {html_path} ---")
    try:
        
        with open(html_path, "r", encoding="utf-8") as f:
            original_html = f.read()
            
        initial_state = {
            "html_code": original_html,
            "user_request": edit_request,
            "messages": []
        }
        
        config = {"configurable": {"thread_id": f"edit-thread-{uuid4()}"}}
        final_state = edit_agent_app.invoke(initial_state, config=config)
        
        new_html_code = final_state['html_code']
        
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(new_html_code)
            
        return json.dumps({
            "status": "success",
            "message": "Code editing complete.",
            "final_html_path": output_path,
            "messages": final_state.get("messages", [])
        })
    except Exception as e:
        print(f"Error in Code Editor helper: {e}")
        return json.dumps({"status": "error", "message": str(e)})

def helper_run_asset_search(description: str, width: int = 512, height: int = 512) -> str:
    """
    Use this tool to find or generate a single image asset.
    ... (docstring args) ...
    """
    print(f"--- BRAIN: Invoking Asset Search for '{description}' ---")
    try:
        result = asset_agent_app.invoke({"instructions": description, "bounding_box": (width, height)})
        
        if final_path := result.get("final_asset_path"):
            return json.dumps({
                "status": "success",
                "message": "Asset found/generated.",
                "asset_path": final_path
            })
        else:
            return json.dumps({"status": "error", "message": "Asset process failed."})
    except Exception as e:
        print(f"Error in Asset Search helper: {e}")
        return json.dumps({"status": "error", "message": str(e)})

# --- NEW: Helper for default chat response ---
def helper_respond_to_user(prompt: str) -> str:
    """
    Use this tool to provide a general text-based response to a user query
    that doesn't match any other tool.
    """
    print(f"--- BRAIN: Invoking Default Chat Response ---")
    try:
        response = models.chat_llm(prompt)
        return json.dumps({
            "status": "success",
            "message": "Chat response generated.",
            "response": response
        })
    except Exception as e:
        print(f"Error in Default Chat helper: {e}")
        return json.dumps({"status": "error", "message": str(e)})


# --- List of all helper functions for the Brain ---
helper_functions = [
    helper_run_azure_vlm_pipeline,
    helper_run_code_editor,
    helper_run_asset_search,
    helper_respond_to_user, # <-- NEW
]

# --- Brain Agent Definition (Router) ---
class BrainRouterAgent: # <-- RENAMED
    def __init__(self, model_manager, functions, system_prompt=""):
        self.model = model_manager
        self.functions = {f.__name__: f for f in functions}
        self.system_prompt = system_prompt

    def __call__(self, state: BrainState):
        messages = state['messages']
        last_user_message = messages[-1]['content']
        
        # *** MODIFIED, HYPER-STRICT PROMPT ***
        router_prompt = f"""
You are a JSON routing agent. Your *only* job is to classify the user's request into one of the following JSON formats. Do not add any other text.

**Function Schemas:**

1.  **Generate a new page from an image:**
    (Use if the prompt mentions 'generate', 'create', 'build' from an image)
    {{
      "function_name": "helper_run_azure_vlm_pipeline",
      "function_args": {{}}
    }}

2.  **Edit an existing HTML file:**
    (Use if the prompt mentions 'edit', 'change', 'modify', 'update' an existing file)
    {{
      "function_name": "helper_run_code_editor",
      "function_args": {{}}
    }}

3.  **Find or generate a single image asset:**
    (Use if the prompt *only* asks to 'find', 'search for', 'get', or 'generate' an image)
    {{
      "function_name": "helper_run_asset_search",
      "function_args": {{}}
    }}
    
4.  **Default chat response:**
    (Use this as a fallback if no other tool matches, e.g., 'hello', 'how are you?')
    {{
      "function_name": "helper_respond_to_user",
      "function_args": {{}}
    }}

**User Request:**
"{last_user_message}"

**Your JSON Response:**
"""
        
        print("--- BRAIN: Routing prompt ---")
        
        # This now calls Azure GPT-4
        vlm_response = self.model.chat_llm(router_prompt)
        
        print(f"--- BRAIN: VLM Response ---\n{vlm_response}\n-------------------------")

        try:
            call_json = json.loads(vlm_response[vlm_response.find("{"):vlm_response.rfind("}")+1])
            func_name = call_json.get("function_name")
            
            # Start with the minimal args from the VLM (e.g., {'find_assets': true})
            func_args = call_json.get("function_args", {}) 
            
            if func_name == "end" or func_name not in self.functions:
                print("--- BRAIN: No operation selected. Ending task. ---")
                return {"next_task": "end", "task_result": "No operation selected."}

            # *** Manually add all required args from context ***
            cli_args = state['cli_args']
            
            if func_name == "helper_run_azure_vlm_pipeline":
                # func_args['find_assets'] = func_args.get('find_assets', False) <-- REMOVED
                func_args['image_path'] = cli_args.image
                func_args['out_html_path'] = cli_args.out_html
                func_args['out_brief_path'] = cli_args.out_brief
                func_args['out_reldesc_path'] = cli_args.out_rel_desc

            elif func_name == "helper_run_code_editor":
                func_args['html_path'] = cli_args.html
                func_args['edit_request'] = last_user_message
                func_args['output_path'] = cli_args.out_html
                
            elif func_name == "helper_run_asset_search":
                func_args['description'] = last_user_message
                func_args['width'] = 512
                func_args['height'] = 512
            
            elif func_name == "helper_respond_to_user": # <-- NEW
                func_args['prompt'] = last_user_message
            
            return {
                "next_task": func_name,
                "task_args": func_args # Pass the *full* dict
            }

        except Exception as e:
            print(f"--- BRAIN: Error parsing VLM response. Ending task. --- \n{e}")
            return {"next_task": "end", "task_result": f"Error parsing VLM response: {e}"}

# --- Graph Nodes ---
def node_run_vlm_pipeline(state: BrainState) -> dict:
    print("---(Brain Graph) NODE: node_run_vlm_pipeline ---")
    args = state['task_args']
    result = helper_run_azure_vlm_pipeline(**args)
    return {"task_result": result, "messages": state['messages'] + [{"role": "assistant", "content": result}]}

def node_run_code_editor(state: BrainState) -> dict:
    print("---(Brain Graph) NODE: node_run_code_editor ---")
    args = state['task_args']
    result = helper_run_code_editor(**args)
    return {"task_result": result, "messages": state['messages'] + [{"role": "assistant", "content": result}]}

def node_run_asset_search(state: BrainState) -> dict:
    print("---(Brain Graph) NODE: node_run_asset_search ---")
    args = state['task_args']
    result = helper_run_asset_search(**args)
    return {"task_result": result, "messages": state['messages'] + [{"role": "assistant", "content": result}]}

# --- NEW: Node for chat response ---
def node_respond_to_user(state: BrainState) -> dict:
    print("---(Brain Graph) NODE: node_respond_to_user ---")
    args = state['task_args']
    result = helper_respond_to_user(**args)
    return {"task_result": result, "messages": state['messages'] + [{"role": "assistant", "content": result}]}


# --- Router Function ---
def brain_router(state: BrainState) -> str:
    """Routes to the correct node based on the 'next_task' state."""
    print(f"---(Brain Graph) ROUTER: Next task is '{state['next_task']}' ---")
    if state['next_task'] == "helper_run_azure_vlm_pipeline":
        return "run_vlm"
    elif state['next_task'] == "helper_run_code_editor":
        return "run_edit"
    elif state['next_task'] == "helper_run_asset_search":
        return "run_asset"
    elif state['next_task'] == "helper_respond_to_user": # <-- NEW
        return "run_chat"
    else:
        return "end"

# --- Graph Builder ---
def build_brain_graph():
    
    brain_agent_node = BrainRouterAgent(models, helper_functions) # <-- RENAMED
    
    workflow = StateGraph(BrainState)
    
    # ... (rest of function is unchanged) ...
    workflow.add_node("agent", brain_agent_node)
    workflow.add_node("run_vlm", node_run_vlm_pipeline)
    workflow.add_node("run_edit", node_run_code_editor)
    workflow.add_node("run_asset", node_run_asset_search)
    workflow.add_node("run_chat", node_respond_to_user) # <-- NEW
    
    workflow.set_entry_point("agent")
    
    workflow.add_conditional_edges(
        "agent",
        brain_router,
        {
            "run_vlm": "run_vlm",
            "run_edit": "run_edit",
            "run_asset": "run_asset",
            "run_chat": "run_chat", # <-- NEW
            "end": END
        }
    )
    
    workflow.add_edge("run_vlm", END)
    workflow.add_edge("run_edit", END)
    workflow.add_edge("run_asset", END)
    workflow.add_edge("run_chat", END) # <-- NEW
    
    return workflow.compile()

# --- Compile the brain graph ---
brain_app = build_brain_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 6: CLI RUNNER (MAIN) - SIMPLIFIED FOR TESTING
#
# This entrypoint is modified to run a full test suite.
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def run_test(test_name: str, initial_state: BrainState):
    """Helper function to run a single test case against the brain."""
    print("\n" + "="*70)
    print(f"--- STARTING TEST: {test_name} ---")
    print(f"--- User Prompt: {initial_state['messages'][0]['content']} ---")
    
    run_id = f"brain-test-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    
    # Invoke the brain
    final_state = brain_app.invoke(initial_state, config=config)
    
    print("\n" + ("-"*25) + " BRAIN INVOCATION COMPLETE " + ("-"*25))
    
    print("--- Final Task Result ---")
    task_result_str = final_state.get('task_result', "No result found (task may have ended early).")
    
    if task_result_str:
        try:
            output_json = json.loads(task_result_str)
            print(json.dumps(output_json, indent=2))
        except json.JSONDecodeError:
            print(task_result_str)
    else:
        print("No final task result recorded.")
        
    print("="*70)


def main():
    """
    Main function modified to run a test suite for all 3 pipeline paths.
    """
    
    # --- Test Data Setup (Paths Re-added) ---
    
    # For Test 1: Azure VLM Pipeline (No assets mentioned)
    cli_args_vlm = argparse.Namespace(
        prompt="Generate a new HTML page from the wireframe at Images/2.png",
        image="Images/2.png",
        html=None,
        out_html="Outputs/test1_output.html",
        out_brief="Outputs/test1_brief.txt",
        out_rel_desc="Outputs/test1_reldesc.txt"
    )

    # For Test 1.5: Azure VLM Pipeline (Assets mentioned)
    asset_test_image_path = "Images/asset_test.png"
    cli_args_vlm_assets = argparse.Namespace(
        prompt=f"Generate the page from {asset_test_image_path}, and please generate all the image assets for the placeholders.",
        image=asset_test_image_path,
        html=None,
        out_html="Outputs/test1.5_output.html",
        out_brief="Outputs/test1.5_brief.txt",
        out_rel_desc="Outputs/test1.5_reldesc.txt"
    )
    
    # For Test 2: Code Editor Pipeline
    test_edit_file = "Outputs/test_page_to_edit.html"
    cli_args_edit = argparse.Namespace(
        prompt="Change the title to 'Edited by Brain Agent' and make the h1 tag red.",
        image=None,
        html=test_edit_file,
        out_html="Outputs/test2_output_edited.html", # Specific output path
        out_brief=None, # Not needed
        out_rel_desc=None # Not needed
    )
    
    # For Test 3: Asset Search Pipeline
    cli_args_asset = argparse.Namespace(
        prompt="Find a high-quality photo of a 'modern office desk with a laptop'",
        image=None,
        html=None,
        out_html=None, # Not needed
        out_brief=None, # Not needed
        out_rel_desc=None # Not needed
    )

    # For Test 4: Default Chat Response
    cli_args_chat = argparse.Namespace(
        prompt="Hello, how does this whole system work?",
        image=None,
        html=None,
        out_html=None, # Not needed
        out_brief=None, # Not needed
        out_rel_desc=None # Not needed
    )

    # =================================================================
    # --- TEST 1: AZURE VLM PIPELINE (Image-to-Code, No Assets) ---
    # =================================================================
    initial_state_vlm = {
        "messages": [{"role": "user", "content": cli_args_vlm.prompt}],
        "cli_args": cli_args_vlm
    }
    if not pathlib.Path(cli_args_vlm.image).exists():
        print(f"\n--- WARNING: Skipping Test 1 ---")
        print(f"Test image not found at: {cli_args_vlm.image}")
    else:
        run_test("Test 1: Azure VLM Pipeline (No Assets Mentioned)", initial_state_vlm)

    # =================================================================
    # --- TEST 1.5: AZURE VLM PIPELINE (with Asset Search) ---
    # =================================================================
    
    '''
    initial_state_vlm_assets = {
        "messages": [{"role": "user", "content": cli_args_vlm_assets.prompt}],
        "cli_args": cli_args_vlm_assets
    }
    if not pathlib.Path(cli_args_vlm_assets.image).exists():
        print(f"\n--- WARNING: Skipping Test 1.5 ---")
        print(f"Asset test image not found at: {cli_args_vlm_assets.image}")
        print("This is the specific test for the internal asset-search pipeline.")
    else:
        run_test("Test 1.5: Azure VLM Pipeline (with Asset Search)", initial_state_vlm_assets)
    '''
    
    # =================================================================
    # --- TEST 2: CODE EDITOR PIPELINE (Edit HTML) ---
    # =================================================================
    initial_state_edit = {
        "messages": [{"role": "user", "content": cli_args_edit.prompt}],
        "cli_args": cli_args_edit
    }
    run_test("Test 2: Code Editor Pipeline", initial_state_edit)
    
    # =================================================================
    # --- TEST 3: ASSET SEARCH PIPELINE (Find Image) ---
    # =================================================================
    '''
    initial_state_asset = {
        "messages": [{"role": "user", "content": cli_args_asset.prompt}],
        "cli_args": cli_args_asset
    }
    run_test("Test 3: Asset Search Pipeline", initial_state_asset)
    '''

    # =================================================================
    # --- TEST 4: DEFAULT CHAT RESPONSE ---
    # =================================================================
    initial_state_chat = {
        "messages": [{"role": "user", "content": cli_args_chat.prompt}],
        "cli_args": cli_args_chat
    }
    run_test("Test 4: Default Chat Response", initial_state_chat)


if __name__ == "__main__":
    # Ensure output directories exist
    pathlib.Path("Outputs/Assets").mkdir(parents=True, exist_ok=True)
    pathlib.Path("Outputs").mkdir(parents=True, exist_ok=True)
    
    # --- Create a dummy HTML file for Test 2 ---
    test_edit_file_path = "Outputs/test_page_to_edit.html"
    dummy_html = "..." # Removed multi-line prompt
    
    with open(test_edit_file_path, "w", encoding="utf-8") as f:
        f.write(dummy_html)
    print(f"Created dummy file for editing at: {test_edit_file_path}")
    
    # Check for test image for Test 1
    if not pathlib.Path("Images/2.png").exists():
        print("\n--- WARNING ---")
        print("Test 1 (Azure VLM) requires an image at 'Images/2.png'.")
        print("Please add an image there or Test 1 will be skipped.")
        print("---------------\n")

    # Check for Test 1.5 image
    if not pathlib.Path("Images/asset_test.png").exists():
        print("\n--- WARNING ---")
        print("Test 1.5 (Azure VLM + Asset Search) requires an image at 'Images/asset_test.png'.")
        print("This test *specifically* verifies the internal asset search.")
        print("Please add an image with clear image placeholders, or Test 1.5 will be skipped.")
        print("---------------\n")

    main()