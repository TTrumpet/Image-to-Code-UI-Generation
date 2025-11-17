# agent_azure_vlm_tools.py
import os
import re
import json
import base64
import mimetypes
import argparse
import pathlib
import io
import requests
from typing import List, Dict, Any, Optional, TypedDict, Literal, Tuple
from dataclasses import dataclass, field
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from uuid import uuid4

# --- LangGraph / LangChain ---
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# --- OpenAI / Azure ---
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- Google Gemini ---
from google import genai
from google.genai import types

# --- Playwright (for screenshots) ---
from playwright.sync_api import sync_playwright

# --- Load Environment Variables ---
load_dotenv()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 1: UNIFIED MODEL MANAGER (Singleton) - GPT-ONLY
#
# Manages Azure / OpenAI clients and provides unified chat helpers.
# All model calls use the Azure/OpenAI GPT deployments (gpt-4.1-mini by default).
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

GPT_DEPLOYMENT_DEFAULT = os.getenv("GPT_DEPLOYMENT", "gpt-4.1-mini")

def _get_closest_aspect_ratio(width: int, height: int) -> str:
    """Calculates the closest supported aspect ratio for the Gemini API."""
    supported_ratios = {
        "1:1": 1.0, "16:9": 16/9, "9:16": 9/16, "4:3": 4/3, "3:4": 3/4,
        "4:5": 4/5, "5:4": 5/4, "2:3": 2/3, "3:2": 3/2, "21:9": 21/9,
    }
    
    target_ratio = width / height
    
    # Find the ratio string whose value is closest to the target_ratio
    closest_ratio_str = min(
        supported_ratios.keys(),
        key=lambda r: abs(supported_ratios[r] - target_ratio)
    )
    print(f"Target W/H ({width}x{height}, ratio {target_ratio:.2f}) -> Closest API ratio: {closest_ratio_str}")
    return closest_ratio_str

class ModelManager:
    """Manages Azure/OpenAI clients and provides chat wrappers for both vision+text and text-only usage."""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'azure_client'):  # Initialize only once
            print("Initializing Azure/OpenAI clients (GPT-only mode)...")

            # Azure OpenAI client (for most GPT calls)
            self.AZURE_ENDPOINT = os.getenv("ENDPOINT_URL", "")
            self.AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
            if not self.AZURE_API_KEY or not self.AZURE_ENDPOINT:
                print("Warning: AZURE_OPENAI_API_KEY or ENDPOINT_URL not set.")
                self.azure_client = None
            else:
                self.azure_client = AzureOpenAI(
                    azure_endpoint=self.AZURE_ENDPOINT,
                    api_key=self.AZURE_API_KEY,
                    api_version="2024-10-21"
                )
                print("AzureOpenAI client loaded.")

            # Default deployment name for GPT calls
            self.default_deployment = GPT_DEPLOYMENT_DEFAULT
            print(f"Default GPT deployment: {self.default_deployment}")

            self.google_api_key = os.getenv("GOOGLE_API_KEY")
            self.genai_client = None
            if not self.google_api_key:
                print("Warning: GOOGLE_API_KEY not set. Real image generation will be disabled.")
            else:
                try:
                    self.genai_client = genai.Client(api_key=self.google_api_key)
                    print("Google GenAI client initialized.")
                except Exception as e:
                    print(f"Error initializing Google GenAI: {e}")
                    self.genai_model = None

    def get_azure_client(self) -> AzureOpenAI:
        if not hasattr(self, 'azure_client') or self.azure_client is None:
            raise RuntimeError("Azure client not initialized. Set AZURE_OPENAI_API_KEY and ENDPOINT_URL.")
        return self.azure_client

    def chat_complete_azure(self, deployment: Optional[str], messages: List[Dict[str, Any]],
                            temperature: float = 0.0, max_tokens: int = 1024) -> str:
        """Calls Azure OpenAI chat completion and returns the assistant content as string."""
        client = self.get_azure_client()
        model_name = deployment or self.default_deployment
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    # For compatibility with earlier code that expected a VLM-style chat, expose chat_vlm and chat_llm wrappers.
    def chat_vlm(self, messages, temperature=0.2, max_new_tokens=2048, deployment: Optional[str] = None):
        """Use GPT (Azure) to handle vision+language style messages. 'messages' should be in the Azure chat format."""
        return self.chat_complete_azure(deployment or self.default_deployment, messages, temperature, max_new_tokens)

    def chat_llm(self, prompt: str, temperature: float = 0.1, max_tokens: int = 1024, deployment: Optional[str] = None):
        """Use GPT to handle plain text prompts: converts prompt to a single-user message."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.chat_vlm(messages, temperature=temperature, max_new_tokens=max_tokens, deployment=deployment)

    def _create_placeholder_image(self, prompt: str, width: int = 1024, height: int = 512) -> Image.Image:
        """
        Internal fallback function to create a placeholder image.
        (This is your original generate_image logic)
        """
        W, H = width, height
        img = Image.new("RGB", (W, H), color=(240, 240, 240))
        d = ImageDraw.Draw(img)
        try:
            font_size = max(16, int(H / 25)) 
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
            font_size = 12
        
        lines = []
        max_chars = max(20, int(W / (font_size * 0.6)))
        for i in range(0, len(prompt), max_chars):
            lines.append(prompt[i:i+max_chars])
        
        y = 20
        d.text((20, y), f"PLACEHOLDER ({W}x{H})", fill=(30, 30, 30), font=font)
        y += (font_size + 10)
        
        for line in lines:
            if y > (H - (font_size + 10)):
                d.text((20, y), "...", fill=(60, 60, 60), font=font)
                break
            d.text((20, y), line, fill=(60, 60, 60), font=font)
            y += (font_size + 2)
            
        return img

    def generate_image(self, prompt: str, width: int = 1024, height: int = 512) -> Image.Image:
        """
        Generates an image using Google GenAI (Nano Banana).
        Falls back to a placeholder if the API is not available or fails.
        """
        if not self.genai_client:
            print("-> GenAI model not configured, using placeholder.")
            return self._create_placeholder_image(prompt, width, height)

        try:
            # 1. Get the closest supported aspect ratio
            aspect_ratio = _get_closest_aspect_ratio(width, height)
            enhanced_prompt = f"{prompt}. Generate a photorealistic image. Aspect ratio {aspect_ratio}."
            print(f"-> Calling Gemini 2.0 Flash with prompt: '{enhanced_prompt}'")

            response = self.genai_client.models.generate_content(
                    model="gemini-2.0-flash-preview-image-generation",
                    contents=enhanced_prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )

            # Process Response
            if response and response.parts:
                for part in response.parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        print("-> Successfully generated image from inline_data.")
                        
                        # FIX: Check if it's ALREADY bytes (new SDK) vs string (old SDK logic)
                        raw_data = part.inline_data.data
                        if isinstance(raw_data, bytes):
                             img_data = raw_data
                        else:
                             img_data = base64.b64decode(raw_data)
                        
                        try:
                            image_stream = BytesIO(img_data)
                            pil_image = Image.open(image_stream)
                            return pil_image
                        
                        except Exception as img_exc:
                            print(f"-> Failed to open/verify image from BytesIO: {img_exc}")
                            return self._create_placeholder_image(f"[GenAI Image Decode Error] {prompt}", width, height)
                    
                    if hasattr(part, "image") and part.image:
                        print("-> Successfully generated image from .image attribute.")
                        return part.image

            print(f"-> GenAI response did not contain an image. Falling back to placeholder.")
            return self._create_placeholder_image(f"[GenAI No Image] {prompt}", width, height)

        except Exception as e:
            print(f"-> ERROR during GenAI image generation: {e}. Falling back to placeholder.")
            return self._create_placeholder_image(f"[API Error] {prompt}", width, height)

# --- Initialize models ONCE ---
models = ModelManager()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 2: ASSET-FINDING TOOL (unchanged logic but generator uses placeholder)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def b64img(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class AssetGraphState(TypedDict):
    instructions: str
    bounding_box: Tuple[int, int]
    search_query: str
    #found_image_url: Optional[str]
    final_asset_path: Optional[str]

def asset_prepare_search_query_node(state: AssetGraphState) -> dict:
    print("---(Asset Tool) NODE: Prepare Search Query---")
    prompt = f"""You are an expert at refining search queries. Extract only the essential visual keywords.
**CRITICAL INSTRUCTIONS:**
- DO NOT include words related to licensing.
- DO NOT include quotation marks.
User's request: "{state['instructions']}"
Respond with ONLY the refined search query."""
    raw_query = models.chat_llm(prompt)
    search_query = raw_query.strip().replace('"', '')
    print(f"Prepared search query: '{search_query}'")
    return {"search_query": search_query}

def asset_generate_image_node(state: AssetGraphState) -> dict:
    print("---(Asset Tool) NODE: Generate Image ---")
    # MODIFIED: Use the refined search_query, fallback to instructions
    prompt = state.get("search_query", state["instructions"]) 
    
    try:
        width, height = state['bounding_box']
        print(f"Generating asset with size {width}x{height} and prompt: '{prompt}'")
    except (KeyError, TypeError, ValueError):
        width, height = 1024, 512
        print(f"Warning: Bounding box not found. Using default {width}x{height}.")

    print(f"Generating asset with prompt: '{prompt}'")
    
    # NOTE: This is where you would swap the placeholder with your 
    # call to a real image gen API (e.g., Google GenAI, Nano Banana)
    generated_image = models.generate_image(prompt, width=width, height=height)
    
    output_dir = pathlib.Path("Outputs/Assets")
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"generated_{uuid4()}.png"
    full_save_path = output_dir / filename
    generated_image.save(full_save_path)
    print(f"Image generated and saved to {full_save_path}")
    html_path = pathlib.Path("Assets") / filename
    final_asset_path = str(html_path.as_posix())
    return {"final_asset_path": final_asset_path}

def build_asset_graph():
    workflow = StateGraph(AssetGraphState)

    # Add only the nodes we need
    workflow.add_node("prepare_search_query", asset_prepare_search_query_node)
    workflow.add_node("generate_image", asset_generate_image_node)

    # Set the entry point
    workflow.set_entry_point("prepare_search_query")

    # Define the simple, linear flow
    workflow.add_edge("prepare_search_query", "generate_image")
    workflow.add_edge("generate_image", END)

    return workflow.compile()

asset_agent_app = build_asset_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 3: CODE EDITOR TOOL (uses GPT via Azure)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

class CodeEditorState(TypedDict):
    html_code: str
    user_request: str
    model_choice: Literal["gpt-4.1-mini", "qwen-local"]
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
    code = code.strip()
    if code.startswith("```html"):
        code = code[7:]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()

def _call_gpt_editor(html_code: str, user_request: str, model: str) -> str:
    user_prompt = f"**User Request:**\n{user_request}\n\n**Original HTML Code:**\n```html\n{html_code}\n```\n\n**Your updated HTML Code:**"
    try:
        out = models.chat_complete_azure(model, [
            {"role": "system", "content": EDITOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ], temperature=0.0, max_tokens=8192)
        return _clean_llm_output(out)
    except Exception as e2:
        print(f"Fallback Azure call failed: {e2}")
        return f"\n{html_code}"

def node_edit_code(state: CodeEditorState) -> dict:
    print("---(Edit Tool) NODE: Edit Code---")
    html_code, user_request, model_choice = state['html_code'], state['user_request'], state['model_choice']
    messages = state.get('messages', [])

    if not user_request:
        return {"messages": messages + ["No user request provided. Skipping edit."]}

    try:
        new_html_code = _call_gpt_editor(html_code, user_request, model_choice)

        msg = f"Code edit complete using {model_choice}."
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

edit_agent_app = build_edit_graph()

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ## SECTION 4: AZURE VLM PIPELINE (now GPT-only for vision+text)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

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

'''
def _patch_html(html: str, asset_paths: Dict[str, str], messages: List[str]) -> Tuple[str, List[str]]:
    if not asset_paths:
        messages.append("Patching: No assets to patch.")
        return html, messages

    for component_id, new_path in asset_paths.items():
        tag_regex = re.compile(rf'<img[^>]+data-asset-id="{re.escape(component_id)}"[^>]*>', re.I | re.S)
        match = tag_regex.search(html)

        if match:
            full_tag = match.group(0)
            src_regex = re.compile(r'src="[^"]*"', re.I)
            patched_tag, count = src_regex.subn(f'src="{new_path}"', full_tag)

            if count > 0:
                html = html.replace(full_tag, patched_tag)
                messages.append(f"Patched <img> tag for {component_id} -> {new_path}")
            else:
                messages.append(f"Warning: Found <img> tag for {component_id} but couldn't replace src.")
        else:
            messages.append(f"Warning: Could not find <img> tag for {component_id} to patch.")

        if "hero" in component_id or "background" in component_id:
            css_regex = re.compile(r'background-image:\s*url\((["\']?)placeholder\1\)', re.I)
            css_replace_with = f'background-image: url("{new_path}")'
            new_html, count = css_regex.subn(css_replace_with, html)
            if count > 0:
                html = new_html
                messages.append(f"Patched CSS background-image for {component_id} -> {new_path}")

    return html, messages
'''

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

CODE_SYSTEM = "You are a meticulous frontend engineer who writes clean, modern, responsive HTML+CSS."
CODE_PROMPT = """
Using the following **RELATIVE LAYOUT DESCRIPTION**, **UI DESIGN BRIEF**, and **ASSET_PATHS**, generate a SINGLE, self-contained HTML document:

Requirements:
- Semantic tags: header/nav/main/section/footer.
- One <style> block; no external CSS/JS.
- Implement the layout: container max-width, gaps, grid columns, and stacking rules per breakpoints.

- **CRITICAL ASSET RULE: You MUST use the asset paths provided in the `ASSET_PATHS` JSON.**
- For each asset in the JSON, find the corresponding element in the brief (e.g., by `component_id`) and set its `src` or `background-image` to the provided path.

- Output ONLY valid HTML starting with <html> and ending with </html>.
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

1. (A) the original wireframe image
2. The CURRENT HTML (single-file) that produced (B) the rendering
3. A critique ("feedback") produced by a rubric-based comparison of A vs B

Task:
Produce a NEW single-file HTML that addresses EVERY feedback point while staying faithful to A.
Fix layout fidelity (columns, spacing, alignment), completeness (ensure all components in A exist),
typography/contrast for legibility, and overall aesthetics and balance.
Keep it self-contained (inline <style>; no external CSS/JS).
Output ONLY valid HTML starting with <html> and ending with </html>.
"""

PLAN_ASSETS_SYSTEM = "You are an expert UI analyst. You extract asset requirements from a brief."
PLAN_ASSETS_PROMPT = """
Read the following UI DESIGN BRIEF. Your task is to identify all image assets required to build the page.

For each image asset, you MUST specify a unique `component_id` (e.g., "hero-image", "card-icon-1")
and a `data-asset-description` (a detailed prompt for an image search/generator).

**CRITICAL:** Respond with ONLY a valid JSON list of objects. Do not include any other text.

Example format:
[
  {{
    "component_id": "logo-nav",
    "description": "minimalist logo for a tech company named 'Innovate'",
    "bounding_box": {{"width": 150, "height": 50}}
  }},
  {{
    "component_id": "hero-background",
    "description": "photo of a modern office building exterior, sunny day",
    "bounding_box": {{"width": 1920, "height": 1080}}
  }}
]

UI DESIGN BRIEF:
---
{brief}
---

Your JSON response:
"""


@dataclass
class CodeRefineState:
    image_path: str
    out_rel_desc: str
    out_brief: str
    out_html: str
    vision_deployment: str
    text_deployment: str
    reldesc_tokens: int
    brief_tokens: int
    code_tokens: int
    judge_tokens: int
    temp: float
    refine_max_iters: int
    refine_threshold: int
    shot_width: int
    shot_height: int

    image_data_url: Optional[str] = None
    rel_desc: Optional[str] = None
    brief: Optional[str] = None
    html: Optional[str] = None
    current_iteration: int = 0
    scores: Optional[Dict[str, Any]] = None
    stop_refinement: bool = False

    find_assets: bool = False
    asset_plan: List[Dict[str, Any]] = field(default_factory=list)
    asset_paths: Dict[str, str] = field(default_factory=dict)

    messages: List[str] = field(default_factory=list)

def parse_text_report(report: str) -> Dict[str, Any]:
    sb = _section(report, "SCORES")
    scores = {k: _score_val(sb, k, 0) for k in _SCORE_KEYS}
    m_agg = re.search(r"aggregate\s*:\s*([0-9]+(?:.[0-9]+)?)", sb, flags=re.I)
    aggregate = float(m_agg.group(1)) if m_agg else sum(scores.values())/5.0

    css_patch = ""
    html_edits = ""
    regenerate_prompt = ""
    feedback = ""
    issues = ""
    layout_diffs = ""

    css_match = re.search(r"CSS_PATCH:\s*css\s+(.*?)\s+", report, flags=re.S|re.I)
    if css_match:
        css_patch = css_match.group(1).strip()
        html_edits = _section(report, "HTML_EDITS")
        regenerate_prompt = _section(report, "REGENERATE_PROMPT")
        feedback = _section(report, "FEEDBACK")
        issues = _section(report, "ISSUES_TOP3")
        layout_diffs = _section(report, "LAYOUT_DIFFS")
    return {
        "scores": scores, "aggregate": aggregate, "css_patch": css_patch, "html_edits": html_edits,
        "regenerate_prompt": regenerate_prompt, "feedback": feedback, "issues_top3": issues,
        "layout_diffs": layout_diffs, "raw": report,
    }

def refine_with_feedback(
    vision_deployment: str,
    wireframe_image: str,
    current_html: str,
    feedback: str,
    css_patch: str = "",
    html_edits: str = "",
    regenerate_prompt: str = "",
    temperature: float = 0.12,
    max_tokens: int = 2200,
) -> str:
    """
    Applies feedback (and optional CSS/HTML patches) to refine the HTML layout using GPT.
    Always safe, even if some patch strings are empty.
    """
    data_a = encode_image_to_data_url(wireframe_image)

    # Combine all feedback info into one instruction text
    refine_instructions = f"""{REFINE_PROMPT.strip()}

The following information may help you improve the HTML:
--- FEEDBACK ---
{feedback or '(none provided)'}

--- CSS_PATCH ---
{css_patch or '(none provided)'}

--- HTML_EDITS ---
{html_edits or '(none provided)'}

--- REGENERATE_PROMPT ---
{regenerate_prompt or '(none provided)'}

Now produce a refined single-file HTML that implements these improvements.
"""

    messages = [
        {"role": "system", "content": REFINE_SYSTEM},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_a}},
            {
                "type": "text",
                "text": refine_instructions
                + "\n\nCURRENT_HTML:\n```html\n"
                + current_html
                + "\n```",
            },
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

def node_stage2(state: CodeRefineState) -> CodeRefineState:
    # Convert asset_paths dict to a JSON string for the prompt
    asset_paths_json = json.dumps(state.asset_paths, indent=2)

    messages = [
        {"role": "system", "content": CODE_SYSTEM},
        {"role": "user", "content": [
            {"type":"text", "text": CODE_PROMPT.strip()},
            {"type":"text", "text": "RELATIVE LAYOUT DESCRIPTION:\n" + state.rel_desc.strip()},
            {"type":"text", "text": "UI DESIGN BRIEF:\n" + state.brief.strip()},
            # MODIFIED: Pass the actual paths, not an empty dict
            {"type":"text", "text": f"ASSET_PATHS:\n{asset_paths_json}"},
        ]},
    ]
    raw = models.chat_complete_azure(state.text_deployment, messages, state.temp, state.code_tokens)
    state.html = extract_html(raw)
    
    # Check if we generated code that *still* has placeholders (e.g., for assets we failed to find)
    if state.find_assets and 'src="placeholder"' in state.html:
        state.messages.append("Stage2: Generated HTML (with some assets included, some placeholders remain).")
    elif state.find_assets:
        state.messages.append("Stage2: Generated HTML (all assets included).")
    else:
        state.messages.append("Stage2: Generated HTML (with placeholders).")
        
    return state

def node_plan_assets_from_brief(state: CodeRefineState) -> CodeRefineState:
    print("---(Azure VLM) NODE: Planning assets from BRIEF---")
    if not state.brief:
        state.messages.append("Stage1.5: Brief is missing, cannot plan assets.")
        return state

    messages = [
        {"role": "system", "content": PLAN_ASSETS_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": PLAN_ASSETS_PROMPT.format(brief=state.brief)}
        ]},
    ]
    
    # Use the text_deployment for this JSON-only task
    raw_json = models.chat_complete_azure(state.text_deployment, messages, 0.0, 1024)
    
    try:
        asset_plan = json.loads(raw_json[raw_json.find("["):raw_json.rfind("]")+1])
        state.asset_plan = asset_plan
        state.messages.append(f"Stage1.5: Planned {len(asset_plan)} assets from Brief.")
        print(f"Asset plan: {asset_plan}")
    except json.JSONDecodeError as e:
        print(f"Error parsing asset plan JSON: {e}")
        state.messages.append(f"Stage1.5: Error parsing asset plan from brief: {e}")
        state.asset_plan = []
    
    return state

def node_stage1_find_assets(state: CodeRefineState) -> CodeRefineState:
    print("---(Azure VLM) NODE: Finding Assets---")
    if not state.asset_plan:
        state.messages.append("Stage2.6: No assets to find.")
        return state
    current_asset_paths = {}
    for asset_request in state.asset_plan:
        component_id = asset_request.get('component_id')
        desc = asset_request.get('description')
        bbox = asset_request.get('bounding_box', {})
        if not all([component_id, desc, bbox]): continue

        print(f"-> Finding asset for '{component_id}': {desc}")
        try:
            width = int(bbox.get('width', 512))
            height = int(bbox.get('height', 512))
        except (ValueError, TypeError):
            width, height = 512, 512

        result = asset_agent_app.invoke({"instructions": desc, "bounding_box": (width, height)})

        if final_path := result.get("final_asset_path"):
            current_asset_paths[component_id] = final_path
            msg = f"Asset resolved for {component_id}: {final_path}"
            state.messages.append(msg); print(f"{msg}")
        else:
            msg = f"Asset process failed for {component_id}."
            state.messages.append(msg); print(f"{msg}")

    state.asset_paths = current_asset_paths
    return state

def node_save_html_pre_score(state: CodeRefineState) -> CodeRefineState:
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
    messages = [
        {"role": "system", "content": "Return the specified PLAIN-TEXT report exactly as instructed."},
        {"role": "user", "content": [
            {"type": "text", "text": SCORING_RUBRIC.strip()},
            {"type": "image_url", "image_url":{"url": data_a}},
            {"type": "image_url", "image_url":{"url": data_b}},
            {"type": "text", "text": "HTML/CSS code used to produce image (B):\n" + state.html}
        ]},
    ]
    resp = models.chat_complete_azure(state.vision_deployment, messages, 0.0, state.judge_tokens)
    state.scores = parse_text_report(resp)
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

    #if state.asset_paths:
    #    state.messages.append(f"Re-patching assets for iteration {state.current_iteration}...")
    #    state.html, state.messages = _patch_html(state.html, state.asset_paths, state.messages)

    versioned_path = pathlib.Path(state.out_html).with_name(pathlib.Path(state.out_html).stem + f"_v{state.current_iteration}" + pathlib.Path(state.out_html).suffix)
    with open(versioned_path, "w", encoding="utf-8") as f: f.write(state.html)
    state.out_html = str(versioned_path)
    state.messages.append(f"Saved refined HTML v{state.current_iteration} -> {versioned_path}")
    return state

def decide_next(state: CodeRefineState) -> str:
    if not state.stop_refinement and state.current_iteration < state.refine_max_iters:
        return "refine_loop"
    return "end"

def route_after_brief(state: CodeRefineState) -> str:
    if state.find_assets:
        print("-> Configured to find assets. Proceeding to plan from BRIEF.")
        return "plan_assets"
    else:
        print("-> Configured to skip asset search. Proceeding to code gen.")
        state.asset_paths = {} # Ensure it's empty for stage2
        return "generate_code"

def build_azure_vlm_graph():
    workflow = StateGraph(CodeRefineState)
    workflow.add_node("stage0", node_stage0)
    workflow.add_node("stage1", node_stage1)
    
    # New node for planning from brief
    workflow.add_node("plan_assets_from_brief", node_plan_assets_from_brief)
    workflow.add_node("stage1_find_assets", node_stage1_find_assets)
    workflow.add_node("stage2", node_stage2)
    
    # Removed: plan_assets_from_html, patch_html
    
    workflow.add_node("save_html_pre_score", node_save_html_pre_score)
    workflow.add_node("stage3_score", node_stage3_score)
    workflow.add_node("refine_loop", node_refine_loop)
    
    workflow.set_entry_point("stage0")
    workflow.add_edge("stage0", "stage1")

    # NEW routing logic after brief (stage1)
    workflow.add_conditional_edges(
        "stage1",
        route_after_brief, # NEW router function
        {
            "plan_assets": "plan_assets_from_brief",
            "generate_code": "stage2" # Skip to code gen if not finding assets
        }
    )
    
    # Asset pipeline: Plan -> Find -> CodeGen
    workflow.add_edge("plan_assets_from_brief", "stage1_find_assets")
    workflow.add_edge("stage1_find_assets", "stage2")

    # NEW: stage2 (code gen) now goes directly to saving/scoring
    workflow.add_edge("stage2", "save_html_pre_score")

    # OLD 'else' branch from removed router
    workflow.add_edge("save_html_pre_score", "stage3_score")

    # Refine loop (unchanged): Score -> Refine -> (loops to Score)
    # This loop still uses _patch_html internally, which is correct
    workflow.add_edge("stage3_score", "refine_loop")
    workflow.add_conditional_edges("refine_loop", decide_next, {"refine_loop": "stage3_score", "end": END})

    return workflow.compile(checkpointer=MemorySaver())

azure_vlm_app = build_azure_vlm_graph()
#----------------------------------------------------------------------
#----------------------------------------------------------------------
## SECTION 5: MAIN "BRAIN" AGENT (Router) - uses GPT-only VLM router
#----------------------------------------------------------------------
#----------------------------------------------------------------------

class BrainState(TypedDict):
    messages: List[Dict[str, Any]]
    cli_args: argparse.Namespace
    next_task: Optional[str] = None
    task_args: Optional[Dict[str, Any]] = None
    task_result: Optional[str] = None

def helper_run_azure_vlm_pipeline(
    image_path: str, 
    find_assets: bool,
    out_html_path: str, 
    out_brief_path: str, 
    out_reldesc_path: str,
    vision_deployment: str,
    text_deployment: str,
    refine_max_iters: int,
    refine_threshold: int
) -> str:
    print(f"--- BRAIN: Invoking Azure VLM Pipeline for {image_path} (Find Assets: {find_assets}) ---")
    try:
        pathlib.Path(out_html_path).parent.mkdir(parents=True, exist_ok=True)

        state = CodeRefineState(
            image_path=image_path,
            out_rel_desc=out_reldesc_path,
            out_brief=out_brief_path,
            out_html=out_html_path,
            vision_deployment=vision_deployment,
            text_deployment=text_deployment,
            find_assets=find_assets,
            refine_max_iters=refine_max_iters,
            refine_threshold=refine_threshold,
            # Hardcoded values remain for now
            reldesc_tokens=700,
            brief_tokens=1100,
            code_tokens=2200,
            judge_tokens=900,
            temp=0.12,
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
    print(f"--- BRAIN: Invoking Code Editor for {html_path} ---")
    try:
        model_choice = "gpt-4.1-mini"
        with open(html_path, "r", encoding="utf-8") as f:
            original_html = f.read()

        initial_state = {
            "html_code": original_html,
            "user_request": edit_request,
            "model_choice": model_choice,
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

def helper_run_general_chat(user_prompt: str) -> str:
    print(f"--- BRAIN: Invoking General Chat for '{user_prompt}' ---")
    try:
        # Use the existing chat_llm method from ModelManager
        response = models.chat_llm(user_prompt, temperature=0.7)
        return response
    except Exception as e:
        print(f"Error in General Chat helper: {e}")
        return json.dumps({"status": "error", "message": str(e)})

helper_functions = [
    helper_run_azure_vlm_pipeline,
    helper_run_code_editor,
    helper_run_asset_search,
    helper_run_general_chat,
    ]

class QwenRouterAgent:
    def __init__(self, model_manager, functions, system_prompt=""):
        self.model = model_manager
        self.functions = {f.__name__: f for f in functions}
        self.system_prompt = system_prompt
    def __call__(self, state: BrainState):
        messages = state['messages']

        qwen_messages = []
        if self.system_prompt:
            qwen_messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})

        for msg in messages:
            qwen_messages.append({
                "role": msg['role'],
                "content": [{"type": "text", "text": msg['content']}]
            })

        last_user_message = messages[-1]['content']
        cli_args = state['cli_args']

        router_prompt = f"""
You are a strict JSON command router. You must look at the user's request and decide which helper
function to call, returning *only one* JSON object, with no extra text, no explanations, and no
code fences.

Valid functions and their schemas:

1. Generate a new HTML page from an image wireframe
{{
 "function_name": "helper_run_azure_vlm_pipeline",
 "function_args": {{}}
}}

2. Edit an existing HTML file
{{ "function_name": "helper_run_code_editor", "function_args": {{}} }}

3. Find or generate a single image asset
{{ "function_name": "helper_run_asset_search", "function_args": {{}} }}

4. Answer a general question, have a conversation, or respond to a greeting
{{ "function_name": "helper_run_general_chat", "function_args": {{}} }}

5. No function needed (if the request is unclear or no action is possible)
{{ "function_name": "end", "function_args": {{}} }}

Return *only* one JSON object, with no markdown or explanation.

User request: "{last_user_message}"

Image argument: {cli_args.image}
HTML argument: {cli_args.html}

Now output the JSON object only:
"""

        print("--- BRAIN: Routing prompt ---")
        vlm_response = self.model.chat_llm(router_prompt)

        print(f"--- BRAIN: VLM Response ---\n{vlm_response}\n-------------------------")
        try:
            call_json = json.loads(vlm_response[vlm_response.find("{"):vlm_response.rfind("}")+1])
            func_name = call_json.get("function_name")

            func_args = call_json.get("function_args", {})

            if func_name == "end" or func_name not in self.functions:
                print("--- BRAIN: No operation selected. Ending task. ---")
                return {"next_task": "end", "task_result": "No operation selected."}

            if func_name == "helper_run_azure_vlm_pipeline":
                func_args['find_assets'] = cli_args.find_assets
                func_args['image_path'] = cli_args.image
                func_args['out_html_path'] = cli_args.out_html
                func_args['out_brief_path'] = cli_args.out_brief
                func_args['out_reldesc_path'] = cli_args.out_rel_desc
                func_args['vision_deployment'] = cli_args.vision_deployment
                func_args['text_deployment'] = cli_args.text_deployment
                func_args['refine_max_iters'] = cli_args.refine_max_iters
                func_args['refine_threshold'] = cli_args.refine_threshold

            elif func_name == "helper_run_code_editor":
                func_args['html_path'] = cli_args.html
                func_args['edit_request'] = last_user_message
                func_args['output_path'] = cli_args.out_html

            elif func_name == "helper_run_asset_search":
                func_args['description'] = last_user_message
                func_args['width'] = 512
                func_args['height'] = 512
            
            elif func_name == "helper_run_general_chat":
                func_args['user_prompt'] = last_user_message

            return {
                "next_task": func_name,
                "task_args": func_args
            }

        except Exception as e:
            print(f"--- BRAIN: Error parsing VLM response. Ending task. --- \n{e}")
            return {"next_task": "end", "task_result": f"Error parsing VLM response: {e}"}

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

def node_run_general_chat(state: BrainState) -> dict:
    print("---(Brain Graph) NODE: node_run_general_char ---")
    args = state['task_args']
    result = helper_run_general_chat(**args)
    return {"task_result": result, "messages": state['messages'] + [{"role": "assistant", "content": result}]}
    
def brain_router(state: BrainState) -> str:
    next_task = state.get("next_task")
    print(f"---(Brain Graph) ROUTER: Next task is '{next_task}' ---")
    if next_task == "helper_run_azure_vlm_pipeline":
        return "run_vlm"
    elif next_task == "helper_run_code_editor":
        return "run_edit"
    elif next_task == "helper_run_asset_search":
        return "run_asset"
    else:
        return "run_chat"

def build_brain_graph():
    brain_agent_node = QwenRouterAgent(models, helper_functions)

    workflow = StateGraph(BrainState)

    workflow.add_node("agent", brain_agent_node)
    workflow.add_node("run_vlm", node_run_vlm_pipeline)
    workflow.add_node("run_edit", node_run_code_editor)
    workflow.add_node("run_asset", node_run_asset_search)
    workflow.add_node("run_chat", node_run_general_chat)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        brain_router,
        {
            "run_vlm": "run_vlm",
            "run_edit": "run_edit",
            "run_asset": "run_asset",
            "run_chat": "run_chat"
        }
    )

    workflow.add_edge("run_vlm", END)
    workflow.add_edge("run_edit", END)
    workflow.add_edge("run_asset", END)
    workflow.add_edge("run_chat", END)

    return workflow.compile()

brain_app = build_brain_graph()
#----------------------------------------------------------------------
#----------------------------------------------------------------------
## SECTION 6: CLI RUNNER (MAIN)
#----------------------------------------------------------------------
#----------------------------------------------------------------------

def main():
    # --- Setup Argparse ---
    parser = argparse.ArgumentParser(description="Image-to-Code Agent CLI")
    
    # Primary task argument
    parser.add_argument(
        "-p", "--prompt", 
        type=str, 
        default="Generate a code using this image.",
        help="The main user request or prompt."
    )
    
    # File inputs
    parser.add_argument(
        "-i", "--image", 
        type=str, 
        default=None, 
        help="Path to the input wireframe image (for VLM pipeline)."
    )
    parser.add_argument(
        "--html", 
        type=str, 
        default=None, 
        help="Path to the input HTML file (for code editor)."
    )
    
    # File outputs
    parser.add_argument(
        "--out_html", 
        type=str, 
        default="Outputs/agent_output.html", 
        help="Path for the final output HTML file."
    )
    parser.add_argument(
        "--out_brief", 
        type=str, 
        default="Outputs/agent_brief.txt", 
        help="Path for the output UI brief."
    )
    parser.add_argument(
        "--out_rel_desc", 
        type=str, 
        default="Outputs/agent_reldesc.txt", 
        help="Path for the output relative description."
    )

    # VLM Pipeline options
    parser.add_argument(
        "--find_assets", 
        action="store_true", 
        help="Enable the asset search/generation pipeline."
    )
    parser.add_argument(
        "--vision_deployment", 
        type=str, 
        default=GPT_DEPLOYMENT_DEFAULT, 
        help="Azure deployment name for vision tasks."
    )
    parser.add_argument(
        "--text_deployment", 
        type=str, 
        default=GPT_DEPLOYMENT_DEFAULT, 
        help="Azure deployment name for text tasks."
    )
    parser.add_argument(
        "--refine_max_iters", 
        type=int, 
        default=3, 
        help="Maximum refinement iterations."
    )
    parser.add_argument(
        "--refine_threshold", 
        type=int, 
        default=8, 
        help="Minimum score to stop refinement."
    )
    
    args = parser.parse_args()

    # --- Setup Environment ---
    print("Setting up output directories...")
    pathlib.Path("Outputs/Assets").mkdir(parents=True, exist_ok=True)
    # Ensure directories for all output files exist
    if args.out_html:
        pathlib.Path(args.out_html).parent.mkdir(parents=True, exist_ok=True)
    if args.out_brief:
        pathlib.Path(args.out_brief).parent.mkdir(parents=True, exist_ok=True)
    if args.out_rel_desc:
        pathlib.Path(args.out_rel_desc).parent.mkdir(parents=True, exist_ok=True)

    # --- Prepare Initial State ---
    print(f"--- STARTING AGENT ---")
    print(f"--- User Prompt: {args.prompt} ---")
    
    initial_state = {
        "messages": [{"role": "user", "content": args.prompt}],
        "cli_args": args  # Pass all parsed args to the agent state
    }

    # --- Run Agent ---
    run_id = f"brain-run-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    
    final_state = brain_app.invoke(initial_state, config=config)

    # --- Print Final Result ---
    print("\n" + ("-"*25) + " BRAIN INVOCATION COMPLETE " + ("-"*25))
    print("--- Final Task Result ---")
    
    task_result_str = final_state.get('task_result', "No result found (task may have ended early).")

    if task_result_str:
        try:
            # Try to parse as JSON for pretty printing
            output_json = json.loads(task_result_str)
            print(json.dumps(output_json, indent=2))
        except json.JSONDecodeError:
            # If it's not JSON (like a chat response), print raw
            print(task_result_str)
    else:
        print("No final task result recorded.")
    
    print("="*70)

if __name__ == "__main__":
    main()