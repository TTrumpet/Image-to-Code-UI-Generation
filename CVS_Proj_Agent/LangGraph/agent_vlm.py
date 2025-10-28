# wireframe_to_code_graph.py
import os, json, base64, io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from uuid import uuid4

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# HF Transformers (Qwen2.5-VL)
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

# -----------------------------
# Configuration
# -----------------------------
# Model names: choose an 8B/7B Qwen2.5-VL Instruct variant you have access to.
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")  # use 8B-equivalent if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# Optional: pin transformers if you hit regressions (see notes in README).
# pip install "transformers==4.49.0"

# -----------------------------
# Utilities
# -----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def pil_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

def b64img(pil_img: Image.Image) -> str:
    return base64.b64encode(pil_to_bytes(pil_img)).decode("utf-8")

def ensure_list(x):
    return x if isinstance(x, list) else [x]

# -----------------------------
# Qwen2.5-VL loader (single instance reused by nodes)
# -----------------------------
class QwenVL:
    def __init__(self, model_name: str = QWEN_VL_MODEL_NAME, device: str = DEVICE, dtype: torch.dtype = DTYPE):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )

    def chat(self, messages, temperature=0.2, max_new_tokens=2048):
        # messages must have typed content blocks as above
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.model.device)
    
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=temperature > 0,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
    
        # slice off the prompt tokens before decoding (per HF example)
        gen_only = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output_ids)
        ]
        text = self.processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        return text


# Singleton (so we don’t reload per node)
qwen_vl = QwenVL()

# -----------------------------
# LangGraph State
# -----------------------------
@dataclass
class UI2CodeState:
    image_path: str
    image_b64: Optional[str] = None

    # Step outputs
    description_json: Optional[Dict[str, Any]] = None  # parsed components
    plan_json: Optional[Dict[str, Any]] = None         # ordered build plan
    html_css: Optional[str] = None                     # final HTML+CSS

    # Trace
    messages: List[str] = field(default_factory=list)

# -----------------------------
# Node 1 — Perception: JSON UI description
# -----------------------------
def node_perception(state: UI2CodeState) -> UI2CodeState:
    img = load_image(state.image_path)
    print("image loaded")
    state.image_b64 = b64img(img)

    sys = {
        "role": "system",
        "content":[{"type": "text", "text": "You are a UI perception expert. Return precise JSON describing UI elements from a wireframe."}]
    }

    user = {"role": "user",
            "content": [
                {"type": "text", "text": "Analyze this wireframe image and produce a STRICT JSON of UI components."},
                {"type": "image", "image": img},
                {"type": "text", "text": """
Output JSON with this schema exactly:
{
  "page": {
    "size": {"width": <int_px>, "height": <int_px>},
    "components": [
      {
        "id": "comp_1",
        "type": "container|navbar|header|footer|section|div|grid|card|table|form|input|button|icon|image|text",
        "role": "logo|menu|hero|sidebar|content|cta|caption|paragraph|label|",
        "bbox": {"x": <int>, "y": <int>, "w": <int>, "h": <int>},
        "props": {"variant":"outlined|filled|text", "icon":"home|search|user|...", "text":"..."},
        "children": ["comp_2","comp_3"]
      }
    ]
  }
}
Only return JSON. No explanations.
"""
}]}


    resp = qwen_vl.chat([sys, user], temperature=0.1, max_new_tokens=1500)
    # Make it robust to stray text
    try:
        json_str = resp[resp.find("{"):resp.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"page": {"size": {}, "components": []}, "raw": resp}

    state.description_json = parsed
    state.messages.append("Perception complete: extracted JSON.")

    print("done describing the UI components in the image")
    return state

# -----------------------------
# Node 2 — Planner: order & hierarchy for code emission
# -----------------------------
def node_planner(state: UI2CodeState) -> UI2CodeState:
    assert state.description_json, "Missing description_json"

    sys = {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a front-end architect. Produce an ordered build plan from component JSON."}
        ],
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Given the component JSON, return a build plan (top->bottom, left->right) with hierarchy and semantic regions."},
            {"type": "text", "text": json.dumps(state.description_json, ensure_ascii=False, indent=2)},
            {"type": "text", "text": """
Return STRICT JSON:

{
  "order": ["comp_5","comp_2","comp_7", ...],
  "regions": [
    {"name":"header","components":["comp_5","comp_2"]},
    {"name":"sidebar","components":[...]}
  ],
  "hierarchy": [
    {"parent":"comp_1","children":["comp_2","comp_3"]},
    ...
  ],
  "css_guidelines": {
    "colors": ["#f5f5f5","#222", ...],
    "fonts": ["Inter","Arial","system-ui"],
    "layout": "CSS Grid or Flexbox, with responsive breakpoints at 768px and 1024px",
    "style": "Material-ish, subtle shadows, 8px radius"
  }
}
Only JSON. No text.
"""}
    ],
}


    resp = qwen_vl.chat([sys, user], temperature=0.2, max_new_tokens=1200)
    try:
        json_str = resp[resp.find("{"):resp.rfind("}")+1]
        parsed = json.loads(json_str)
    except Exception:
        parsed = {"order": [], "regions": [], "hierarchy": [], "css_guidelines": {}, "raw": resp}

    state.plan_json = parsed
    state.messages.append("Planner complete: produced build order and guidelines.")
    return state

# -----------------------------
# Node 3 — Codegen: HTML+CSS from plan
# -----------------------------
CODE_PROMPT = """You are a senior UI engineer.
Generate a SINGLE self-contained HTML file with <style> (no external CSS/JS).
Follow the plan exactly. Use semantic tags where sensible. Prefer CSS Grid/Flex.
Add minimal responsive rules (≤ 60 lines CSS if possible).
Include placeholder text/images where needed, using accessible alt/aria.
"""

def node_codegen(state: UI2CodeState) -> UI2CodeState:
    assert state.plan_json and state.description_json, "Missing inputs for codegen"

    sys = {
        "role": "system",
        "content": [{"type": "text", "text": "You write production-quality HTML+CSS from structured UI plans."}],
    }
    user = {
        "role": "user",
        "content": [
            {"type": "text", "text": CODE_PROMPT},
            {"type": "text", "text": "Component description JSON:"},
            {"type": "text", "text": json.dumps(state.description_json, ensure_ascii=False, indent=2)},
            {"type": "text", "text": "Ordered build plan JSON:"},
            {"type": "text", "text": json.dumps(state.plan_json, ensure_ascii=False, indent=2)},
            {"type": "text", "text": "Now return only the final HTML."},
        ],
    }


    # We can reuse Qwen2.5-VL for code generation; it’s text-only here.
    resp = qwen_vl.chat([sys, user], temperature=0.15, max_new_tokens=2500)
    # Light post-filter to keep just the HTML
    start = resp.find("<html")
    if start == -1:
        start = resp.find("<!DOCTYPE")
    end = resp.rfind("</html>")
    html = resp[start:end+7] if end != -1 and start != -1 else resp

    state.html_css = html
    state.messages.append("Codegen complete: produced HTML+CSS.")
    return state

# -----------------------------
# Build the LangGraph
# -----------------------------
workflow = StateGraph(UI2CodeState)
workflow.add_node("perception", node_perception)
workflow.add_node("planner", node_planner)
workflow.add_node("codegen", node_codegen)

workflow.set_entry_point("perception")
workflow.add_edge("perception", "planner")
workflow.add_edge("planner", "codegen")
workflow.add_edge("codegen", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# -----------------------------
# CLI Runner
# -----------------------------
if __name__ == "__main__":
    import argparse, pathlib
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=False, help="Path to wireframe PNG/JPG", default="Images/1.png")
    p.add_argument("--out_html", default="Outputs/generated_1.html")
    args = p.parse_args()

    img_path = str(pathlib.Path(args.image).resolve())
    state = UI2CodeState(image_path=img_path)

    run_id = f"wireframe-{uuid4()}"
    config = {"configurable": {"thread_id": run_id}}
    final = app.invoke(state, config=config)

    # Save HTML
    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(final['html_css'] or "<!-- empty -->")

    # Log
    print("\n--- PIPELINE MESSAGES ---")
    print("\n".join(final['messages']))
    print(f"\nSaved: {args.out_html}")
