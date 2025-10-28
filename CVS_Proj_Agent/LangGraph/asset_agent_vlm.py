import os
import json
import base64
import io
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from io import BytesIO
from PIL import Image
from uuid import uuid4
import requests

# LangGraph / LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# HF Transformers & Diffusers
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer, pipeline, BitsAndBytesConfig
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Configuration
# -----------------------------
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
SD_GENERATOR_MODEL = os.getenv("SD_GENERATOR_MODEL", "segmind/tiny-sd")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

# -----------------------------
# UNIFIED Graph State
# -----------------------------
class GraphState(TypedDict):
    """A single, unified state for the entire graph."""
    image_path: str
    image_b64: Optional[str]
    description_json: Optional[Dict[str, Any]]
    plan_json: Optional[Dict[str, Any]]
    html_css: Optional[str]
    asset_paths: Dict[str, str]
    messages: List[str]
    instructions: str
    search_query: str
    bounding_box: Tuple[int, int]
    found_image_url: Optional[str]
    asset_is_valid: bool
    final_asset_path: Optional[str]

# -----------------------------
# Model Loaders
# -----------------------------
class OpenSourceModels:
    """Manages loading all models at startup."""
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(OpenSourceModels, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'vlm_model'): # Initialize only once
            print("Initializing and loading all models into VRAM...")
            
            # 1. Configure and load the VLM with 4-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )
            self.vlm_processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_NAME, trust_remote_code=True)
            self.vlm_model = AutoModelForVision2Seq.from_pretrained(
                QWEN_VL_MODEL_NAME,
                torch_dtype=DTYPE,
                device_map="auto",
                quantization_config=quantization_config,
                trust_remote_code=True
            )
            print("VLM loaded.")

            # 2. Configure and load the Generator with CPU offloading
            self.generator_pipe = DiffusionPipeline.from_pretrained(
                SD_GENERATOR_MODEL, torch_dtype=DTYPE
            )
            self.generator_pipe.enable_model_cpu_offload()
            print("Generator loaded.")
            
            print("All models loaded and ready.")

    def chat_vlm(self, messages, temperature=0.2, max_new_tokens=2048):
        gen_kwargs = {"do_sample": temperature > 0, "max_new_tokens": max_new_tokens}
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        inputs = self.vlm_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(self.vlm_model.device)
        
        with torch.no_grad():
            output_ids = self.vlm_model.generate(**inputs, **gen_kwargs)

        gen_only = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output_ids)]
        return self.vlm_processor.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    def chat_llm(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.chat_vlm(messages, temperature=0.1, max_new_tokens=50)

    def generate_image(self, prompt: str) -> Image.Image:
        print(f"Generating image with prompt: '{prompt}'")
        return self.generator_pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
        
models = OpenSourceModels()

# -----------------------------
# Utilities
# -----------------------------
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def b64img(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# -----------------------------
# Main Graph Nodes
# -----------------------------
def node_perception(state: GraphState) -> dict:
    img = load_image(state['image_path'])
    image_b64 = b64img(img)
    prompt = """{ "page": { "size": {"width": <w>, "height": <h>}, "components": [ { "id": "<id>", "type": "image", ... } ] } }"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a UI perception expert."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this wireframe image and produce a STRICT JSON object describing its UI components."},
            {"type": "image", "image": image_b64},
            {"type": "text", "text": prompt}
        ]}
    ]
    resp = models.chat_vlm(messages, temperature=0.1, max_new_tokens=2000)
    parsed = json.loads(resp[resp.find("{"):resp.rfind("}")+1])
    print("Perception complete.")
    return {"description_json": parsed, "image_b64": image_b64, "messages": state['messages'] + ["Perception complete."]}

def node_planner(state: GraphState) -> dict:
    assert state['description_json'], "Missing description_json"
    prompt = """Return a STRICT JSON object. Your most important task is to correctly populate the `assets_needed` list.
**Your instructions are:**
1.  Go through **every single object** in the `components` array from the input JSON.
2.  For **each** object, check the value of its `"type"` key.
3.  If `type` is exactly `"image"`, you **MUST** create a corresponding entry in the `assets_needed` array.
4.  Derive the `description` and `bounding_box` from that component's data.
**Only if NO components have `type: "image"` should you return an empty list.**
**JSON Schema:**
{ "order": ["<id>"], "assets_needed": [ { "component_id": "<id_of_image_component>", "description": "<description>", "bounding_box": {"width": "<w>", "height": "<h>"} } ] }"""
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a front-end architect."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Given the component JSON, return a build plan by following the rules precisely."},
            {"type": "text", "text": json.dumps(state['description_json'], ensure_ascii=False, indent=2)},
            {"type": "text", "text": prompt}
        ]}
    ]
    resp = models.chat_vlm(messages, temperature=0.1, max_new_tokens=1500)
    parsed = json.loads(resp[resp.find("{"):resp.rfind("}")+1])
    print("Planner complete.")
    return {"plan_json": parsed, "messages": state['messages'] + ["Planner complete."]}

def node_codegen(state: GraphState) -> dict:
    assert state['plan_json'] and state['description_json'], "Missing inputs for codegen"
    code_prompt = "You are a senior UI engineer. Generate a SINGLE self-contained HTML file with <style>. Follow the plan exactly. When generating an `<img>` tag for a component ID found in the `asset_paths` JSON, you MUST use the provided local file path in the `src` attribute. For all other images, use a placeholder."
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You write production-quality HTML+CSS."}]},
        {"role": "user", "content": [
            {"type": "text", "text": code_prompt},
            {"type": "text", "text": f"Component JSON: {json.dumps(state['description_json'], ensure_ascii=False)}"},
            {"type": "text", "text": f"Build plan JSON: {json.dumps(state['plan_json'], ensure_ascii=False)}"},
            {"type": "text", "text": f"Asset paths JSON: {json.dumps(state['asset_paths'], ensure_ascii=False)}"},
            {"type": "text", "text": "Now return only the final HTML."},
        ]}
    ]
    resp = models.chat_vlm(messages, temperature=0.15, max_new_tokens=3000)
    start = resp.find("<html")
    if start == -1: start = resp.find("<!DOCTYPE")
    end = resp.rfind("</html>")
    html = resp[start:end+7] if end != -1 and start != -1 else resp
    print("Codegen complete.")
    return {"html_css": html, "messages": state['messages'] + ["Codegen complete."]}

# -------------------------------------------------------------
# OPEN-SOURCE ASSET AGENT WORKFLOW
# -------------------------------------------------------------
def prepare_search_query_node(state: GraphState) -> dict:
    print("---NODE: Prepare Search Query---")
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

def generate_image_node(state: GraphState) -> dict:
    print("---NODE: Generate Image---")
    prompt = state["instructions"]
    generated_image = models.generate_image(prompt)
    output_dir = "Outputs/Assets"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"generated_{uuid4()}.png"
    output_path = os.path.join(output_dir, filename)
    generated_image.save(output_path)
    print(f"Image generated and saved to {output_path}")
    return {"final_asset_path": output_path}
    
def download_and_resize_node(state: GraphState) -> dict:
    print("---NODE: Download and Resize---")
    image_url = state.get("found_image_url")
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        img.thumbnail(state['bounding_box'])
        output_dir = "Outputs/Assets"
        filename = f"asset_{uuid4()}.png"
        output_path = os.path.join(output_dir, filename)
        img.save(output_path)
        print(f"Image saved and resized to {output_path}")
        return {"final_asset_path": output_path}
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"final_asset_path": None}

def route_after_search(state: GraphState) -> str:
    if state.get("found_image_url"):
        return "download_and_resize"
    else:
        print("Search failed. Routing to generate a new image.")
        return "generate_image"
        
def pexels_search_node(state: GraphState) -> dict:
    print("---TOOL: Searching Pexels---")
    api_key = os.getenv("PEXELS_API_KEY")
    search_query = state.get("search_query")
    if not api_key or not search_query:
        return {"found_image_url": None}

    headers = {"Authorization": api_key}
    params = {"query": search_query, "per_page": 1}
    try:
        response = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        if response.json().get('photos'):
            image_url = response.json()['photos'][0]['src']['original']
            print(f"Found a candidate image: {image_url}")
            return {"found_image_url": image_url}
    except requests.exceptions.RequestException as e:
        print(f"Pexels API Error: {e}")
    return {"found_image_url": None}

asset_agent_app = None
def init_asset_agent_workflow():
    global asset_agent_app
    if asset_agent_app:
        return asset_agent_app
    workflow = StateGraph(GraphState)
    workflow.add_node("prepare_search_query", prepare_search_query_node)
    workflow.add_node("pexels_search", pexels_search_node)
    workflow.add_node("generate_image", generate_image_node)
    workflow.add_node("download_and_resize", download_and_resize_node)
    workflow.set_entry_point("prepare_search_query")
    workflow.add_edge("prepare_search_query", "pexels_search")
    workflow.add_conditional_edges("pexels_search", route_after_search)
    workflow.add_edge("generate_image", END)
    workflow.add_edge("download_and_resize", END)
    asset_agent_app = workflow.compile()
    return asset_agent_app

# -----------------------------
# BRIDGE NODE & MAIN GRAPH
# -----------------------------
def node_asset_search(state: GraphState) -> dict:
    print("--- Starting Asset Search Loop ---")
    assets_to_find = state['plan_json'].get("assets_needed", [])
    current_asset_paths, updated_messages = state.get('asset_paths', {}), state.get('messages', [])
    
    # Initialize the agent once before the loop
    agent = init_asset_agent_workflow()

    for asset_request in assets_to_find:
        component_id, desc, bbox = asset_request.get('component_id'), asset_request.get('description'), asset_request.get('bounding_box', {})
        if not all([component_id, desc, bbox]): continue
        print(f"-> Finding asset for '{component_id}': {desc}")
        
        try:
            width = int(bbox.get('width', 512))
            height = int(bbox.get('height', 512))
        except (ValueError, TypeError):
            print(f"Warning: Could not parse bounding box for {component_id}. Using default 512x512.")
            width, height = 512, 512
            
        result = agent.invoke({"instructions": desc, "bounding_box": (width, height)})
        
        if final_path := result.get("final_asset_path"):
            current_asset_paths[component_id] = final_path; msg = f"Asset resolved for {component_id}: {final_path}"
            updated_messages.append(msg); print(f"  ✅ {msg}")
        else:
            msg = f"Asset process failed for {component_id}."; updated_messages.append(msg); print(f"  error: {msg}")
            
    return {"asset_paths": current_asset_paths, "messages": updated_messages}

def route_to_assets_or_codegen(state: GraphState) -> str:
    if state.get('plan_json', {}).get("assets_needed"):
        return "asset_search"
    return "codegen"

# -----------------------------
# Build and Run Main Graph
# -----------------------------
workflow = StateGraph(GraphState)
workflow.add_node("perception", node_perception)
workflow.add_node("planner", node_planner)
workflow.add_node("asset_search", node_asset_search)
workflow.add_node("codegen", node_codegen)

workflow.set_entry_point("perception")
workflow.add_edge("perception", "planner")
workflow.add_conditional_edges(
    "planner",
    route_to_assets_or_codegen,
    {
        "asset_search": "asset_search",
        "codegen": "codegen"
    }
)
workflow.add_edge("asset_search", "codegen")
workflow.add_edge("codegen", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    import argparse, pathlib
    pathlib.Path("Outputs/Assets").mkdir(parents=True, exist_ok=True)
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=False, help="Path to wireframe PNG/JPG", default="Images/asset_test.png")
    p.add_argument("--out_html", default="Outputs/generated_asset_test.html")
    args = p.parse_args()

    initial_state = {
        "image_path": str(pathlib.Path(args.image).resolve()),
        "messages": [],
        "asset_paths": {},
    }
    
    config = {"configurable": {"thread_id": f"wireframe-{uuid4()}"}}
    final = app.invoke(initial_state, config=config)

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(final['html_css'] or "")

    print("\n" + "="*50)
    print("--- PIPELINE COMPLETE ---")
    print("\n".join(final['messages']))
    print(f"\nSaved final HTML to: {args.out_html}")
    print("="*50)