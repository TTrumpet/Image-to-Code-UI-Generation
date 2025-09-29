import base64
import requests
import os

from constants import *

# This file contains mock functions for making API calls.
# TODO: Replace these with your actual model inference logic.

def image_to_base64_str(filepath):
    """Converts an image file to a base64 string."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found at {filepath}")

    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_vlm_api(prompt: str, image_path: str | None):
    """Placeholder for a call to a powerful VLM like GPT-4o."""
    print(f"  [Mock VLM Call] Prompt: {prompt[:80]}...")
    if image_path:
        print(f"  [Mock VLM Call] with image: {image_path}")
    
    # MOCK RESPONSE LOGIC
    if "plan" in prompt.lower():
        return "1. Find the eye. 2. Draw a circle for the monocle around the eye. 3. Draw a line for the chain."
    else:
        return [
            "1. Use Image.open() to load the image.",
            "2. Identify approximate coordinates for the person's right eye.",
            "3. Use ImageDraw.Draw to create a drawing context.",
            "4. Draw an ellipse (circle) for the monocle lens.",
            "5. Draw a line for the chain extending from the monocle.",
        ]

def call_codegen_api(prompt: str):
    """Placeholder for a call to a code generation model."""
    print(f"  [Mock CodeGen Call] Prompt: {prompt[:80]}...")
    
    # MOCK RESPONSE (this code works if you have an image at the path)
    return """
from PIL import Image, ImageDraw

# WARNING: This is generated code and could be unsafe.
# Always run in a sandboxed environment.

try:
    img_path = 'path/to/your/image.jpg' # IMPORTANT: Match the path in agent.py
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)

    # These are hardcoded coordinates; a real agent would make them dynamic
    eye_center = (200, 250)
    radius = 30
    
    # Draw monocle lens
    draw.ellipse(
        (eye_center[0]-radius, eye_center[1]-radius, eye_center[0]+radius, eye_center[1]+radius), 
        outline='black', 
        width=5
    )
    
    # Draw chain
    draw.line(
        (eye_center[0]+radius, eye_center[1], eye_center[0]+radius+50, eye_center[1]+50),
        fill='black',
        width=3
    )
    
    img.save('output.png')
    print("Image with monocle saved to output.png")

except FileNotFoundError:
    print(f"Error: The file at {img_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
"""

def call_vllm_comparison_api(vllm_client, prompt: str, original_image_path: str, generated_image_path: str):
    """
    Calls the local vLLM server with two images and a prompt to get a comparison.
    """
    if not vllm_client:
        return "Error: vLLM client not initialized. Cannot perform comparison."
        
    print(f"  [Real vLLM Critic Call] Comparing {original_image_path} and {generated_image_path}...")

    try:
        # Encode both images
        base64_original = image_to_base64_str(original_image_path)
        base64_generated = image_to_base64_str(generated_image_path)

        # Send the request with the prompt and both images
        response = vllm_client.chat.completions.create(
            model=EVAL_MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Original Image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_original}"}
                        },
                        {"type": "text", "text": "Generated Image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_generated}"}
                        },
                        {"type": "text", "text": f"\n\n{prompt}"}
                    ]
                }
            ],
            max_tokens=100, # Feedback should be concise
            temperature=0.2, # Low temperature for deterministic feedback
        )
        
        feedback = response.choices[0].message.content.strip()
        return feedback

    except FileNotFoundError as e:
        print(f"Error during VLM call: {e}")
        return f"Error: Image file not found. {e}"
    except Exception as e:
        print(f"An error occurred during the VLM call: {e}")
        return f"Error: Could not get a response from the vLLM server. {e}"
