import subprocess
from utils import call_vlm_api, call_codegen_api, call_vllm_comparison_api

# This file contains the functions that will be the nodes in our graph.
# Each class has a `run` method that takes the current state and returns an updated state.

class VlmBrain:
    def run(self, state):
        print(f"\n--- Iteration {state['iteration']} | Running Brain ---")
        
        # Prepare the prompt for the brain
        prompt = f"""You are the 'Brain' of an AI image editing agent.
        Your goal is to create a high-level plan to achieve the user's request.
        
        User Request: "{state['prompt']}"
        Previous Attempt Feedback: "{state['feedback']}"
        
        Analyze the request and feedback, then create a simple, step-by-step plan.
        """
        
        # This is a placeholder for your actual API call
        plan = call_vlm_api(
            prompt=prompt,
            image_path=state['original_image_path']
        )
        
        return {
            "plan": plan, 
            "iteration": state["iteration"] + 1,
            "error": None, # Clear error from previous run
        }

class VlmReasoning:
    def run(self, state):
        print("--- Running Reasoning ---")
        prompt = f"""You are the 'Code Strategist'. Your job is to convert a high-level plan into a detailed, concrete strategy for a Python coder using the Pillow library.

        High-level plan: "{state['plan']}"

        Provide a numbered list of precise instructions for the coder.
        """
        reasoning_steps = call_vlm_api(prompt=prompt, image_path=None) # No image needed here
        return {"reasoning_steps": reasoning_steps}

class CodeGenerator:
    def run(self, state):
        print("--- Generating Code ---")
        prompt = f"""You are a Python code generation expert specializing in the Pillow (PIL) library. Write a complete, executable Python script based on the following strategy. The script must load an image from '{state['original_image_path']}' and save the final image to 'output.png'.

        Strategy:
        {state['reasoning_steps']}
        """
        code = call_codegen_api(prompt=prompt)
        return {"code": code}

class CodeExecutor:
    def run(self, state):
        print("--- Executing Code ---")
        try:
            # Write the generated code to a file
            with open("generated_script.py", "w") as f:
                f.write(state["code"])
            
            # Execute in a sandbox! This is a simplified, UNSAFE example.
            # For a real application, use a Docker container or similar sandbox.
            result = subprocess.run(
                ["python", "generated_script.py"],
                capture_output=True, text=True, timeout=30, check=True
            )
            print("Code execution successful.")
            return {
                "generated_image_path": "output.png", 
                "error": None
            }
        except subprocess.CalledProcessError as e:
            # Capture errors if the script fails
            error_message = f"Execution failed:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
            print(error_message)
            return {
                "error": error_message, 
                "feedback": f"The generated code failed with an error: {e.stderr[-200:]}. Please generate different code to fix this."
            }
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {
                "error": str(e),
                "feedback": f"An unexpected executor error occurred: {str(e)}. Please try a different approach."
            }


class VlmComparer:
    def run(self, state):
        print("--- Comparing Images (vLLM Critic) ---")
        prompt = f"""You are an AI critic. Your task is to determine if the 'Generated Image' successfully fulfills the original user request when compared to the 'Original Image'.
        
        User Request: "{state['prompt']}"

        Your answer must be one of two things:
        1. The single word "success" if the request is fully satisfied.
        2. A brief, constructive feedback sentence explaining what is wrong if it is not.
        """
        
        feedback = call_vllm_comparison_api(
            prompt=prompt,
            original_image_path=state["original_image_path"],
            generated_image_path=state["generated_image_path"]
        )
        return {"feedback": feedback}

