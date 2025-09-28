# This script defines an agentic workflow for an Image-to-Code task
# using LangGraph. The agent takes a prompt and an image, generates
# Python code to modify the image, executes it, and then uses a VLM
# to check the result, looping until the task is complete.
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Import the node functions we will define in another file
from nodes import VlmBrain, VlmReasoning, CodeGenerator, CodeExecutor, VlmComparer

# Define the state that will be passed between nodes in our graph
class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    prompt: str
    original_image_path: str
    generated_image_path: str | None
    plan: str
    reasoning_steps: list[str]
    code: str
    feedback: str
    error: str | None
    iteration: int

# --- Instantiate Nodes ---
# Each class corresponds to a step in our loop.
vlm_brain = VlmBrain()
vlm_reasoning = VlmReasoning()
code_generator = CodeGenerator()
code_executor = CodeExecutor()
vlm_comparer = VlmComparer()

# --- Define Graph and Nodes ---
workflow = StateGraph(AgentState)

# Add each function as a node in the graph
workflow.add_node("brain", vlm_brain.run)
workflow.add_node("reasoning", vlm_reasoning.run)
workflow.add_node("generate_code", code_generator.run)
workflow.add_node("execute_code", code_executor.run)
workflow.add_node("compare_images", vlm_comparer.run)

# --- Define Edges ---
workflow.set_entry_point("brain")
workflow.add_edge("brain", "reasoning")
workflow.add_edge("reasoning", "generate_code")
workflow.add_edge("generate_code", "execute_code")

# --- Define Conditional Edges ---

# 1. After executing code, check if it ran successfully.
def check_code_execution(state: AgentState):
    if state.get("error"):
        # If there was an error, go back to the brain with feedback
        print("Code execution failed. Looping back to brain.")
        return "brain"
    # Otherwise, proceed to comparison
    return "compare_images"

workflow.add_conditional_edges(
    "execute_code",
    check_code_execution,
    {
        "brain": "brain",
        "compare_images": "compare_images"
    }
)

# 2. After comparing images, check if the task is complete.
def check_image_comparison(state: AgentState):
    if state["feedback"].lower() == "success":
        print("Task completed successfully!")
        return END
    # If not successful, loop back to the brain for another attempt
    print(f"Comparison failed. Feedback: '{state['feedback']}'. Looping back.")
    return "brain"

workflow.add_conditional_edges(
    "compare_images",
    check_image_comparison,
    {
        END: END,
        "brain": "brain"
    }
)

# --- Compile and Run ---
app = workflow.compile()

# Define the initial input for the agent
initial_input = {
    "prompt": "Task: Image-to-Code. Goal: Generate Python code to add a monocle to the person in the image.",
    "original_image_path": "path/to/your/image.jpg", # <--- CHANGE THIS
    "iteration": 0,
    "feedback": "First attempt." # Initial feedback
}

# The recursion_limit sets the maximum number of loops
for s in app.stream(initial_input, {"recursion_limit": 5}):
    print("---")
    print(s)


