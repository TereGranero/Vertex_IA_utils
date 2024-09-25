"""
VERTEX AI utils
"""
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import generative_models
import json


def generate_vertex_generative_text(max_output_tokens, system_instruction, prompt, tools=[]):
    """Uses Vertex AI Generative Model gemini-1.5-flash-001
    to generate text using the given prompt and system instruction.
    
    Args:
        max_output_tokens (int): The maximum number of tokens to generate.
        system_instruction (str): The instruction to use for the model.
        prompt (str): The prompt to use for the model.
        tools (list): The tools to use for the model. OPTIONAL

    Returns:
        str: The generated text.
        str: error message or None.
    """

    # ---------------------------- Initialize Vertex ------------------------------

    aiplatform.init(project = "Enter here your PROJECT_ID", location = "Enter your GCP REGION")

    # Model
    model = GenerativeModel("gemini-1.5-flash-001",
        system_instruction=[system_instruction])
    
    # Hyperparameters
    hyperparameters = {
            "max_output_tokens": max_output_tokens,
            "temperature": 0.1,
            "top_p": 1
        }

    safety_settings= {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    }

    # ------------------------------- Generate text -------------------------------

    try:
        if len(tools) > 0:
               
            if "Google_Search" in tools:
                search_tool = Tool.from_google_search_retrieval(
                    google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval(disable_attribution=False)
                )
                my_tools = [search_tool]

                response = model.generate_content(
                    prompt,
                    generation_config=hyperparameters,
                    tools = my_tools,
                    safety_settings=safety_settings
                )
        else:
            response = model.generate_content(
                prompt,
                generation_config=hyperparameters,
                safety_settings=safety_settings
            )

        return response.text, None
    except Exception as e:
        error_message = (f"Error vertex: {str(e)}")
        return None, error_message

