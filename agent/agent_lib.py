
import asyncio
import json
import os
import random
import re
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from langchain_core.language_models import BaseChatModel

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agent.types import TextComparisonResult, TextDiffResult, VisualComparisonResult

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock").lower()


def get_llm(provider: str) -> BaseChatModel:
    """
    Factory function to get an instance of a LangChain chat model based on the provider.
    """
    print(f"  -> Initializing LLM for provider: {provider}")
    if provider == "bedrock":
        BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        return ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            model_kwargs={"temperature": 0.0},
            # It's good practice to have your region configured via AWS CLI or env vars
            # region_name="us-east-1",
        )
    elif provider == "openai":
        # Assumes OPENAI_API_KEY is set in your environment
        OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4o")
        return ChatOpenAI(
            model=OPENAI_MODEL_ID,
            temperature=0.0
        )
    elif provider == "gemini":
        GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-1.5-pro-latest")
        # Assumes GOOGLE_API_KEY is set in your environment
        return ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_ID,
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}. Supported providers are 'bedrock', 'openai', 'gemini'.")

# Initialize the LLM based on the environment configuration
LLM = get_llm(LLM_PROVIDER)

def parse_llm_json_output(json_string: str, output_model: BaseModel) -> Optional[BaseModel]:
    match = re.search(r"```(json)?(.*)```", json_string, re.DOTALL)
    clean_json_string = match.group(2).strip() if match else json_string.strip()
    try:
        data = json.loads(clean_json_string)
        status_fields = ["VISUAL_DIFF_STATUS", "SEMANTIC_DIFF_STATUS", "TEXT_DIFF_STATUS"]
        synonym_map = {
            "NO_DIFFERENCE": "IDENTICAL", "NONE": "IDENTICAL", "SAME": "IDENTICAL",
            "SLIGHT_DIFFERENCE": "MINOR_DIFFERENCE", "LARGE_DIFFERENCE": "VAST_DIFFERENCE",
            "MAJOR_DIFFERENCE": "VAST_DIFFERENCE"
        }
        for field in status_fields:
            if field in data and data[field] in synonym_map:
                original_value = data[field]
                corrected_value = synonym_map[original_value]
                print(f"  -> Correcting status '{original_value}' to '{corrected_value}' in field '{field}'.")
                data[field] = corrected_value
        return output_model(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        print(f"Error parsing/validating LLM output. Error: {e}. Raw output: '{clean_json_string}'")
        return None

# --- Schemas for Prompts ---
text_schema_json = TextComparisonResult.schema_json(indent=2)
visual_schema_json = VisualComparisonResult.schema_json(indent=2)
diff_schema_json = TextDiffResult.schema_json(indent=2)

# --- Prompt Templates ---
text_analysis_prompt = ChatPromptTemplate.from_template(
"""You are a neutral and objective text comparison engine. Your function is to determine the semantic relationship between two texts and report it in a structured JSON format.
You MUST respond with a single, valid JSON object that conforms to the schema below and nothing else. Do not add any explanatory text.

**JSON Schema:**
```json
{text_schema}
Task:
Perform a semantic analysis of the following text blocks and format your response according to the schema.
Expected Text:
{text_prod}
Actual Text:
{text_staging}"""
)

diff_summarizer_prompt = ChatPromptTemplate.from_template(
"""You are an expert assistant that explains code diffs in plain English.
You will be given a diff output where lines starting with '-' were removed, and lines starting with '+' were added.
Your task is to summarize these changes clearly and concisely.

You MUST respond with a single, valid JSON object that conforms to the schema below.
{diff_schema}
Diff Output to Summarize:
{diff_output}
"""
)


visual_analysis_prompt = ChatPromptTemplate.from_messages([
("system", "You are a QA Engineer. Your task is to analyze a 'diff image' that highlights visual differences between two website screenshots."),
("human", [
{"type": "text", "text": """
Diff images is provided: This image highlights all changed areas in full color against a grayscale background.
**Your Task:**
    -   Look at the **Diff Image**.
    -   Describe what the **full-color highlighted areas** represent.
    -   Based on your analysis, provide a structured JSON response. The `VISUAL_DIFF_STATUS` must be "MINOR_DIFFERENCE" or "VAST_DIFFERENCE", as a diff image is only provided when changes exist.
    -   Your entire response must be only the JSON object.
    
    **JSON Schema:**
    ```json
    {visual_schema}
    ```
    """},
    {"type": "text", "text": "Diff Image (changes are in color):"},
    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "{diff_image_b64}"}}
])
])

semantic_text_analyzer_chain = text_analysis_prompt | LLM | StrOutputParser()
literal_text_analyzer_chain = diff_summarizer_prompt | LLM | StrOutputParser()
visual_analyzer_chain = visual_analysis_prompt | LLM | StrOutputParser()




async def invoke_with_retry_async(chain, params: dict, retries: int = 3, base_delay: float = 2.0, max_delay: float = 60.0):
    """
    Invokes a LangChain chain asynchronously with an exponential backoff and jitter retry mechanism.
    """
    last_exception = None
    for attempt in range(retries):
        try:
            result = await chain.ainvoke(params)
            return result
        except Exception as e:
            last_exception = e
            print(f"    -> ⚠️ API call attempt {attempt + 1}/{retries} failed: {e}")
            
            if attempt < retries - 1:
                # Calculate exponential backoff
                backoff_delay = base_delay * (2 ** attempt)
                
                # Add jitter (random value between 0 and 1 second)
                jitter = random.uniform(0, 1)
                
                # Calculate total delay, ensuring it doesn't exceed max_delay
                total_delay = min(backoff_delay + jitter, max_delay)
                
                print(f"    -> Retrying in {total_delay:.2f} seconds...")
                await asyncio.sleep(total_delay)
    
    # If all retries fail, raise a comprehensive error
    raise Exception(f"API call failed after {retries} retries.") from last_exception