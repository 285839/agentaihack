#!/usr/bin/env python
import sys
import warnings

from datetime import datetime
from openai import OpenAI
import uvicorn

from neurosync.crew import Neurosync
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Enable CORS for all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """

    # try:
    #     uvicorn.run("neurosync.main:app",host="0.0.0.0", port=8000)
    # except Exception as e:
    #     raise Exception(f"An error occurred while running the crew : {e}")
    mol()


def mol():

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
        base_url= "http://10.31.207.8:8000/v1"
    )
    chat_response = client.chat.completions.create(
        model = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages =[
            {"roles" : "system", "content": "You are helpful"},
            {"roles" : "system", "content": "Create a document to give step for running docker in aws."}
        ]
    )

    print(chat_response.choices[0].message.content)
def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        Neurosync().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Neurosync().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        Neurosync().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")