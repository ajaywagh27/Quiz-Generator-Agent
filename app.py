import os
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- API Key Setup ---
# Securely get the key from the environment variable
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# --- Flask Setup ---
app = Flask(__name__, static_folder='static', template_folder='static')
CORS(app)

# --- LangChain and Gemini Setup ---
# Initialize the AI model using the environment variable and temperature
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.8
    )
except Exception as e:
    llm = None
    print(f"--- WARNING: Could not initialize the AI Model. Please check your API key. Error: {e} ---")

# --- Pydantic Schemas for Strict Output Formatting ---
class Question(BaseModel):
    question: str = Field(description="The text of the quiz question.")
    options: List[str] = Field(description="A list of 2 or 4 possible answers.")
    correctAnswerIndex: Union[int, List[int]] = Field(description="The index or indices of the correct answer(s) in the options list.")

class Quiz(BaseModel):
    quiz: List[Question] = Field(description="The list of quiz questions.")


def create_quiz_chain(topic, num_questions, question_type):
    """Dynamically creates the prompt and invokes the chain with a strict schema."""
    if not llm:
        raise Exception("AI model not initialized. Please ensure your GOOGLE_API_KEY is correctly set in your .env file.")

    parser = JsonOutputParser(pydantic_object=Quiz)

    is_mixed = question_type == 'mixed'
    is_multi_choice = question_type == 'multi_choice'

    if is_mixed:
        type_label = 'Mixed Type (Single Choice, Multiple Correct, True/False)'
        options_desc = 'some questions having two options (True/False) and others having four options.'
        answer_desc = 'For single-answer questions, correctAnswerIndex must be an Integer. For multiple-answer questions, it must be an Array of Integers. You must include questions of all three types.'
    elif is_multi_choice:
        type_label = 'Multiple Correct Answer'
        options_desc = 'exactly four options.'
        answer_desc = 'one or more correct answers. The correctAnswerIndex must ALWAYS be an Array of Integers.'
    else: # Single Choice
        type_label = 'Single Choice'
        options_desc = 'exactly four options.'
        answer_desc = 'only one correct answer. The correctAnswerIndex must ALWAYS be an Integer.'

    system_instruction = (
        f"You are a quiz master. Your task is to generate a {type_label} quiz based on the topic \"{topic}\". "
        f"The quiz must contain exactly {num_questions} questions. Each question must have {options_desc}. "
        f"The questions must have {answer_desc}. Your response must be a single valid JSON object with a key named 'quiz' "
        "which contains the array of question objects. Strictly adhere to the provided schema. "
        "Do not include any introductory or concluding text outside the JSON object."
    )

    prompt = PromptTemplate(
        template="{system_instruction}\n\nUser Query: {user_query}\n\n{format_instructions}",
        input_variables=["system_instruction", "user_query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    user_query = f"Generate a quiz with {num_questions} questions about: {topic}."

    quiz_chain = prompt.partial(system_instruction=system_instruction) | llm | parser

    try:
        response = quiz_chain.invoke({"user_query": user_query})
        quiz_data = response.get('quiz', []) 
        
        if not quiz_data or not isinstance(quiz_data, list) or not all(isinstance(q, dict) for q in quiz_data):
            raise ValueError("AI returned an empty or improperly formatted list of questions.")
        
        return quiz_data
    except OutputParserException as e:
        print(f"LangChain OutputParser Error: {e}")
        raise Exception("AI failed to generate quiz in the required format. Please try a different topic or question type.")
    except Exception as e:
        print(f"An unexpected error occurred during quiz generation: {e}")
        raise Exception("An error occurred while communicating with the AI. Please try again.")

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def serve_index():
    """Serves the main HTML file from the static folder (GET request)."""
    return send_from_directory('static', 'index.html')

@app.route('/generate-quiz', methods=['POST'])
def generate_quiz_route():
    """API endpoint to generate the quiz using LangChain and Gemini (POST request)."""
    try:
        data = request.json
        topic = data.get('topic')
        num_questions = data.get('numQuestions')
        question_type = data.get('questionType')

        if not all([topic, num_questions, question_type]):
            return jsonify({"error": "Missing topic, number of questions, or question type."}), 400

        quiz_data = create_quiz_chain(topic, num_questions, question_type)
        
        return jsonify({"quiz": quiz_data})

    except Exception as e:
        print(f"Server Error during quiz generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

