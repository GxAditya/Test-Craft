import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from dotenv import load_dotenv
import requests
from flask_cors import CORS
import re
import json
from typing import List, Dict, Any

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
USE_OFFLINE_MODE = GEMINI_API_KEY is None or GEMINI_API_KEY.strip() == ""

# Print warning if offline mode is used
if USE_OFFLINE_MODE:
    print("WARNING: No valid GEMINI_API_KEY found. Running in offline mode with sample questions.")

EXAM_STRUCTURES = {
    'JEE': {
        'subjects': ['Physics', 'Chemistry', 'Mathematics'],
        'distribution': [0.33, 0.33, 0.34]
    },
    'NEET': {
        'subjects': ['Physics', 'Chemistry', 'Biology'],
        'distribution': [0.30, 0.30, 0.40]
    }
}

def detect_exam_type(text_content: str) -> str:
    text_content_lower = text_content.lower()
    jee_keywords = ['jee', 'mathematics', 'math', 'algebra', 'calculus', 'trigonometry']
    neet_keywords = ['neet', 'biology', 'zoology', 'botany', 'anatomy', 'physiology']
    
    jee_count = sum(1 for keyword in jee_keywords if keyword in text_content_lower)
    neet_count = sum(1 for keyword in neet_keywords if keyword in text_content_lower)
    
    if neet_count > jee_count:
        return 'NEET'
    return 'JEE'  # Default to JEE if unclear

def detect_question_format(text_content: str) -> str:
    # Check for MCQ patterns
    if re.search(r'\([a-d]\)', text_content.lower()) or re.search(r'\b[a-d]\)', text_content.lower()):
        return 'MCQ'
    # Check for numerical answer patterns
    elif re.search(r'numerical\s+answer|answer\s*:', text_content.lower()):
        return 'NUMERICAL'
    # Check for short answer patterns - questions with ? and relatively short
    elif '?' in text_content and len(text_content.split()) < 30:
        return 'SHORT_ANSWER'
    return 'LONG_ANSWER'

def retrieve_relevant_sections(text_content: str, exam_type: str) -> Dict[str, List[str]]:
    sections = {}
    exam_subjects = EXAM_STRUCTURES[exam_type]['subjects']
    
    # Split into paragraphs and potential question blocks
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text_content)
    
    # Initialize sections with empty lists
    for subject in exam_subjects:
        sections[subject] = []
    
    # Enhanced subject detection with expanded keywords and context
    for para in paragraphs:
        if not para.strip():
            continue
            
        classified = False
        para_lower = para.lower()
        
        for subject in exam_subjects:
            keywords = get_subject_keywords(subject)
            # Check both for exact keywords and contextual clues
            if any(keyword in para_lower for keyword in keywords):
                sections[subject].append(para)
                classified = True
                break
        
        # If couldn't classify, use a backup approach based on subject-specific patterns
        if not classified:
            # Try to detect based on equations, terms, or patterns
            if any(term in para_lower for term in ['force', 'motion', 'energy', 'newton', 'voltage', 'current']):
                sections['Physics'].append(para)
            elif any(term in para_lower for term in ['reaction', 'compound', 'element', 'acid', 'molecule', 'bond']):
                sections['Chemistry'].append(para)
            elif exam_type == 'JEE' and any(term in para_lower for term in ['equation', 'function', 'graph', 'calculate', 'solve']):
                sections['Mathematics'].append(para)
            elif exam_type == 'NEET' and any(term in para_lower for term in ['cell', 'organism', 'tissue', 'species', 'dna']):
                sections['Biology'].append(para)
    
    # Handle empty sections by distributing general content
    general_content = []
    for para in paragraphs:
        if para.strip() and not any(para in sections[subject] for subject in exam_subjects):
            general_content.append(para)
    
    # Distribute general content to empty sections
    for subject in exam_subjects:
        if not sections[subject] and general_content:
            # Assign some general content proportionally
            section_share = max(1, len(general_content) // len(exam_subjects))
            sections[subject] = general_content[:section_share]
            general_content = general_content[section_share:]
    
    return sections

def get_subject_keywords(subject: str) -> List[str]:
    keywords = {
        'Physics': ['force', 'energy', 'motion', 'waves', 'electricity', 'quantum', 'mechanics', 'optics', 
                   'thermodynamics', 'kinematics', 'newton', 'current', 'voltage', 'resistance', 'gravity',
                   'momentum', 'joule', 'watt', 'ohm', 'magneti', 'nuclear', 'semiconductor'],
        'Chemistry': ['reaction', 'compound', 'molecule', 'acid', 'base', 'organic', 'inorganic', 'element',
                     'periodic', 'bond', 'atom', 'electron', 'proton', 'neutron', 'equilibrium', 'titration',
                     'solution', 'mole', 'oxidation', 'reduction', 'polymer', 'alkane', 'alkene', 'alkyne'],
        'Mathematics': ['equation', 'function', 'integral', 'derivative', 'matrix', 'vector', 'algebra',
                       'geometry', 'calculus', 'trigonometry', 'probability', 'statistics', 'arithmetic',
                       'sequence', 'series', 'logarithm', 'exponential', 'coordinate', 'polynomial', 'quadratic'],
        'Biology': ['cell', 'organism', 'gene', 'evolution', 'tissue', 'system', 'dna', 'rna', 'chromosome',
                   'protein', 'enzyme', 'respiration', 'photosynthesis', 'ecology', 'anatomy', 'physiology',
                   'botany', 'zoology', 'reproduction', 'mitosis', 'meiosis', 'heredity', 'taxonomy']
    }
    return keywords.get(subject, [])

def generate_questions_with_gemini(sections: Dict[str, List[str]], num_questions: int, exam_type: str, question_format: str) -> Dict[str, List[Dict]]:
    distribution = EXAM_STRUCTURES[exam_type]['distribution']
    questions_per_subject = [int(num_questions * d) for d in distribution]
    
    # Adjust for rounding errors to ensure we get exactly num_questions
    total_assigned = sum(questions_per_subject)
    if total_assigned < num_questions:
        # Add the remaining questions to the last subject
        questions_per_subject[-1] += (num_questions - total_assigned)
    
    generated_questions = {}
    
    for subject, count in zip(EXAM_STRUCTURES[exam_type]['subjects'], questions_per_subject):
        if count == 0:
            generated_questions[subject] = []
            continue
            
        # If we're in offline mode or using a demo, skip API call and go straight to fallback
        if USE_OFFLINE_MODE:
            generated_questions[subject] = generate_fallback_questions(subject, count, question_format, demo=True)
            continue
            
        # Limit content size to avoid API limits
        subject_content = ' '.join(sections[subject])
        if len(subject_content) > 10000:
            subject_content = subject_content[:10000]
        
        format_instructions = get_format_instructions(question_format)
        
        prompt = f"""
        Generate {count} {question_format} questions for {subject} based on the following content:
        {subject_content}
        
        {format_instructions}
        
        Important:
        1. Ensure each question is relevant to {subject} and the content provided
        2. Generate questions at the difficulty level appropriate for {exam_type} exam
        3. Include detailed explanations for each answer
        4. Make questions similar to the source material but NOT identical - change wording, structure, or context
        5. For multiple-choice questions, change the options or values while maintaining the same concepts
        6. Return the response in valid JSON format
        """
        
        response = call_gemini_api(prompt)
        try:
            questions = parse_gemini_response(response)
            if questions:
                generated_questions[subject] = questions
            else:
                # Fallback questions if parsing fails
                generated_questions[subject] = generate_fallback_questions(subject, count, question_format)
        except Exception as e:
            print(f"Error parsing Gemini response for {subject}: {str(e)}")
            generated_questions[subject] = generate_fallback_questions(subject, count, question_format)
    
    return generated_questions

def get_format_instructions(question_format: str) -> str:
    if question_format == 'MCQ':
        return """
        Format each question as JSON with the following structure:
        {
            "question": "question text",
            "options": ["option A", "option B", "option C", "option D"],
            "correct_answer": "correct option (A, B, C, or D)",
            "explanation": "detailed explanation of the answer"
        }
        """
    else:
        return """
        Format each question as JSON with the following structure:
        {
            "question": "question text",
            "correct_answer": "correct answer",
            "explanation": "detailed explanation of the answer"
        }
        """

def parse_gemini_response(response: Dict[str, Any]) -> List[Dict]:
    # Check if we're in offline mode or had API failure
    if response.get("offline", False):
        return []
        
    try:
        # Check if response has the expected structure
        if 'candidates' not in response or not response['candidates']:
            return []
            
        # Extract the text content from the response
        content = response['candidates'][0]['content']
        if 'parts' not in content or not content['parts']:
            return []
            
        text = content['parts'][0]['text']
        
        # Extract JSON objects from the text
        # First try to find JSON arrays
        json_matches = re.findall(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if json_matches:
            for match in json_matches:
                try:
                    return json.loads(match)
                except:
                    pass
        
        # If no arrays found, try to find individual JSON objects
        json_matches = re.findall(r'\{.*?\}', text, re.DOTALL)
        if json_matches:
            questions = []
            for match in json_matches:
                try:
                    question = json.loads(match)
                    if 'question' in question and 'correct_answer' in question:
                        questions.append(question)
                except:
                    continue
            return questions
            
        # Try a fallback approach: check for markdown code blocks with JSON
        markdown_json = re.findall(r'```(?:json)?\s*([\s\S]*?)```', text)
        if markdown_json:
            for block in markdown_json:
                try:
                    # Try as array first
                    try:
                        return json.loads(block)
                    except:
                        # Try cleaning the string first
                        cleaned = block.strip().replace('\n', '').replace('\\', '\\\\')
                        return json.loads(cleaned)
                except:
                    pass
        
        return []
    except Exception as e:
        print(f"Error parsing Gemini response: {str(e)}")
        return []

def generate_fallback_questions(subject: str, count: int, format: str, demo: bool = False) -> List[Dict]:
    """Generate simple fallback questions if the API call fails"""
    questions = []
    
    # If in demo mode, provide more realistic sample questions
    if demo:
        sample_questions = get_sample_questions_for_subject(subject, format)
        num_samples = len(sample_questions)
        
        # Use sample questions and repeat if needed
        for i in range(count):
            if num_samples > 0:
                questions.append(sample_questions[i % num_samples])
            else:
                # Fallback to generic if no samples for this subject
                questions.append(create_generic_question(subject, i, format))
    else:
        # Simple generic fallback
        for i in range(count):
            questions.append(create_generic_question(subject, i, format))
    
    return questions

def create_generic_question(subject: str, index: int, format: str) -> Dict:
    """Create a generic question when no other options are available"""
    if format == 'MCQ':
        return {
            "question": f"Sample {subject} question {index+1}. This is a fallback question due to generation error.",
            "options": ["Option A", "Option B", "Option C", "Option D"],
            "correct_answer": "A",
            "explanation": "This is a fallback explanation. The actual question generation failed."
        }
    else:
        return {
            "question": f"Sample {subject} question {index+1}. This is a fallback question due to generation error.",
            "correct_answer": "Sample answer",
            "explanation": "This is a fallback explanation. The actual question generation failed."
        }

def get_sample_questions_for_subject(subject: str, format: str) -> List[Dict]:
    """Return pre-defined sample questions for each subject"""
    if subject == 'Physics':
        if format == 'MCQ':
            return [
                {
                    "question": "A ball is thrown vertically upward with a velocity of 20 m/s. How high will it go?",
                    "options": ["20 m", "40 m", "20.4 m", "10 m"],
                    "correct_answer": "B",
                    "explanation": "Using the equation v² = u² + 2as, where final velocity v = 0, initial velocity u = 20 m/s, and acceleration a = -9.8 m/s², we get: 0 = 20² + 2(-9.8)h, which gives h = 20²/19.6 ≈ 20.4 m, rounded to 20 m."
                },
                {
                    "question": "Which of the following is a vector quantity?",
                    "options": ["Mass", "Temperature", "Time", "Velocity"],
                    "correct_answer": "D",
                    "explanation": "Velocity is a vector quantity as it has both magnitude and direction. Mass, temperature, and time are scalar quantities with only magnitude."
                }
            ]
        else:
            return [
                {
                    "question": "State Newton's Third Law of Motion.",
                    "correct_answer": "For every action, there is an equal and opposite reaction.",
                    "explanation": "Newton's Third Law states that when one body exerts a force on a second body, the second body exerts a force equal in magnitude and opposite in direction on the first body."
                }
            ]
            
    elif subject == 'Chemistry':
        if format == 'MCQ':
            return [
                {
                    "question": "What is the pH of a neutral solution at 25°C?",
                    "options": ["0", "7", "14", "1"],
                    "correct_answer": "B",
                    "explanation": "A neutral solution has a pH of 7 at 25°C. This is because the ion product of water (Kw) is 10^-14, and pH = -log[H+], where [H+] = 10^-7 in a neutral solution."
                },
                {
                    "question": "Which of the following elements has the highest electronegativity?",
                    "options": ["Sodium", "Fluorine", "Carbon", "Oxygen"],
                    "correct_answer": "B",
                    "explanation": "Fluorine has the highest electronegativity value (3.98 on the Pauling scale) of all elements. This is due to its small atomic radius and high effective nuclear charge."
                }
            ]
        else:
            return [
                {
                    "question": "Define oxidation and reduction in terms of electron transfer.",
                    "correct_answer": "Oxidation is the loss of electrons, and reduction is the gain of electrons.",
                    "explanation": "In redox reactions, the substance that loses electrons is oxidized (oxidation), while the substance that gains electrons is reduced (reduction)."
                }
            ]
            
    elif subject == 'Mathematics':
        if format == 'MCQ':
            return [
                {
                    "question": "Find the derivative of f(x) = x³ - 3x² + 2x - 1.",
                    "options": ["3x² - 6x + 2", "3x² - 6x", "x² - 6x + 2", "3x³ - 6x + 2"],
                    "correct_answer": "A",
                    "explanation": "Using the power rule and linearity of differentiation: f'(x) = 3x² - 6x + 2."
                },
                {
                    "question": "Solve for x: 2x - 5 = 3x + 2",
                    "options": ["x = -7", "x = 7", "x = -3", "x = 3"],
                    "correct_answer": "A",
                    "explanation": "2x - 5 = 3x + 2\n2x - 3x = 2 + 5\n-x = 7\nx = -7"
                }
            ]
        else:
            return [
                {
                    "question": "Evaluate the indefinite integral ∫(2x + 3) dx.",
                    "correct_answer": "x² + 3x + C, where C is the constant of integration.",
                    "explanation": "∫(2x + 3) dx = ∫2x dx + ∫3 dx = 2(x²/2) + 3x + C = x² + 3x + C"
                }
            ]
            
    elif subject == 'Biology':
        if format == 'MCQ':
            return [
                {
                    "question": "Which organelle is known as the 'powerhouse of the cell'?",
                    "options": ["Nucleus", "Mitochondria", "Golgi apparatus", "Endoplasmic reticulum"],
                    "correct_answer": "B",
                    "explanation": "Mitochondria are called the 'powerhouse of the cell' because they generate most of the cell's supply of ATP (adenosine triphosphate), which is used as a source of chemical energy."
                },
                {
                    "question": "What is the process by which plants make their own food?",
                    "options": ["Respiration", "Photosynthesis", "Transpiration", "Digestion"],
                    "correct_answer": "B",
                    "explanation": "Photosynthesis is the process by which green plants, algae, and some bacteria convert light energy from the sun into chemical energy stored in glucose, using carbon dioxide and water."
                }
            ]
        else:
            return [
                {
                    "question": "Describe the structure and function of DNA.",
                    "correct_answer": "DNA is a double-helix structure made of nucleotides. It contains genetic information and instructions for protein synthesis.",
                    "explanation": "DNA (deoxyribonucleic acid) consists of two complementary strands coiled around each other in a double helix. Each strand is made up of nucleotides containing a sugar (deoxyribose), a phosphate group, and a nitrogenous base (adenine, thymine, guanine, or cytosine). The sequence of these bases encodes genetic information that determines protein synthesis and hereditary traits."
                }
            ]
            
    # Default empty list if subject not recognized
    return []

def call_gemini_api(prompt: str) -> Dict[str, Any]:
    # If offline mode or no API key, don't even try the API
    if USE_OFFLINE_MODE:
        return {"offline": True}
        
    # Try gemini-1.5-flash first
    try:
        return _call_gemini_model("gemini-1.5-flash", prompt)
    except Exception as e:
        print(f"Error with gemini-1.5-flash: {str(e)}")
        # Fall back to gemini-2.0-flash if the first model fails
        try:
            return _call_gemini_model("gemini-2.0-flash", prompt)
        except Exception as e2:
            print(f"Error with fallback model gemini-2.0-flash: {str(e2)}")
            # Fall back to gemini-pro as a last resort
            try:
                return _call_gemini_model("gemini-pro", prompt, use_beta=True)
            except Exception as e3:
                print(f"Error with fallback model gemini-pro: {str(e3)}")
                # Return empty response that won't break parsing
                return {"offline": True}

def _call_gemini_model(model_name: str, prompt: str, use_beta: bool = False) -> Dict[str, Any]:
    """Helper function to call different Gemini models with appropriate API versions"""
    api_version = "v1beta" if use_beta else "v1"
    url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        'contents': [{'parts': [{'text': prompt}]}],
        'generationConfig': {
            'temperature': 0.4,
            'topP': 0.8,
            'topK': 40,
            'maxOutputTokens': 8192
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()  # This will raise an exception for 4XX/5XX responses
    return response.json()

@app.route('/upload', methods=['POST'])
def upload_document():
    try:
        if 'document' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No document provided'
            }), 400
            
        document = request.files['document']
        if document.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No selected file'
            }), 400
            
        if 'numQuestions' not in request.form:
            return jsonify({
                'status': 'error',
                'message': 'Number of questions not specified'
            }), 400
            
        try:
            num_questions = int(request.form['numQuestions'])
            if num_questions <= 0:
                raise ValueError("Number of questions must be positive")
        except ValueError:
            return jsonify({
                'status': 'error',
                'message': 'Invalid number of questions'
            }), 400
        
        filename = secure_filename(document.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        document.save(filepath)
        
        try:
            text_content = parse_document(filepath)
            if not text_content or text_content == "Unsupported file format":
                return jsonify({
                    'status': 'error',
                    'message': 'Unable to extract text from document or unsupported file format'
                }), 400
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Error parsing document: {str(e)}'
            }), 500
            
        exam_type = detect_exam_type(text_content)
        question_format = detect_question_format(text_content)
        
        relevant_sections = retrieve_relevant_sections(text_content, exam_type)
        
        # Check if we have content for at least one subject
        if not any(sections for sections in relevant_sections.values()):
            return jsonify({
                'status': 'error',
                'message': 'Could not extract relevant content for any subject'
            }), 400
            
        generated_questions = generate_questions_with_gemini(
            relevant_sections, 
            num_questions, 
            exam_type, 
            question_format
        )
        
        os.remove(filepath)  # Clean up uploaded file
        
        return jsonify({
            'status': 'success',
            'exam_type': exam_type,
            'question_format': question_format,
            'questions': generated_questions
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

def parse_document(filepath):
    if filepath.endswith('.pdf'):
        return parse_pdf(filepath)
    elif filepath.endswith('.docx'):
        return parse_docx(filepath)
    elif filepath.endswith('.txt'):
        return parse_txt(filepath)
    else:
        return "Unsupported file format"

def parse_pdf(filepath):
    text = ""
    with open(filepath, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def parse_docx(filepath):
    doc = docx.Document(filepath)
    return "\n".join([para.text for para in doc.paragraphs])

def parse_txt(filepath):
    with open(filepath, 'r') as file:
        return file.read()

if __name__ == '__main__':
    app.run(debug=True)