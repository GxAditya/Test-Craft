# TestCraft

A modern AI-powered test generation application that creates custom exam questions from your study materials.

![TestCraft Screenshot](https://via.placeholder.com/800x400?text=TestCraft+Screenshot)

## Features

- **Document Upload**: Support for PDF, DOCX, and TXT formats
- **Smart Question Generation**: Uses Google's Gemini AI models to create contextually relevant questions
- **Exam Format Detection**: Automatically detects and formats questions based on the content
- **Subject Classification**: Categorizes questions by subject (Physics, Chemistry, Mathematics, Biology)
- **Exam Type Support**: Optimized for JEE and NEET exam formats
- **Interactive UI**: Modern card-based interface with show/hide answer functionality
- **Fallback Mechanism**: Includes sample questions when AI generation is unavailable

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **AI Model**: Google Gemini API (1.5-flash, 2.0-flash, pro)
- **Document Processing**: PyPDF2, python-docx

## Setup and Installation

### Prerequisites

- Python 3.8+
- Google Gemini API key

### Backend Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/testcraft.git
   cd testcraft
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   cd backend
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the backend directory with your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

5. Start the Flask server:
   ```
   python -m flask run
   ```
   The backend will run on http://localhost:5000

### Frontend Setup

1. Open a new terminal window/tab
2. Navigate to the frontend directory:
   ```
   cd frontend
   ```

3. Start a simple HTTP server:
   ```
   python -m http.server
   ```
   The frontend will be available at http://localhost:8000

## How to Use

1. Open the application in your browser (http://localhost:8000)
2. Upload your study document (PDF, DOCX, or TXT)
3. Specify the number of questions you want to generate
4. Click "Generate Test"
5. Review the generated questions, which are organized by subject
6. Use the "Show Answer" button to reveal answers and explanations

## Architecture

The application follows a client-server architecture:

- **Frontend**: Handles user interaction, document upload, and displays generated tests
- **Backend**: Processes documents, detects exam types, extracts content, and generates questions using AI

### API Workflow

1. Document upload → Text extraction
2. Exam type detection → Question format detection
3. Subject classification → Content extraction
4. AI question generation → Response formatting

## Offline Mode

If no Gemini API key is provided, the application runs in offline mode, providing sample questions for demonstration purposes.

## License

[MIT License](LICENSE)

## Acknowledgements

- Google Gemini API for AI capabilities
- Flask for the backend framework
- FontAwesome for icons
