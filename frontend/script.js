document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const output = document.getElementById('output');
    const examInfo = document.getElementById('examInfo');
    const questionsDiv = document.getElementById('questions');
    
    if (!uploadForm || !loadingIndicator || !output || !examInfo || !questionsDiv) {
        console.error('Required DOM elements not found.');
        return;
    }
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Clear previous results
        examInfo.innerHTML = '';
        questionsDiv.innerHTML = '';
        
        // Show loading indicator
        loadingIndicator.style.display = 'block';
        output.style.display = 'none';
        
        // Get form data
        const documentFile = uploadForm.document.files[0];
        const numQuestions = uploadForm.numQuestions.value;
        
        if (!documentFile) {
            alert('Please upload a document.');
            loadingIndicator.style.display = 'none';
            return;
        }
        
        const formData = new FormData();
        formData.append('document', documentFile);
        formData.append('numQuestions', numQuestions);
        
        // Send request to backend
        fetch('http://localhost:5000/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            loadingIndicator.style.display = 'none';
            output.style.display = 'block';
            
            // Display exam info
            const examDetails = document.createElement('div');
            examDetails.className = 'exam-details';
            examDetails.innerHTML = `
                <h2>Generated Test</h2>
                <p><strong>Exam Type:</strong> ${data.exam_type || data.examType || 'Multiple Choice'}</p>
                <p><strong>Format:</strong> ${data.question_format || data.questionFormat || 'Question with options'}</p>
            `;
            examInfo.appendChild(examDetails);
            
            // Display questions by subject
            Object.entries(data.questions).forEach(([subject, questions]) => {
                const subjectSection = document.createElement('div');
                subjectSection.className = 'subject-section';
                
                const subjectTitle = document.createElement('h3');
                subjectTitle.className = 'subject-title';
                subjectTitle.textContent = subject;
                subjectSection.appendChild(subjectTitle);
                
                // Create a card for each question
                questions.forEach((question, index) => {
                    const questionCard = document.createElement('div');
                    questionCard.className = 'question-card';
                    
                    // Question header with number
                    const questionHeader = document.createElement('div');
                    questionHeader.className = 'question-header';
                    questionHeader.innerHTML = `<span class="question-number">Question ${index + 1}</span>`;
                    questionCard.appendChild(questionHeader);
                    
                    // Question content
                    const questionContent = document.createElement('div');
                    questionContent.className = 'question-content';
                    
                    // Question text
                    const questionText = document.createElement('div');
                    questionText.className = 'question-text';
                    questionText.textContent = question.question;
                    questionContent.appendChild(questionText);
                    
                    // Options
                    if (question.options && question.options.length > 0) {
                        const optionsDiv = document.createElement('div');
                        optionsDiv.className = 'options';
                        
                        question.options.forEach((option, optIndex) => {
                            const optionDiv = document.createElement('div');
                            optionDiv.className = 'option';
                            optionDiv.innerHTML = `<strong>${String.fromCharCode(65 + optIndex)}.</strong> ${option}`;
                            optionsDiv.appendChild(optionDiv);
                        });
                        
                        questionContent.appendChild(optionsDiv);
                    }

                    // Create button to show/hide answer
                    const toggleButton = document.createElement('button');
                    toggleButton.type = 'button';
                    toggleButton.className = 'toggle-answer-btn';
                    toggleButton.innerHTML = '<i class="fas fa-eye"></i> Show Answer';
                    questionContent.appendChild(toggleButton);
                    
                    // Create answer section (hidden by default)
                    const answerSection = document.createElement('div');
                    answerSection.className = 'answer-section';
                    answerSection.style.display = 'none';

                    // Answer
                    const correctAnswer = question.correct_answer || question.answer;
                    if (correctAnswer) {
                        const answerDiv = document.createElement('div');
                        answerDiv.className = 'answer';
                        answerDiv.innerHTML = `<strong>Answer:</strong> ${correctAnswer}`;
                        answerSection.appendChild(answerDiv);
                    }
                    
                    // Explanation
                    if (question.explanation) {
                        const explanationDiv = document.createElement('div');
                        explanationDiv.className = 'explanation';
                        explanationDiv.innerHTML = `<strong>Explanation:</strong> ${question.explanation}`;
                        answerSection.appendChild(explanationDiv);
                    }
                    
                    // Add answer section to question content
                    questionContent.appendChild(answerSection);
                    
                    // Add event listener to toggle button
                    toggleButton.addEventListener('click', function() {
                        const isHidden = answerSection.style.display === 'none';
                        answerSection.style.display = isHidden ? 'block' : 'none';
                        toggleButton.innerHTML = isHidden ? 
                            '<i class="fas fa-eye-slash"></i> Hide Answer' : 
                            '<i class="fas fa-eye"></i> Show Answer';
                    });
                    
                    questionCard.appendChild(questionContent);
                    subjectSection.appendChild(questionCard);
                });
                
                questionsDiv.appendChild(subjectSection);
            });
        })
        .catch(error => {
            loadingIndicator.style.display = 'none';
            output.style.display = 'block';
            
            console.error('Error:', error);
            
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error';
            errorDiv.textContent = `Error: ${error.message || 'Failed to generate questions. Please try again.'}`;
            questionsDiv.appendChild(errorDiv);
        });
    });
});