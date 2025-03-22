Design a Test Generator App that allows users to input a document of any format (e.g., previous year questions, sample papers, etc.). The app should process the document through the following workflow:

    Document Input: The user uploads a document (PDF, Word, text file, etc.) containing previous year questions or sample papers.

    Document Processing:

        The document is first processed using a Retrieval-Augmented Generation (RAG) system to extract relevant information.

        After processing, the extracted content is fed into a Generative AI (GenAI) model.

    Test Generation:

        The GenAI model generates a test based on the content and in the exact same format as the input document (e.g., question format, layout, etc.).

        The test should include questions from multiple subjects based on the type of exam. For example:

            For JEE MAINS, the test should contain questions from Physics, Chemistry, and Mathematics.

            For NEET, the test should contain questions from Physics, Chemistry, and Biology.

        The number of questions to be generated is set by the user as an input, with each subject getting a proportionate number of questions based on the typical structure of the exam.

    Output: The generated test should match the format of the original document, ensuring consistency (e.g., multiple-choice questions, short answer, or essay format as per the input document).

Instructions:

    Do not hallucinate or invent additional questions; only generate questions that align with the content and format of the input document.

    The subjects covered should match those typically found in the exam (e.g., Physics, Chemistry, Mathematics for JEE MAINS or Physics, Chemistry, Biology for NEET).

    If any clarification is needed about the format or content of the input document, ask for further details before proceeding.