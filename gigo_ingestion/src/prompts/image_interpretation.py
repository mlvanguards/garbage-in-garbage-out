IMAGE_INTERPRETATION_PROMPT = """
You are reading a technical manual page. Please answer this specific question based on what you see in the image.

QUESTION: {question}

INSTRUCTIONS:
1. Read all text visible in this manual page carefully
2. Look for information that directly answers the question
3. If you find the answer:
   - Quote the exact text from the manual
   - Provide a clear, specific answer
   - Include any relevant details like frequencies, procedures, or warnings
4. If you find related information but not the exact answer:
   - Explain what related information you found
   - Indicate what might be missing
5. If this page doesn't contain the answer:
   - Say "This page does not contain the answer to this question"
   - Briefly mention what information IS on this page

Focus on maintenance schedules, operating procedures, safety instructions, and technical specifications.

Be precise and factual. Quote the manual text when possible.
"""
