import os
import re
import json
import requests
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv

load_dotenv()

EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
EMAIL_SMTP_PORT = int(os.getenv("EMAIL_SMTP_PORT", 587))
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def send_email(to_email, subject, body):
    """Send an email using SMTP"""
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        return "Email sending failed: EMAIL_FROM or EMAIL_PASSWORD not set in environment."

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = to_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
        return "Email successfully sent."
    except Exception as e:
        return f"Failed to send email: {e}"


def call_groq_for_email(prompt, temperature=0.7):
    """Call Groq API for email composition and extraction"""
    if not GROQ_API_KEY:
        return None

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "system",
                "content": "You are a professional email assistant. You help compose clear, concise, and professional emails."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": 800,
    }

    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return None


class EmailAgent:
    """
    Smart AI-powered email agent that can:
    1. Accept all email details at once or gather them step by step
    2. Compose email bodies based on user instructions
    3. Extract information intelligently from natural language
    """

    def __init__(self):
        self.state = "COLLECTING"  # COLLECTING, CONFIRMING, COMPLETED
        self.recipient = None
        self.subject = None
        self.body = None
        self.user_instructions = []  # Track what user has said
        self.missing_info = []

    def _extract_email(self, text):
        """Extract email address from text"""
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        return emails[0] if emails else None

    def _is_confirmation(self, text):
        """Check if text contains confirmation keywords"""
        text_lower = text.lower().strip()
        confirmations = ["yes", "send", "confirm", "go ahead", "send it", "looks good", "perfect", "correct"]
        return any(conf in text_lower for conf in confirmations)

    def _is_cancellation(self, text):
        """Check if text contains cancellation keywords"""
        text_lower = text.lower().strip()
        cancellations = ["no", "cancel", "stop", "don't send", "nevermind", "never mind"]
        return any(canc in text_lower for canc in cancellations)

    def _extract_info_with_ai(self, user_input):
        """Use AI to extract recipient, subject, and body instructions from user input"""
        prompt = f"""Analyze this email request and extract information in JSON format.

User request: "{user_input}"

Extract:
1. recipient: The email address if mentioned (null if not mentioned)
2. subject: The subject line if mentioned (null if not mentioned)
3. body_instruction: What the user wants in the email body (null if not clear)
4. needs_composition: true if user wants you to write the email, false if they provided exact text

Return ONLY valid JSON with these fields. Examples:

Input: "Send an email to john@company.com about the meeting tomorrow"
Output: {{"recipient": "john@company.com", "subject": "Meeting Tomorrow", "body_instruction": "remind about the meeting tomorrow", "needs_composition": true}}

Input: "Email sarah@test.com with subject 'Project Update' and tell her the project is on track and we'll deliver by Friday"
Output: {{"recipient": "sarah@test.com", "subject": "Project Update", "body_instruction": "inform that project is on track and will deliver by Friday", "needs_composition": true}}

Input: "Send to mike@example.com: Hi Mike, Just checking in. Let me know if you need anything. Thanks!"
Output: {{"recipient": "mike@example.com", "subject": null, "body_instruction": "Hi Mike, Just checking in. Let me know if you need anything. Thanks!", "needs_composition": false}}

Now analyze: "{user_input}"
Return ONLY the JSON object, no other text."""

        result = call_groq_for_email(prompt, temperature=0.3)
        if result:
            try:
                # Clean up response to extract JSON
                result = result.strip()
                if result.startswith("```json"):
                    result = result[7:]
                if result.startswith("```"):
                    result = result[3:]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()

                return json.loads(result)
            except json.JSONDecodeError:
                return None
        return None

    def _compose_email_body(self, instruction, recipient=None, subject=None):
        """Use AI to compose professional email body based on instructions"""
        context = ""
        if recipient:
            context += f"Recipient: {recipient}\n"
        if subject:
            context += f"Subject: {subject}\n"

        prompt = f"""Write a professional email based on these instructions:

{context}
Instructions: {instruction}

Write ONLY the email body. Make it:
- Professional and clear
- Concise but complete
- Properly formatted with appropriate greeting and closing
- Natural and friendly in tone

Do not include subject line or "Subject:" prefix. Just write the email body."""

        result = call_groq_for_email(prompt, temperature=0.7)
        return result if result else instruction

    def _identify_missing_info(self):
        """Determine what information is still needed"""
        missing = []
        if not self.recipient:
            missing.append("recipient email address")
        if not self.subject:
            missing.append("subject line")
        if not self.body:
            missing.append("email content")
        return missing

    def process_message(self, user_input):
        """Process user input with intelligent extraction and composition"""

        # COLLECTING state - gathering email details
        if self.state == "COLLECTING":
            # Add to user instructions
            self.user_instructions.append(user_input)

            # Try to extract information with AI
            extracted = self._extract_info_with_ai(user_input)

            if extracted:
                # Update recipient if found
                if extracted.get("recipient") and not self.recipient:
                    self.recipient = extracted["recipient"]

                # Update subject if found
                if extracted.get("subject") and not self.subject:
                    self.subject = extracted["subject"]

                # Handle body
                body_instruction = extracted.get("body_instruction")
                needs_composition = extracted.get("needs_composition", True)

                if body_instruction and not self.body:
                    if needs_composition:
                        # User wants us to compose the email
                        composed = self._compose_email_body(
                            body_instruction,
                            self.recipient,
                            self.subject
                        )
                        self.body = composed
                    else:
                        # User provided exact text
                        self.body = body_instruction
            else:
                # Fallback: Try simple email extraction
                if not self.recipient:
                    email = self._extract_email(user_input)
                    if email:
                        self.recipient = email

            # Check what's still missing
            self.missing_info = self._identify_missing_info()

            if not self.missing_info:
                # We have everything! Move to confirmation
                self.state = "CONFIRMING"
                return self._get_confirmation_message()
            else:
                # Ask for missing information
                return self._ask_for_missing_info()

        # CONFIRMING state - waiting for user confirmation
        elif self.state == "CONFIRMING":
            # Check for confirmation first (highest priority)
            if self._is_confirmation(user_input):
                result = send_email(self.recipient, self.subject, self.body)
                self.state = "COMPLETED"
                return f"{result}\n\nThe email has been sent to {self.recipient}."

            # Check for cancellation
            elif self._is_cancellation(user_input):
                self.state = "COMPLETED"
                return "Email cancelled. The email was not sent."

            # Any other input is treated as a modification request
            else:
                # User wants to make changes - handle with AI
                return self._handle_change_request(user_input)

        # COMPLETED state
        elif self.state == "COMPLETED":
            return "This email session is complete. Click the 'Send Email' button to compose a new email."

        return "Something went wrong. Please try again."

    def _ask_for_missing_info(self):
        """Generate a natural request for missing information"""
        if len(self.missing_info) == 3:
            # Missing everything - first message
            return "I'll help you send an email. You can provide all the details at once, or I'll ask for what's needed.\n\nPlease tell me:\n- Who to send it to\n- What the subject should be\n- What you want the email to say"
        elif len(self.missing_info) == 2:
            missing_str = " and ".join(self.missing_info)
            return f"Thanks! I still need the {missing_str}. Please provide those details."
        elif len(self.missing_info) == 1:
            return f"Great! I just need the {self.missing_info[0]}. What should it be?"
        else:
            return "Let me confirm the details..."

    def _get_confirmation_message(self):
        """Generate the confirmation message showing email details"""
        return f"""Here's the email I've prepared:

**To:** {self.recipient}
**Subject:** {self.subject}
**Message:**
{self.body}

---
Reply with:
- **'yes'** or **'send'** to send this email
- **'no'** or **'cancel'** to cancel
- Or tell me what you'd like to change"""

    def _handle_change_request(self, user_input):
        """Handle requests to change email details using AI"""
        text_lower = user_input.lower()

        # Check if they want to change a specific field completely
        if "change recipient" in text_lower or "change email address" in text_lower or "different recipient" in text_lower:
            self.recipient = None
            self.missing_info = ["recipient email address"]
            return "What email address should I send to?"

        elif "change subject" in text_lower or "different subject" in text_lower:
            self.subject = None
            self.missing_info = ["subject line"]
            return "What should the new subject line be?"

        # For everything else, use AI to modify the email intelligently
        else:
            # Use AI to understand what changes they want
            prompt = f"""The user wants to modify this email. Apply their requested changes.

Current email:
To: {self.recipient}
Subject: {self.subject}
Body:
{self.body}

User's change request: "{user_input}"

Based on their request, generate the UPDATED email in this JSON format:
{{
    "recipient": "email@example.com",
    "subject": "Updated Subject",
    "body": "Updated email body text"
}}

Important:
- Only change what the user asked to change
- Keep everything else the same
- If they mention tone/style changes, adjust accordingly
- If they want to add/remove content, do so
- Return ONLY the JSON, no other text"""

            result = call_groq_for_email(prompt, temperature=0.7)

            if result:
                try:
                    # Clean and parse JSON
                    result = result.strip()
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.startswith("```"):
                        result = result[3:]
                    if result.endswith("```"):
                        result = result[:-3]
                    result = result.strip()

                    updated = json.loads(result)

                    # Update fields
                    if updated.get("recipient"):
                        self.recipient = updated["recipient"]
                    if updated.get("subject"):
                        self.subject = updated["subject"]
                    if updated.get("body"):
                        self.body = updated["body"]

                    return self._get_confirmation_message()

                except (json.JSONDecodeError, KeyError) as e:
                    # Fallback: ask them to be more specific
                    return """I'm not sure exactly what you'd like to change. You can:
- Say "change recipient" to update the email address
- Say "change subject" to update the subject line
- Be specific like "make it more formal" or "add a deadline" or "remove the first paragraph"
- Or just tell me what you'd like different"""
            else:
                return "I had trouble processing that change. Could you rephrase what you'd like different?"


# Legacy function for backward compatibility
def email_agent_interaction(question, chat_history=None):
    """
    Legacy function - checks if message is email-related.
    Returns None if not an email request (to be handled by main KB system).
    """
    if "send email" not in question.lower() and "compose email" not in question.lower():
        return None

    return "To send an email, please click the '📧 Send Email' button in the sidebar."
