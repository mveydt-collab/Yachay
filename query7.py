import os
import io
import json
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import streamlit.components.v1 as components
from PIL import Image
import easyocr
import nltk
from datetime import datetime
import uuid
from email_agent import email_agent_interaction, EmailAgent

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY missing in .env")
    st.stop()

BASE_DIR = r"C:\Users\MattVeydt\OneDrive - AMEND Consulting\Desktop\kb_assistant"
INDEX_PATH = os.path.join(BASE_DIR, "embeddings.json")
CHAT_HISTORY_DIR = os.path.join(BASE_DIR, "chat_histories")

# Create chat history directory if it doesn't exist
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)


# Load and cache index for performance
@st.cache_data
def load_index():
    if not os.path.exists(INDEX_PATH):
        raise RuntimeError(f"Index file not found: {INDEX_PATH}. Run ingest.py first.")
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        item["embedding"] = np.array(item["embedding"], dtype=np.float32)
    return data


index = load_index()


# Load embedding model and cache
@st.cache_resource
def load_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return SentenceTransformer(model_name)


# Load EasyOCR reader and cache
@st.cache_resource
def load_ocr_reader():
    """Load EasyOCR reader for English text"""
    return easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have CUDA


embedding_model = load_model()
ocr_reader = load_ocr_reader()

# Download punkt for tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def embed_query(text: str) -> np.ndarray:
    vec = embedding_model.encode([text], show_progress_bar=False)[0]
    return np.array(vec, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return -1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

SIMILARITY_THRESHOLD = 0.3


def ask_groq(question: str, context_chunks, chat_history=None, temperature=0.3, max_tokens=500):
    """Ask Groq API with configurable temperature and max_tokens"""
    context_text = ""
    for i, c in enumerate(context_chunks, start=1):
        context_text += f"[Source {i}: {c['source']}]\n{c['text']}\n\n"

    if context_text:
        prompt = f"""
You are a knowledgeable assistant that helps users understand security, IT, and business concepts.

You can:
1. Answer questions based on the provided knowledge base context
2. Provide examples, explanations, and educational content about general topics
3. Help users learn about security risks, best practices, and concepts

If specific information is in the context, use it and cite your sources.
If the question is general (like "give an example of X"), provide a helpful, educational answer.
If you truly cannot answer, ask a clarifying question.

Context from Knowledge Base:
{context_text}

Question: {question}

If you use information from the context, provide citations at the end in the form:
Sources: [filename1, filename2, ...]
"""
    else:
        prompt = f"""
You are a knowledgeable assistant that helps users understand security, IT, and business concepts.

The user asked: {question}

Provide a helpful, educational answer. If it's a request for examples or general knowledge (like security risks, best practices, etc.), provide clear, informative examples and explanations.
"""

    messages = [{"role": "system", "content": "You are a helpful knowledge base assistant."}]

    # Add recent chat history for context (last 3 exchanges)
    if chat_history and len(chat_history) > 0:
        for msg in chat_history[-6:]:  # Last 3 Q&A pairs
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(GROQ_CHAT_URL, headers=GROQ_HEADERS, data=json.dumps(payload))
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def speak_text(text, voice_type="normal", rate=0.9, stop=False):
    """TTS with actual browser voices for more natural speech"""
    if stop:
        # Stop any ongoing speech
        js_code = """
        <script>
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel();
            console.log('🔇 Speech stopped');
        }
        </script>
        """
        components.html(js_code, height=0)
        return

    # Clean text for speaking (remove markdown, citations, etc.)
    clean_text = text.replace("**", "").replace("*", "")

    # Remove source citations from spoken text
    if "Sources:" in clean_text:
        clean_text = clean_text.split("Sources:")[0].strip()

    # Escape quotes and newlines for JavaScript
    clean_text = clean_text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')

    # Voice selection patterns - these patterns help select better quality voices
    voice_patterns = {
        "normal": ["Alex", "Daniel", "Microsoft David", "Google US English"],
        "female": ["Samantha", "Karen", "Victoria", "Zira", "Microsoft Zira", "Google UK English Female"],
    }

    patterns = voice_patterns.get(voice_type, voice_patterns["normal"])
    pattern_str = '", "'.join(patterns)

    js_code = f"""
    <script>
    function speakNow() {{
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();

            // Get available voices
            let voices = window.speechSynthesis.getVoices();

            // If voices aren't loaded yet, wait for them
            if (voices.length === 0) {{
                window.speechSynthesis.onvoiceschanged = function() {{
                    voices = window.speechSynthesis.getVoices();
                    speakWithVoice(voices);
                }};
            }} else {{
                speakWithVoice(voices);
            }}

            function speakWithVoice(voices) {{
                const utterance = new SpeechSynthesisUtterance('{clean_text}');
                utterance.lang = 'en-US';
                utterance.rate = {rate};
                utterance.volume = 1.0;

                // Voice preferences for {voice_type}
                const voicePreferences = ["{pattern_str}"];

                // Try to find a preferred voice
                let selectedVoice = null;

                // First pass: exact matches
                for (let pref of voicePreferences) {{
                    selectedVoice = voices.find(v => v.name.includes(pref));
                    if (selectedVoice) break;
                }}

                // Second pass: any en-US voice if no preference found
                if (!selectedVoice) {{
                    selectedVoice = voices.find(v => v.lang.startsWith('en-US') || v.lang.startsWith('en_US'));
                }}

                // Third pass: any English voice
                if (!selectedVoice) {{
                    selectedVoice = voices.find(v => v.lang.startsWith('en'));
                }}

                // Use selected voice if found
                if (selectedVoice) {{
                    utterance.voice = selectedVoice;
                    console.log('🔊 Using voice:', selectedVoice.name);
                }} else {{
                    console.log('🔊 Using default voice');
                }}

                utterance.onend = function() {{
                    console.log('✅ Speech finished');
                }};

                utterance.onerror = function(e) {{
                    console.error('Speech error:', e);
                }};

                window.speechSynthesis.speak(utterance);
            }}
        }} else {{
            console.error('Speech synthesis not supported');
        }}
    }}

    // Small delay to ensure proper initialization
    setTimeout(speakNow, 100);
    </script>
    """
    components.html(js_code, height=0)


def ocr_image_to_text(image: Image.Image):
    """Extract text from image using EasyOCR"""
    try:
        # Convert PIL Image to numpy array
        import numpy as np
        img_array = np.array(image)

        # Perform OCR
        results = ocr_reader.readtext(img_array)

        # Extract text from results
        text = ' '.join([result[1] for result in results])
        return text.strip()
    except Exception as e:
        st.error(f"Error performing OCR: {e}")
        return None


def extract_text_from_file(uploaded_file):
    """Extract text from various file types"""
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension in ['png', 'jpeg', 'jpg']:
            # OCR for images
            image = Image.open(uploaded_file)
            text = ocr_image_to_text(image)
            if text is None:
                return None, None
            return text, "image"

        elif file_extension == 'txt':
            # Plain text
            return uploaded_file.read().decode('utf-8'), "text"

        elif file_extension == 'pdf':
            # PDF extraction
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip(), "pdf"
            except ImportError:
                st.error("PyPDF2 not installed. Install with: pip install PyPDF2")
                return None, None

        elif file_extension in ['docx', 'doc']:
            # Word document extraction
            try:
                import docx
                doc = docx.Document(uploaded_file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text.strip(), "document"
            except ImportError:
                st.error("python-docx not installed. Install with: pip install python-docx")
                return None, None

        else:
            st.warning(f"Unsupported file type: .{file_extension}")
            return None, None

    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return None, None


def search_index(query_text):
    q_vec = embed_query(query_text)
    sims = []
    for item in index:
        sim = cosine_similarity(q_vec, item["embedding"])
        sims.append((sim, item))
    sims.sort(key=lambda x: x[0], reverse=True)
    return sims[:5], sims[0][0] if sims else 0.0


def load_user_conversations(username):
    """Load all conversations for a user"""
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{username}_conversations.json")
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_user_conversations(username, conversations):
    """Save all conversations for a user"""
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{username}_conversations.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2, ensure_ascii=False)


def create_new_conversation():
    """Create a new conversation with a unique ID"""
    conv_id = str(uuid.uuid4())
    return {
        "id": conv_id,
        "title": "New Chat",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "messages": []
    }


def get_conversation_title(messages):
    """Generate a title from the first user message"""
    if messages:
        first_msg = next((m for m in messages if m["role"] == "user"), None)
        if first_msg:
            content = first_msg["content"][:50]
            return content + ("..." if len(first_msg["content"]) > 50 else "")
    return "New Chat"


# --- Login / Session Management ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.conversations = {}
    st.session_state.current_conversation_id = None

# Initialize email agent state
if "email_mode" not in st.session_state:
    st.session_state.email_mode = False
if "email_agent" not in st.session_state:
    st.session_state.email_agent = None

# Initialize settings in session state if not present
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 500
if "voice" not in st.session_state:
    st.session_state.voice = "normal"
if "speech_rate" not in st.session_state:
    st.session_state.speech_rate = 0.9
if "should_speak" not in st.session_state:
    st.session_state.should_speak = False
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""

if not st.session_state.logged_in:
    st.title("🔐 Knowledge Base AI Assistant - Login")

    with st.form("login_form"):
        username_input = st.text_input("Enter your username:")
        submit_button = st.form_submit_button("Login")

        if submit_button:
            if username_input.strip():
                st.session_state.logged_in = True
                st.session_state.username = username_input.strip()
                st.session_state.conversations = load_user_conversations(st.session_state.username)

                # If no conversations exist, create a new one
                if not st.session_state.conversations:
                    new_conv = create_new_conversation()
                    st.session_state.conversations[new_conv["id"]] = new_conv
                    st.session_state.current_conversation_id = new_conv["id"]
                else:
                    # Load the most recent conversation
                    sorted_convs = sorted(
                        st.session_state.conversations.items(),
                        key=lambda x: x[1]["updated_at"],
                        reverse=True
                    )
                    st.session_state.current_conversation_id = sorted_convs[0][0]

                st.rerun()
            else:
                st.warning("Please enter a username.")
    st.stop()

# --- Sidebar with Conversations ---
with st.sidebar:
    st.header(f"👤 {st.session_state.username}")

    if st.button("➕ New Chat", use_container_width=True):
        new_conv = create_new_conversation()
        st.session_state.conversations[new_conv["id"]] = new_conv
        st.session_state.current_conversation_id = new_conv["id"]
        st.session_state.email_mode = False  # Reset email mode
        st.session_state.email_agent = None
        save_user_conversations(st.session_state.username, st.session_state.conversations)
        st.rerun()

    st.divider()

    # Email Agent Button
    st.header("✉️ Quick Actions")

    if st.session_state.email_mode:
        if st.button("🚫 Exit Email Mode", use_container_width=True, type="primary"):
            st.session_state.email_mode = False
            st.session_state.email_agent = None
            st.rerun()
        st.info("📧 Email mode active. The assistant will help you compose and send an email.")
    else:
        if st.button("📧 Send Email", use_container_width=True):
            st.session_state.email_mode = True
            st.session_state.email_agent = EmailAgent()
            # Add initial email prompt to conversation
            current_conv = st.session_state.conversations.get(st.session_state.current_conversation_id)
            if current_conv:
                initial_message = {
                    "role": "assistant",
                    "content": "I'll help you send an email. Please provide the recipient's email address.",
                    "timestamp": datetime.now().isoformat()
                }
                current_conv["messages"].append(initial_message)
                current_conv["updated_at"] = datetime.now().isoformat()
                save_user_conversations(st.session_state.username, st.session_state.conversations)
            st.rerun()

    st.divider()
    st.header("🎛️ AI Settings")

    # Store settings in session state so they persist
    st.session_state.temperature = st.slider(
        "Temperature",
        0.0, 1.0,
        st.session_state.temperature,
        0.1,
        help="Higher values make output more random, lower values more focused"
    )

    st.session_state.max_tokens = st.slider(
        "Max tokens",
        100, 2000,
        st.session_state.max_tokens,
        50,
        help="Maximum length of the response"
    )

    st.subheader("🗣️ Voice Settings")
    st.session_state.voice = st.selectbox(
        "AI Voice",
        ["normal", "female", "deep", "bright"],
        index=["normal", "female", "deep", "bright"].index(st.session_state.voice)
    )

    st.session_state.speech_rate = st.slider(
        "Speech speed",
        0.5, 1.5,
        st.session_state.speech_rate,
        0.1
    )

    if st.button("🔊 Test Voice"):
        speak_text(
            "This is a test of the AI voice settings.",
            st.session_state.voice,
            st.session_state.speech_rate
        )

    st.divider()
    st.subheader("💬 Conversations")

    # Sort conversations by most recent
    sorted_conversations = sorted(
        st.session_state.conversations.items(),
        key=lambda x: x[1]["updated_at"],
        reverse=True
    )

    for conv_id, conv in sorted_conversations:
        # Update title if messages exist
        if conv["messages"]:
            conv["title"] = get_conversation_title(conv["messages"])

        # Create a button for each conversation
        is_current = conv_id == st.session_state.current_conversation_id
        button_label = f"💬 {conv['title']}"

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                    button_label,
                    key=f"conv_{conv_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
            ):
                st.session_state.current_conversation_id = conv_id
                st.session_state.email_mode = False  # Reset email mode when switching conversations
                st.session_state.email_agent = None
                st.rerun()

        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                del st.session_state.conversations[conv_id]
                # If deleting current conversation, switch to another or create new
                if conv_id == st.session_state.current_conversation_id:
                    if st.session_state.conversations:
                        st.session_state.current_conversation_id = list(st.session_state.conversations.keys())[0]
                    else:
                        new_conv = create_new_conversation()
                        st.session_state.conversations[new_conv["id"]] = new_conv
                        st.session_state.current_conversation_id = new_conv["id"]
                    st.session_state.email_mode = False
                    st.session_state.email_agent = None
                save_user_conversations(st.session_state.username, st.session_state.conversations)
                st.rerun()

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.conversations = {}
        st.session_state.current_conversation_id = None
        st.session_state.email_mode = False
        st.session_state.email_agent = None
        st.rerun()

# --- Main Chat Interface ---
st.title("Yachay")

# Get current conversation
current_conv = st.session_state.conversations.get(st.session_state.current_conversation_id)

if not current_conv:
    st.error("No conversation selected")
    st.stop()

# Display chat history
for i, msg in enumerate(current_conv["messages"]):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
            if "file" in msg:
                st.caption(f"📎 {msg['file']['name']} ({msg['file']['type']})")
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])

            # Add play/stop button for each assistant message
            col1, col2 = st.columns([6, 1])
            with col2:
                button_key = f"speak_btn_{i}"
                if st.button("🔊", key=button_key, help="Play/Stop audio"):
                    # Toggle speech state
                    if st.session_state.get(f"speaking_{i}", False):
                        # Stop speaking
                        speak_text("", stop=True)
                        st.session_state[f"speaking_{i}"] = False
                    else:
                        # Start speaking
                        speak_text(
                            msg["content"],
                            st.session_state.voice,
                            st.session_state.speech_rate
                        )
                        st.session_state[f"speaking_{i}"] = True

# File uploader above chat input (only show when not in email mode)
uploaded_file = None
if not st.session_state.email_mode:
    uploaded_file = st.file_uploader("📎 Add files (images, PDFs, documents)",
                                     type=['png', 'jpeg', 'jpg', 'pdf', 'txt', 'docx', 'doc'])

# Chat input area at the bottom
user_input = st.chat_input("Ask a question about your knowledge base..." if not st.session_state.email_mode
                           else "Continue with email details...")

# Process user input
if user_input:
    user_message = {
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().isoformat()
    }

    # Check if file was uploaded (only in non-email mode)
    extracted_text = None
    file_type = None
    if uploaded_file and not st.session_state.email_mode:
        extracted_text, file_type = extract_text_from_file(uploaded_file)
        if extracted_text:
            user_message["file"] = {
                "name": uploaded_file.name,
                "type": file_type
            }
            user_input = f"File content from {uploaded_file.name}:\n{extracted_text}\n\nQuestion: {user_input}"

    current_conv["messages"].append(user_message)
    current_conv["updated_at"] = datetime.now().isoformat()

    # Display user message
    with st.chat_message("user"):
        st.write(user_message["content"])
        if "file" in user_message:
            st.caption(f"📎 {user_message['file']['name']} ({user_message['file']['type']})")

    # Process based on mode
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            # EMAIL MODE: Use the EmailAgent class
            if st.session_state.email_mode and st.session_state.email_agent:
                response = st.session_state.email_agent.process_message(user_input)
                st.write(response)

                # Check if email was sent successfully to exit email mode
                if "Email successfully sent" in response or "Email sending cancelled" in response:
                    st.session_state.email_mode = False
                    st.session_state.email_agent = None

            # NORMAL MODE: Use knowledge base
            else:
                top_results, top_sim = search_index(user_input)

                # Prepare context
                context_chunks = [{"text": item["text"], "source": item.get("source", "unknown")}
                                  for _, item in top_results if _ > SIMILARITY_THRESHOLD]

                # Get conversational history for context
                conv_history = [{"role": msg["role"], "content": msg["content"]}
                                for msg in current_conv["messages"][:-1]]

                # Generate answer using settings from session state
                answer = ask_groq(
                    user_input,
                    context_chunks,
                    conv_history,
                    temperature=st.session_state.temperature,
                    max_tokens=st.session_state.max_tokens
                )

                st.write(answer)
                response = answer

                # Show sources if context was used
                if context_chunks:
                    sources = sorted({c["source"] for c in context_chunks})
                    st.caption(f"📚 Sources: {', '.join(sources)}")

    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    }
    current_conv["messages"].append(assistant_message)
    current_conv["updated_at"] = datetime.now().isoformat()

    # Save all conversations
    save_user_conversations(st.session_state.username, st.session_state.conversations)

    # Clear the file uploader by rerunning

    st.rerun()



