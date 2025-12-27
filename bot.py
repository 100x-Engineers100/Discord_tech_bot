"""
Discord Support Bot V2 - Technical Learning Assistant
====================================================
A RAG-powered Discord bot that helps students with technical queries in a forum setting.

Features:
- RAG (Retrieval-Augmented Generation) using FAISS vector store
- Per-thread conversation history (last 5 exchanges)
- Image analysis support via OpenAI Vision
- Forum thread detection with bot mention handling
- Smart message splitting for long responses
"""

import os
import re
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv

import discord
from discord.ext import commands

import torch
from openai import OpenAI

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Discord Bot Configuration
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = "gpt-4.1-mini"  # Using the model you specified

# RAG Configuration
CHUNK_SIZE = 1500  # Size of text chunks for embedding
CHUNK_OVERLAP = 150  # Overlap between chunks to maintain context
MAX_HISTORY_MESSAGES = 5  # Number of conversation exchanges to remember per thread

# Message Configuration
MAX_DISCORD_MESSAGE_LENGTH = 1900  # Discord limit is 2000, keeping buffer

# File Paths (relative to script location for deployment compatibility)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = os.path.join(BASE_DIR, "Data_Doc_main.txt")

# ============================================================================
# GLOBAL STATE
# ============================================================================

# Store conversation history per thread
# Structure: {thread_id: [{"role": "user/assistant", "content": "..."}]}
conversation_history: Dict[int, List[Dict[str, str]]] = {}

# RAG components (initialized on bot startup)
vector_store = None
openai_client = None

# Semaphore to limit concurrent API calls (prevents rate limiting)
api_semaphore = asyncio.Semaphore(3)

# ============================================================================
# DISCORD BOT SETUP
# ============================================================================

# Configure Discord intents (permissions for bot functionality)
intents = discord.Intents.default()
intents.message_content = True  # Required to read message text
intents.messages = True  # Required to receive message events

# Initialize bot with command prefix (not used for slash commands, but required)
bot = commands.Bot(command_prefix='!', intents=intents)

# ============================================================================
# RAG SYSTEM FUNCTIONS
# ============================================================================

def load_and_preprocess_text(file_path: str) -> str:
    """
    Load text file and preprocess it for embedding.
    
    Args:
        file_path: Path to the text file containing curriculum data
        
    Returns:
        Preprocessed text ready for chunking and embedding
        
    Preprocessing steps:
        1. Read file with UTF-8 encoding
        2. Remove non-ASCII characters (prevents encoding issues)
        3. Convert to lowercase (improves semantic matching)
        4. Remove special characters except spaces
    """
    # Read the raw text file
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Remove non-ASCII characters (keeps only standard English chars)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Normalize to lowercase for better semantic matching
    text = text.lower()
    
    # Remove punctuation and special characters (keeps alphanumeric + spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    return text


def create_vector_store(text: str) -> FAISS:
    """
    Create FAISS vector store from preprocessed text.
    
    Args:
        text: Preprocessed curriculum text
        
    Returns:
        FAISS vector store with embedded document chunks
        
    Process:
        1. Convert text into LangChain Document object
        2. Split into overlapping chunks (maintains context)
        3. Generate embeddings using HuggingFace model
        4. Store embeddings in FAISS index for fast retrieval
    """
    print("Creating vector store...")
    
    # Wrap text in LangChain Document object
    doc = Document(page_content=text)
    
    # Initialize text splitter with overlap to maintain context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,  # Use character count for length
        separators=["\n\n", "\n", " ", ""]  # Split priority: paragraphs > lines > words
    )
    
    # Split document into chunks
    chunks = text_splitter.split_documents([doc])
    
    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not chunks:
        raise ValueError("No valid document chunks created. Check your data file.")
    
    print(f"Created {len(chunks)} document chunks")
    
    # Detect if GPU is available for faster embedding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize embedding model (384-dimensional vectors)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Fast, efficient model
        model_kwargs={"device": device}
    )
    
    # Create FAISS vector store from chunks
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    print("Vector store created successfully")
    return vector_store


def retrieve_relevant_context(query: str, k: int = 3) -> str:
    """
    Retrieve most relevant document chunks for a query.
    
    Args:
        query: User's question or search query
        k: Number of relevant chunks to retrieve (default: 3)
        
    Returns:
        Concatenated text from top-k most relevant chunks
        
    How it works:
        1. Convert query to embedding vector
        2. Search FAISS index for nearest neighbors
        3. Return content from top-k matches
    """
    if not vector_store:
        return ""
    
    # Search vector store for most similar chunks
    docs = vector_store.similarity_search(query, k=k)
    
    # Concatenate retrieved chunks with separators
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    
    return context


# ============================================================================
# CONVERSATION MANAGEMENT
# ============================================================================

def get_thread_history(thread_id: int) -> List[Dict[str, str]]:
    """
    Retrieve conversation history for a specific thread.
    
    Args:
        thread_id: Discord thread ID
        
    Returns:
        List of message dictionaries with 'role' and 'content' keys
        Returns empty list if no history exists
    """
    return conversation_history.get(thread_id, [])


def add_to_thread_history(thread_id: int, role: str, content: str):
    """
    Add a message to thread's conversation history.
    
    Args:
        thread_id: Discord thread ID
        role: Either "user" or "assistant"
        content: Message content
        
    Automatically maintains only last MAX_HISTORY_MESSAGES exchanges
    """
    # Initialize history list if thread is new
    if thread_id not in conversation_history:
        conversation_history[thread_id] = []
    
    # Append new message
    conversation_history[thread_id].append({
        "role": role,
        "content": content
    })
    
    # Keep only last N messages to prevent context overflow
    conversation_history[thread_id] = conversation_history[thread_id][-MAX_HISTORY_MESSAGES:]


def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Convert history list into formatted string for LLM context.
    
    Args:
        history: List of conversation messages
        
    Returns:
        Formatted string showing conversation flow
        
    Example output:
        "User: How do I use ControlNet?
         Assistant: ControlNet is a tool that..."
    """
    if not history:
        return "No previous conversation in this thread."
    
    formatted = []
    for msg in history:
        role_label = "Student" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role_label}: {msg['content'][:200]}...")  # Truncate long messages
    
    return "\n".join(formatted)


# ============================================================================
# OPENAI INTEGRATION
# ============================================================================

async def generate_response(
    query: str, 
    context: str, 
    history: List[Dict[str, str]],
    image_url: Optional[str] = None
) -> str:
    """
    Generate response using OpenAI with RAG context and conversation history.
    
    Args:
        query: User's question
        context: Retrieved relevant documentation from RAG
        history: Previous conversation messages in this thread
        image_url: Optional image URL if user attached an image
        
    Returns:
        Generated response from OpenAI
        
    Uses semaphore to limit concurrent API calls and prevent rate limiting
    """
    async with api_semaphore:  # Limit concurrent API calls
        try:
            system_prompt = f"""You're Sage - you help students debug their AI projects. You've been through this curriculum yourself.

CURRICULUM CONTEXT:
{context}

RECENT CONVERSATION:
{format_history_for_prompt(history)}

HOW YOU TALK (examples):

Student: "I'm getting errors with ControlNet in ComfyUI"
You: "What's the error message? Also - are you using the workflow from Lecture 5 or building custom?"

Student: "How do I deploy my RAG app?"
You: "Lecture 10 in Module 2 covers this - did you check out the Replicate deployment section? What's blocking you specifically?"

Student: "Explain how FLUX works"
You: "That's a big topic. What part are you stuck on? The model architecture, or actually using it in ComfyUI? (Lectures 6-7 cover both)"

Student: "My code isn't working [shares screenshot]"
You: "I see the issue - you're missing the API key in line 23. This was covered in Week 2's FastAPI lecture. Add it to your .env file."

IMPORTANT:
- Ask clarifying questions FIRST if the query is vague
- Don't dump everything - respond to what they actually asked
- Only explain more if they ask for it
- Cite lectures naturally, like "Week 8 covered this" not "According to Lecture 8..."
- If you don't know, say "not sure - which module is this from?"

Current question: {query}"""

            # Prepare messages for OpenAI API
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Handle image + text query
            if image_url:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                })
            else:
                # Text-only query
                messages.append({
                    "role": "user",
                    "content": query
                })
            
            # Call OpenAI API
            response = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=500,# Limit response length
                temperature=0.7, # Balance between creativity and consistency  
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm having trouble generating a response right now. Please try again in a moment."


# ============================================================================
# MESSAGE HANDLING UTILITIES
# ============================================================================

def split_long_message(content: str) -> List[str]:
    """
    Split long messages into Discord-compatible chunks.
    
    Args:
        content: Full response text
        
    Returns:
        List of message chunks, each under Discord's character limit
        
    Splitting strategy:
        1. Try splitting at paragraph breaks (\n\n)
        2. If paragraphs too long, split at sentences
        3. If sentences too long, split at word boundaries
        4. Never split mid-word
    """
    if len(content) <= MAX_DISCORD_MESSAGE_LENGTH:
        return [content]
    
    chunks = []
    
    # Split by paragraphs first
    paragraphs = content.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph exceeds limit
        if len(current_chunk) + len(para) + 2 > MAX_DISCORD_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                # Single paragraph is too long, need to split it
                words = para.split()
                for word in words:
                    if len(current_chunk) + len(word) + 1 > MAX_DISCORD_MESSAGE_LENGTH:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk += " " + word if current_chunk else word
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# ============================================================================
# DISCORD EVENT HANDLERS
# ============================================================================

@bot.event
async def on_ready():
    """
    Called when bot successfully connects to Discord.
    
    Initializes:
        1. OpenAI client
        2. RAG vector store
        3. Logs ready status
    """
    global openai_client, vector_store
    
    print(f'Bot logged in as {bot.user.name} (ID: {bot.user.id})')
    print('----------------------------------------')
    
    # Initialize OpenAI client
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print('✓ OpenAI client initialized')
    
    # Load and create vector store
    print('Loading curriculum data...')
    text = load_and_preprocess_text(DATA_FILE_PATH)
    vector_store = create_vector_store(text)
    print('✓ RAG system initialized')
    
    print('----------------------------------------')
    print('Bot is ready to help students!')


@bot.event
async def on_message(message):
    """
    Main message handler - processes all incoming messages.
    
    Workflow:
        1. Check if message is in a forum thread
        2. Check if bot was mentioned
        3. Ignore bot's own messages
        4. Extract query and optional image
        5. Retrieve relevant context from RAG
        6. Generate response with OpenAI
        7. Send response (split if needed)
        8. Update conversation history
    """
    # Ignore messages from the bot itself (prevents infinite loops)
    if message.author == bot.user:
        return
    
    # Only respond in forum threads (public_thread or private_thread)
    if message.channel.type not in [discord.ChannelType.public_thread, discord.ChannelType.private_thread]:
        return
    
    # Only respond if bot is mentioned
    if bot.user not in message.mentions:
        return
    
    # Extract thread ID for history tracking
    thread_id = message.channel.id
    
    # Remove bot mention from query text
    query = message.content
    for mention in message.mentions:
        query = query.replace(f'<@{mention.id}>', '').strip()
    
    # Handle empty query (user just mentioned bot)
    if not query and not message.attachments:
        await message.reply(
            "Hi! I'm Sage, your technical learning assistant. "
            "Please share your doubt or question, and I'll do my best to assist you!"
        )
        return
    
    # Show typing indicator while processing
    async with message.channel.typing():
        try:
            # Check for image attachment
            image_url = None
            if message.attachments:
                # Get first image attachment
                for attachment in message.attachments:
                    if any(attachment.filename.lower().endswith(ext) 
                           for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        image_url = attachment.url
                        break
            
            # Retrieve relevant context from curriculum
            context = retrieve_relevant_context(query, k=3)
            
            # Get conversation history for this thread
            history = get_thread_history(thread_id)
            
            # Generate response using OpenAI
            response = await generate_response(query, context, history, image_url)
            
            # Split response if too long for Discord
            message_chunks = split_long_message(response)
            
            # Send response (first chunk as reply, rest as follow-ups)
            first_message = await message.reply(message_chunks[0])
            
            for chunk in message_chunks[1:]:
                await message.channel.send(chunk)
                await asyncio.sleep(0.5)  # Small delay to maintain order
            
            # Update conversation history
            add_to_thread_history(thread_id, "user", query)
            add_to_thread_history(thread_id, "assistant", response)
            
        except Exception as e:
            print(f"Error processing message: {e}")
            await message.reply(
                "I encountered an error while processing your question. "
                "Please try rephrasing or contact a mentor if the issue persists."
            )


# ============================================================================
# BOT STARTUP
# ============================================================================

if __name__ == "__main__":
    """
    Entry point: Start the Discord bot.
    
    Note: The bot runs indefinitely until manually stopped (Ctrl+C)
    or the process is killed by the hosting platform.
    """
    if not DISCORD_BOT_TOKEN:
        print("ERROR: DISCORD_BOT_TOKEN not found in environment variables")
        print("Please create a .env file with your bot token")
        exit(1)
    
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not found in environment variables")
        print("Please add your OpenAI API key to .env file")
        exit(1)
    
    print("Starting Discord bot...")
    bot.run(DISCORD_BOT_TOKEN)