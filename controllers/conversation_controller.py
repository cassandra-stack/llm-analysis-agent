from typing import List, Dict
from fastapi import HTTPException, Depends
from google import genai
from google.genai import types
import uuid
import os
import json
import logging
import time
from pathlib import Path
from markdown_pdf import MarkdownPdf, Section
from sqlalchemy.orm import Session

from schemas.models import (
    StartConversationRequest, 
    ContinueConversationRequest, 
    StartConversationResponse,
    ConversationResponse,
    ConversationHistory,
    Message as MessageSchema
)
from database import get_db, Conversation, Message

# Configuration from environment variables
RAG_CORPUS = os.getenv("RAG_CORPUS")
MODEL_ID = os.getenv("MODEL_ID")
GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_system_prompt() -> str:
    """Read system prompt from text file"""
    with open("/app/prompts/system_prompt.txt", "r", encoding="utf-8") as file:
        return file.read().strip()

def read_report_task() -> str:
    """Read report task instructions from text file"""
    with open("/app/prompts/report_task.txt", "r", encoding="utf-8") as file:
        return file.read().strip()

def read_study_metrics(study_code: str) -> Dict:
    """Read study metrics from JSON file"""
    study_dir = Path("storage") / "studies" / study_code
    metrics_file = study_dir / "metrics.json"
    
    with open(metrics_file, "r", encoding="utf-8") as file:
        return json.load(file)

def save_report_md(study_code: str, report_content: str) -> str:
    """Save the generated report as markdown file in the study directory"""
    study_dir = Path("storage") / "studies" / study_code
    study_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = study_dir / "report.md"
    
    with open(report_file, "w", encoding="utf-8") as file:
        file.write(report_content)
    
    return str(report_file)

def convert_md_to_pdf(study_code: str, markdown_content: str) -> str:
    """Convert markdown content to PDF and save it in the study directory"""
    study_dir = Path("storage") / "studies" / study_code
    study_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_file = study_dir / "report.pdf"
    
    # Create PDF with table of contents from headings up to level 3
    pdf = MarkdownPdf(toc_level=3, optimize=True)
    
    # Add the markdown content as a section with custom CSS for better formatting
    css = """
    body { font-family: Arial, sans-serif; line-height: 1.6; }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
    h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }
    h3 { color: #7f8c8d; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    table, th, td { border: 1px solid #bdc3c7; }
    th { background-color: #ecf0f1; font-weight: bold; padding: 12px; text-align: left; }
    td { padding: 10px; }
    strong { color: #2c3e50; }
    ul { margin: 10px 0; }
    li { margin: 5px 0; }
    """
    
    pdf.add_section(Section(markdown_content), user_css=css)
    
    # Set PDF metadata
    pdf.meta["title"] = f"Brain Tumor Analysis Report - {study_code}"
    pdf.meta["author"] = "LLM Analysis Agent"
    pdf.meta["subject"] = "Medical Brain Tumor Segmentation Analysis"
    pdf.meta["creator"] = "LLM Analysis Agent API"
    
    # Save the PDF
    pdf.save(str(pdf_file))
    
    return str(pdf_file)

def generate_response(messages: List[Dict]) -> str:
    """Generate response using Google genai with RAG"""
    
    logger.info(f"Starting response generation with {len(messages)} messages")
    logger.info(f"Using model: {MODEL_ID}")
    logger.info(f"Using RAG corpus: {RAG_CORPUS}")
    
    start_time = time.time()
    
    try:
        client = genai.Client(
            vertexai=True,
            api_key=GOOGLE_CLOUD_API_KEY
        )
        logger.info("GenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize GenAI client: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize GenAI client: {str(e)}")

    # Convert messages to genai format
    contents = []
    for i, msg in enumerate(messages):
        role = "model" if msg["role"] == "assistant" else "user"
        contents.append(types.Content(
            role=role,
            parts=[types.Part(text=msg["content"])]
        ))
        logger.debug(f"Message {i}: role={role}, content_length={len(msg['content'])}")

    # Configure RAG tools
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus=RAG_CORPUS
                        )
                    ],
                )
            )
        )
    ]
    logger.info("RAG tools configured")

    # Generation configuration
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        seed=0,
        max_output_tokens=65535,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF"
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF"
            )
        ],
        tools=tools,
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
    )
    logger.info("Generation configuration created")

    # Generate response
    full_response = ""
    chunk_count = 0
    try:
        logger.info("Starting content generation stream...")
        for chunk in client.models.generate_content_stream(
            model=MODEL_ID,
            contents=contents,
            config=generate_content_config,
        ):
            chunk_count += 1
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                logger.debug(f"Chunk {chunk_count}: Empty or invalid chunk")
                continue
            full_response += chunk.text
            if chunk_count % 10 == 0:  # Log every 10 chunks
                logger.debug(f"Processed {chunk_count} chunks, response length: {len(full_response)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Content generation completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Total chunks processed: {chunk_count}")
        logger.info(f"Final response length: {len(full_response)} characters")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Content generation failed after {elapsed_time:.2f} seconds")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {error_msg}")
        
        # Check for specific error types
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            logger.error("Rate limit or quota exceeded - this is a Google API limit")
            logger.error("Consider implementing exponential backoff or reducing request frequency")
        elif "PERMISSION_DENIED" in error_msg:
            logger.error("Permission denied - check API key and project permissions")
        elif "INVALID_ARGUMENT" in error_msg:
            logger.error("Invalid argument - check request parameters")
        
        logger.error(f"Full error details: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    
    return full_response

async def start_conversation(req: StartConversationRequest, db: Session) -> StartConversationResponse:
    """Start a new conversation with RAG"""
    
    logger.info(f"Starting conversation for study_id: {req.study_id}, study_code: {req.study_code}")
    
    conversation_id = str(uuid.uuid4())
    
    messages = []
    
    system_prompt = req.system_prompt if req.system_prompt else read_system_prompt()
    messages.append({"role": "system", "content": system_prompt})
    
    try:
        study_metrics = read_study_metrics(req.study_code)
        logger.info(f"Successfully loaded study metrics for {req.study_code}")
    except Exception as e:
        logger.error(f"Failed to load study metrics for {req.study_code}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Study metrics not found for {req.study_code}")
    
    report_task = read_report_task()

    # Combine report task and study metrics
    enhanced_prompt = f"""
    {report_task}

    Study Metrics Data:
    {json.dumps(study_metrics, indent=2)}
    """
    
    # Add enhanced user message
    messages.append({"role": "user", "content": enhanced_prompt})
    
    # Generate response
    logger.info("About to call generate_response")
    response_text = generate_response(messages)
    logger.info("Successfully generated response")
    
    # Extract JSON from markdown code blocks and then extract report_md
    logger.info("Processing response to extract report markdown")
    if response_text.strip().startswith("```json"):
        start_marker = "```json"
        end_marker = "```"
        start_idx = response_text.find(start_marker) + len(start_marker)
        end_idx = response_text.rfind(end_marker)
        json_content = response_text[start_idx:end_idx].strip()
        logger.info("Extracted JSON from markdown code blocks")
    else:
        json_content = response_text.strip()
        logger.info("Using response as-is (no markdown code blocks)")
    
    # Parse JSON and extract report_md
    try:
        response_data = json.loads(json_content)
        logger.info("Successfully parsed JSON response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {str(e)}")
        logger.error(f"JSON content preview: {json_content[:200]}...")
        raise HTTPException(status_code=500, detail="Failed to parse AI response as JSON")
    
    # Extract the report_md content from the patient data
    if isinstance(response_data, list) and len(response_data) > 0:
        patient_data = response_data[0]
        if "report_md" in patient_data:
            report_md_content = patient_data["report_md"]
            logger.info("Successfully extracted report_md from response")
            try:
                report_file_path = save_report_md(req.study_code, report_md_content)
                logger.info(f"Saved markdown report to: {report_file_path}")
                pdf_file_path = convert_md_to_pdf(req.study_code, report_md_content)
                logger.info(f"Converted to PDF: {pdf_file_path}")
            except Exception as e:
                logger.error(f"Failed to save reports: {str(e)}")
                # Continue anyway - don't fail the whole request
        else:
            logger.warning("No report_md found in response, saving full response")
            report_file_path = save_report_md(req.study_code, response_text)
    else:
        logger.warning("Unexpected response format, saving full response")
        report_file_path = save_report_md(req.study_code, response_text)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response_text})
    
    # Store conversation in database
    try:
        logger.info(f"Storing conversation in database with ID: {conversation_id}")
        db_conversation = Conversation(
            id=conversation_id,
            study_id=str(req.study_id),
            system_prompt=req.system_prompt
        )
        db.add(db_conversation)
        db.flush()
        
        # Store messages in database
        logger.info(f"Storing {len(messages)} messages in database")
        for msg in messages:
            db_message = Message(
                conversation_id=conversation_id,
                role=msg["role"],
                content=msg["content"]
            )
            db.add(db_message)
        
        db.commit()
        logger.info("Successfully stored conversation and messages in database")
        
    except Exception as e:
        logger.error(f"Failed to store conversation in database: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to store conversation")
    
    logger.info(f"Conversation completed successfully for study {req.study_code}")
    
    return StartConversationResponse(
        conversation_id=conversation_id,
        study_id=req.study_id
    )

def detect_correction(question: str) -> bool:
    """Detect if the question contains correction keywords"""
    correction_keywords = [
        "cassandra correction",
        "correction",
    ]
    
    question_lower = question.lower()
    for keyword in correction_keywords:
        if keyword in question_lower:
            return True
    return False

async def continue_conversation(conversation_id: str, req: ContinueConversationRequest, db: Session) -> ConversationResponse:
    """Continue an existing conversation"""
    
    logger.info(f"Continuing conversation {conversation_id} with question: {req.question[:100]}...")
    
    # Check if this is a correction
    is_correction = detect_correction(req.question)
    
    if is_correction:
        logger.info(f"Detected correction in conversation {conversation_id}: {req.question}")
        
        # Store the correction message in database
        # user_message = Message(
        #     conversation_id=conversation_id,
        #     role="user",
        #     content=req.question
        # )
        # db.add(user_message)
        # db.commit()
        
        # Return feedback response without generating LLM response
        return ConversationResponse(
            conversation_id=conversation_id,
            response="Thanks for feedback",
            messages=[],
            feedback=True
        )
    
    # Get conversation from database
    db_conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not db_conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get existing messages from database
    db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).all()
    messages = [{"role": msg.role, "content": msg.content} for msg in db_messages]
    
    # Add new user message
    messages.append({"role": "user", "content": req.question})
    
    # Generate response with full history
    response_text = generate_response(messages)
    
    # Add assistant response
    messages.append({"role": "assistant", "content": response_text})
    
    # Store new messages in database
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=req.question
    )
    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text
    )
    
    db.add(user_message)
    db.add(assistant_message)
    db.commit()
    
    return ConversationResponse(
        conversation_id=conversation_id,
        response=response_text,
        feedback=False
    )

async def get_conversation(conversation_id: str, db: Session) -> ConversationHistory:
    """Get conversation history"""
    
    # Get conversation from database
    db_conversation = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not db_conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Get messages from database
    db_messages = db.query(Message).filter(Message.conversation_id == conversation_id).all()
    
    return ConversationHistory(
        conversation_id=conversation_id,
        study_id=int(db_conversation.study_id),
        messages=[MessageSchema(role=msg.role, content=msg.content) for msg in db_messages]
    )

async def health_check() -> Dict:
    """Health check endpoint that verifies configuration and API access"""
    try:
        config_status = {
            "RAG_CORPUS": bool(RAG_CORPUS),
            "MODEL_ID": bool(MODEL_ID),
            "GOOGLE_CLOUD_API_KEY": bool(GOOGLE_CLOUD_API_KEY) and len(GOOGLE_CLOUD_API_KEY) > 10
        }

        try:
            client = genai.Client(
                vertexai=True,
                api_key=GOOGLE_CLOUD_API_KEY
            )
            api_status = "api_key_valid"
        except Exception as e:
            api_status = f"api_key_invalid: {str(e)}"
        
        return {
            "status": "healthy",
            "config": config_status,
            "api_status": api_status,
            "model": MODEL_ID,
            "rag_corpus": RAG_CORPUS
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }