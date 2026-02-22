from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import os
import logging
from datetime import datetime

from src.structure_analyzer import PaperStructureAnalyzer
from src.latex_to_speech import EquationToSpeech
from src.table_summarizer import TableSummarizer
from src.tts_engine.tts_audiogeenrator import AudioGenerator
from src.evaluation_metrics import ModelEvaluator

# Initialize FastAPI app
app = FastAPI(
    title="Research Paper Audiobook API",
    description="Convert research papers to smart audiobooks with equation and table handling",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize components
structure_analyzer = PaperStructureAnalyzer(config.get("pdf_processor", {}))
equation_converter = EquationToSpeech()
table_summarizer = TableSummarizer(config.get("table_handler", {}))
audio_generator = AudioGenerator(config.get("tts_engine", {}))
evaluator = ModelEvaluator(config.get("evaluation", {}))

# Store job statuses
jobs = {}

class ConversionRequest(BaseModel):
    """Request model for paper conversion"""
    paper_url: Optional[str] = None
    voice_params: Optional[Dict[str, Any]] = {}
    include_metadata: bool = True
    evaluation_metrics: bool = True

class ConversionResponse(BaseModel):
    """Response model for conversion"""
    job_id: str
    status: str
    message: str
    audio_url: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Research Paper Audiobook Converter",
        "version": "1.0.0",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/convert", response_model=ConversionResponse)
async def convert_paper(
    background_tasks: BackgroundTasks,
    request: ConversionRequest = None,
    file: UploadFile = File(None)
):
    """
    Convert research paper to audiobook
    
    Either provide a URL or upload a PDF file
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file or download from URL
    if file:
        # Handle file upload
        file_path = f"/tmp/{job_id}_{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    elif request and request.paper_url:
        # Handle URL - download file
        import requests
        file_path = f"/tmp/{job_id}_paper.pdf"
        response = requests.get(request.paper_url)
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        raise HTTPException(status_code=400, detail="Either file or paper_url must be provided")
    
    # Initialize job status
    jobs[job_id] = {
        "status": "processing",
        "progress": 0,
        "created_at": datetime.now().isoformat()
    }
    
    # Start background processing
    background_tasks.add_task(
        process_paper,
        job_id,
        file_path,
        request.voice_params if request else {},
        request.include_metadata if request else True,
        request.evaluation_metrics if request else True
    )
    
    return ConversionResponse(
        job_id=job_id,
        status="processing",
        message="Paper conversion started"
    )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]

@app.get("/download/{job_id}")
async def download_audio(job_id: str):
    """Download converted audio file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    audio_path = job.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(
        audio_path,
        media_type="audio/wav",
        filename=f"paper_audiobook_{job_id}.wav"
    )

async def process_paper(
    job_id: str,
    file_path: str,
    voice_params: Dict,
    include_metadata: bool,
    evaluate: bool
):
    """Background task to process paper"""
    try:
        jobs[job_id]["progress"] = 10
        
        # Step 1: Analyze paper structure
        logger.info(f"Analyzing paper structure for job {job_id}")
        structure = structure_analyzer.analyze_paper(file_path)
        jobs[job_id]["progress"] = 30
        
        # Step 2: Process text content
        logger.info(f"Processing text content for job {job_id}")
        text_segments = []
        
        # Add metadata if requested
        if include_metadata and structure["metadata"]["title"]:
            text_segments.append(f"Title: {structure['metadata']['title']}")
            if structure["metadata"]["authors"]:
                text_segments.append(f"Authors: {', '.join(structure['metadata']['authors'])}")
        
        jobs[job_id]["progress"] = 40
        
        # Step 3: Convert equations to speech
        logger.info(f"Converting equations for job {job_id}")
        for eq in structure["equations"]:
            speech = equation_converter.equation_to_speech(eq["latex"])
            text_segments.append(f"Equation: {speech}")
        
        jobs[job_id]["progress"] = 60
        
        # Step 4: Summarize tables
        logger.info(f"Summarizing tables for job {job_id}")
        for table in structure["tables"]:
            # Convert table text to DataFrame (simplified)
            import pandas as pd
            from io import StringIO
            try:
                df = pd.read_csv(StringIO(table["text"]), sep="\s+")
                summary = table_summarizer.summarize_table(df)
                text_segments.append(f"Table summary: {summary['narrative_summary']}")
            except:
                text_segments.append(f"Table found on page {table['page']}")
        
        jobs[job_id]["progress"] = 80
        
        # Step 5: Add section content
        for section in structure["sections"]:
            text_segments.append(f"Section: {section}")
        
        # Step 6: Generate audio
        logger.info(f"Generating audio for job {job_id}")
        full_text = " ".join(text_segments)
        
        # Split text into chunks for TTS
        chunk_size = 500  # characters
        text_chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
        
        audio_segments = []
        for chunk in text_chunks:
            audio = audio_generator.generate_audio(chunk, voice_params)
            audio_segments.append(audio)
        
        # Concatenate audio segments
        final_audio = audio_generator.concatenate_audio_segments(audio_segments)
        
        # Save audio file
        audio_path = f"/tmp/{job_id}_output.wav"
        with open(audio_path, "wb") as f:
            f.write(final_audio.getvalue())
        
        jobs[job_id]["progress"] = 95
        
        # Step 7: Evaluate if requested
        if evaluate:
            logger.info(f"Evaluating results for job {job_id}")
            metrics = evaluator.evaluate_paper_to_speech(
                original_text=open(file_path, 'rb').read().decode('utf-8', errors='ignore'),
                generated_text=full_text,
                audio_file=audio_path
            )
            jobs[job_id]["metrics"] = metrics
        
        # Update job status
        jobs[job_id].update({
            "status": "completed",
            "progress": 100,
            "audio_path": audio_path,
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })
    
    finally:
        # Clean up uploaded file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)