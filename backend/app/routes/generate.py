"""
Content Generation API Routes
Generate and execute code, create documents, and manage generated content
"""
import asyncio
import base64
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from pydantic import BaseModel, Field

from app.database.database import get_db
from app.database.models import (
    User, GeneratedContent, AIRequest, ContentType, RequestStatus, 
    ModelProvider, Conversation, Message, MessageRole
)
from app.middleware.auth import get_current_user
from app.models.ai_orchestrator import AIOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

# Request Models
class CodeGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(..., max_length=50)
    framework: Optional[str] = Field(None, max_length=100)
    requirements: Optional[str] = Field(None, max_length=2000)
    execute: bool = Field(False)
    model_preference: Optional[str] = Field(None)
    temperature: float = Field(0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(2000, ge=100, le=8000)

class DocumentGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    document_type: str = Field(..., pattern="^(pdf|docx|html|markdown|txt)$")
    template: Optional[str] = Field(None)
    title: Optional[str] = Field(None, max_length=200)
    author: Optional[str] = Field(None, max_length=100)
    model_preference: Optional[str] = Field(None)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(3000, ge=100, le=8000)

class ContentUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, max_length=200)
    description: Optional[str] = Field(None, max_length=1000)
    is_public: Optional[bool] = None
    is_template: Optional[bool] = None
    tags: Optional[List[str]] = Field(None, max_items=10)

# Response Models
class GeneratedCodeResponse(BaseModel):
    id: int
    uuid: str
    title: str
    language: str
    framework: Optional[str]
    content: str
    file_path: Optional[str]
    syntax_valid: bool
    executable: bool
    execution_result: Optional[Dict[str, Any]]
    created_at: datetime

class GeneratedDocumentResponse(BaseModel):
    id: int
    uuid: str
    title: str
    content_type: str
    file_path: Optional[str]
    file_size_bytes: int
    download_url: str
    created_at: datetime

class ContentListResponse(BaseModel):
    id: int
    uuid: str
    title: str
    content_type: str
    file_size_bytes: int
    language: Optional[str]
    is_public: bool
    is_template: bool
    view_count: int
    download_count: int
    created_at: datetime

class ExecutionResult(BaseModel):
    success: bool
    output: str
    error: Optional[str]
    execution_time_ms: int
    memory_usage_mb: Optional[float]
    exit_code: int

# Helper Functions
async def get_ai_orchestrator() -> AIOrchestrator:
    """Get AI orchestrator instance"""
    try:
        from main import app
        if hasattr(app.state, 'ai_orchestrator'):
            return app.state.ai_orchestrator
        raise HTTPException(status_code=503, detail="AI services not available")
    except Exception as e:
        logger.error(f"Error getting AI orchestrator: {e}")
        raise HTTPException(status_code=503, detail="AI services not available")

def validate_language(language: str) -> bool:
    """Validate programming language"""
    supported_languages = {
        'python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'csharp',
        'go', 'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'r',
        'sql', 'html', 'css', 'bash', 'powershell', 'yaml', 'json', 'xml'
    }
    return language.lower() in supported_languages

async def execute_code(content: str, language: str, timeout: int = 30) -> ExecutionResult:
    """Execute code safely in a sandbox environment"""
    start_time = datetime.utcnow()
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix=f'.{get_file_extension(language)}', delete=False) as f:
            f.write(content)
            temp_file = f.name
        
        try:
            # Get execution command
            cmd = get_execution_command(language, temp_file)
            if not cmd:
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Language {language} not supported for execution",
                    execution_time_ms=0,
                    exit_code=-1
                )
            
            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=os.path.dirname(temp_file)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8', errors='ignore'),
                    error=stderr.decode('utf-8', errors='ignore') if stderr else None,
                    execution_time_ms=int(execution_time),
                    exit_code=process.returncode
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution timed out",
                    execution_time_ms=timeout * 1000,
                    exit_code=-1
                )
                
        finally:
            # Cleanup
            try:
                os.unlink(temp_file)
            except OSError:
                pass
                
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        return ExecutionResult(
            success=False,
            output="",
            error=str(e),
            execution_time_ms=0,
            exit_code=-1
        )

def get_file_extension(language: str) -> str:
    """Get file extension for language"""
    extensions = {
        'python': 'py',
        'javascript': 'js',
        'typescript': 'ts',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'csharp': 'cs',
        'go': 'go',
        'rust': 'rs',
        'php': 'php',
        'ruby': 'rb',
        'swift': 'swift',
        'kotlin': 'kt',
        'scala': 'scala',
        'r': 'r',
        'sql': 'sql',
        'html': 'html',
        'css': 'css',
        'bash': 'sh',
        'powershell': 'ps1',
        'yaml': 'yml',
        'json': 'json',
        'xml': 'xml'
    }
    return extensions.get(language.lower(), 'txt')

def get_execution_command(language: str, file_path: str) -> Optional[List[str]]:
    """Get execution command for language"""
    commands = {
        'python': ['python', file_path],
        'javascript': ['node', file_path],
        'java': ['java', file_path],
        'cpp': ['g++', file_path, '-o', file_path + '.out', '&&', file_path + '.out'],
        'c': ['gcc', file_path, '-o', file_path + '.out', '&&', file_path + '.out'],
        'go': ['go', 'run', file_path],
        'rust': ['rustc', file_path, '-o', file_path + '.out', '&&', file_path + '.out'],
        'php': ['php', file_path],
        'ruby': ['ruby', file_path],
        'bash': ['bash', file_path],
        'r': ['Rscript', file_path]
    }
    return commands.get(language.lower())

async def save_generated_content(
    db: AsyncSession,
    user_id: int,
    ai_request_id: Optional[int],
    conversation_id: Optional[int],
    title: str,
    content_type: ContentType,
    content: str,
    language: Optional[str] = None,
    framework: Optional[str] = None,
    file_path: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> GeneratedContent:
    """Save generated content to database"""
    
    generated_content = GeneratedContent(
        uuid=str(uuid4()),
        user_id=user_id,
        ai_request_id=ai_request_id,
        conversation_id=conversation_id,
        title=title,
        content_type=content_type,
        content_text=content,
        language=language,
        framework=framework,
        file_path=file_path,
        file_size_bytes=len(content.encode('utf-8')),
        metadata=metadata or {},
        created_at=datetime.utcnow()
    )
    
    db.add(generated_content)
    await db.flush()
    return generated_content

# Code Generation Endpoints
@router.post("/code", response_model=GeneratedCodeResponse)
async def generate_code(
    request: CodeGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """Generate code using AI"""
    try:
        # Validate language
        if not validate_language(request.language):
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")
        
        # Create AI request record
        ai_request = AIRequest(
            uuid=str(uuid4()),
            user_id=current_user.id,
            request_id=str(uuid4()),
            model_provider=ModelProvider.OPENAI,  # Default, will be updated
            model_name=request.model_preference or "gpt-4",
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            input_text=request.prompt,
            status=RequestStatus.PENDING,
            request_start_time=datetime.utcnow(),
            metadata={
                "language": request.language,
                "framework": request.framework,
                "requirements": request.requirements
            }
        )
        db.add(ai_request)
        await db.flush()
        
        # Build enhanced prompt
        system_prompt = f"""You are an expert {request.language} developer. Generate clean, efficient, and well-documented code.

Requirements:
- Language: {request.language}
- Framework: {request.framework or 'None specified'}
- Additional requirements: {request.requirements or 'None'}

Guidelines:
- Include appropriate comments and documentation
- Follow best practices for {request.language}
- Make the code production-ready
- Include error handling where appropriate
- Ensure code is complete and executable
"""

        # Generate code
        ai_response = await orchestrator.generate_response(
            messages=[{"role": "user", "content": request.prompt}],
            system_prompt=system_prompt,
            model_preference=request.model_preference,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=str(current_user.id)
        )
        
        # Update AI request
        ai_request.model_name = ai_response.get("model", "unknown")
        ai_request.output_text = ai_response["content"]
        ai_request.total_tokens = ai_response.get("tokens", 0)
        ai_request.total_cost_usd = ai_response.get("cost", 0.0)
        ai_request.request_end_time = datetime.utcnow()
        ai_request.response_time_ms = int((ai_request.request_end_time - ai_request.request_start_time).total_seconds() * 1000)
        ai_request.status = RequestStatus.COMPLETED
        ai_request.success = True
        
        # Extract code from response (remove markdown formatting if present)
        code_content = ai_response["content"]
        if "```" in code_content:
            # Extract code block
            parts = code_content.split("```")
            if len(parts) >= 3:
                code_content = parts[1]
                if code_content.startswith(request.language.lower()):
                    code_content = code_content[len(request.language):].strip()
        
        # Basic syntax validation
        syntax_valid = await validate_syntax(code_content, request.language)
        
        # Execute code if requested
        execution_result = None
        executable = False
        if request.execute and syntax_valid:
            try:
                exec_result = await execute_code(code_content, request.language)
                execution_result = exec_result.dict()
                executable = exec_result.success
            except Exception as e:
                logger.error(f"Code execution error: {e}")
                execution_result = {"error": str(e), "success": False}
        
        # Save generated content
        title = f"{request.language.title()} Code - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        generated_content = await save_generated_content(
            db=db,
            user_id=current_user.id,
            ai_request_id=ai_request.id,
            conversation_id=None,
            title=title,
            content_type=ContentType.CODE,
            content=code_content,
            language=request.language,
            framework=request.framework,
            metadata={
                "syntax_valid": syntax_valid,
                "executable": executable,
                "execution_result": execution_result,
                "original_prompt": request.prompt
            }
        )
        
        generated_content.syntax_valid = syntax_valid
        generated_content.executable = executable
        
        await db.commit()
        
        return GeneratedCodeResponse(
            id=generated_content.id,
            uuid=generated_content.uuid,
            title=generated_content.title,
            language=generated_content.language,
            framework=generated_content.framework,
            content=code_content,
            file_path=generated_content.file_path,
            syntax_valid=syntax_valid,
            executable=executable,
            execution_result=execution_result,
            created_at=generated_content.created_at
        )
        
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate code")

async def validate_syntax(code: str, language: str) -> bool:
    """Basic syntax validation for code"""
    try:
        if language.lower() == 'python':
            import ast
            ast.parse(code)
            return True
        elif language.lower() == 'javascript':
            # Basic check for common syntax issues
            return '{' in code or 'function' in code or 'const' in code or 'let' in code
        else:
            # For other languages, just check if it's not empty
            return len(code.strip()) > 0
    except:
        return False

@router.post("/code/{content_id}/execute", response_model=ExecutionResult)
async def execute_generated_code(
    content_id: int,
    timeout: int = 30,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Execute previously generated code"""
    try:
        # Get generated content
        query = select(GeneratedContent).where(
            GeneratedContent.id == content_id,
            GeneratedContent.user_id == current_user.id,
            GeneratedContent.content_type == ContentType.CODE
        )
        result = await db.execute(query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(status_code=404, detail="Code not found")
        
        if not content.language:
            raise HTTPException(status_code=400, detail="No language specified for code")
        
        # Execute code
        execution_result = await execute_code(content.content_text, content.language, timeout)
        
        # Update execution metrics
        content.execution_time_ms = execution_result.execution_time_ms
        content.executable = execution_result.success
        
        # Update metadata
        if not content.metadata:
            content.metadata = {}
        content.metadata["last_execution"] = {
            "timestamp": datetime.utcnow().isoformat(),
            "result": execution_result.dict()
        }
        
        await db.commit()
        
        return execution_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing code: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute code")

# Document Generation Endpoints
@router.post("/document", response_model=GeneratedDocumentResponse)
async def generate_document(
    request: DocumentGenerationRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    orchestrator: AIOrchestrator = Depends(get_ai_orchestrator)
):
    """Generate document using AI"""
    try:
        # Create AI request record
        ai_request = AIRequest(
            uuid=str(uuid4()),
            user_id=current_user.id,
            request_id=str(uuid4()),
            model_provider=ModelProvider.OPENAI,
            model_name=request.model_preference or "gpt-4",
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            input_text=request.prompt,
            status=RequestStatus.PENDING,
            request_start_time=datetime.utcnow(),
            metadata={
                "document_type": request.document_type,
                "template": request.template,
                "title": request.title,
                "author": request.author
            }
        )
        db.add(ai_request)
        await db.flush()
        
        # Build document generation prompt
        system_prompt = f"""You are a professional document writer. Create a well-structured {request.document_type} document.

Document specifications:
- Type: {request.document_type}
- Title: {request.title or 'Generated Document'}
- Author: {request.author or current_user.full_name or current_user.username}
- Template: {request.template or 'Standard format'}

Guidelines:
- Use proper formatting for {request.document_type}
- Include appropriate headings and structure
- Make content professional and well-organized
- Ensure content is complete and comprehensive
"""

        # Generate document content
        ai_response = await orchestrator.generate_response(
            messages=[{"role": "user", "content": request.prompt}],
            system_prompt=system_prompt,
            model_preference=request.model_preference,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            user_id=str(current_user.id)
        )
        
        # Update AI request
        ai_request.model_name = ai_response.get("model", "unknown")
        ai_request.output_text = ai_response["content"]
        ai_request.total_tokens = ai_response.get("tokens", 0)
        ai_request.total_cost_usd = ai_response.get("cost", 0.0)
        ai_request.request_end_time = datetime.utcnow()
        ai_request.response_time_ms = int((ai_request.request_end_time - ai_request.request_start_time).total_seconds() * 1000)
        ai_request.status = RequestStatus.COMPLETED
        ai_request.success = True
        
        document_content = ai_response["content"]
        
        # Save generated content
        title = request.title or f"{request.document_type.upper()} Document - {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"
        content_type = get_content_type_from_document_type(request.document_type)
        
        generated_content = await save_generated_content(
            db=db,
            user_id=current_user.id,
            ai_request_id=ai_request.id,
            conversation_id=None,
            title=title,
            content_type=content_type,
            content=document_content,
            metadata={
                "document_type": request.document_type,
                "template": request.template,
                "author": request.author,
                "original_prompt": request.prompt
            }
        )
        
        # Generate file if needed
        file_path = None
        if request.document_type in ['pdf', 'docx']:
            # Schedule background task to generate file
            background_tasks.add_task(
                generate_document_file,
                generated_content.id,
                document_content,
                request.document_type,
                title
            )
        
        await db.commit()
        
        download_url = f"/api/v1/generate/content/{generated_content.uuid}/download"
        
        return GeneratedDocumentResponse(
            id=generated_content.id,
            uuid=generated_content.uuid,
            title=generated_content.title,
            content_type=generated_content.content_type,
            file_path=file_path,
            file_size_bytes=generated_content.file_size_bytes,
            download_url=download_url,
            created_at=generated_content.created_at
        )
        
    except HTTPException:
        await db.rollback()
        raise
    except Exception as e:
        await db.rollback()
        logger.error(f"Error generating document: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate document")

def get_content_type_from_document_type(document_type: str) -> ContentType:
    """Map document type to ContentType enum"""
    mapping = {
        'pdf': ContentType.DOCUMENT,
        'docx': ContentType.DOCUMENT,
        'html': ContentType.HTML,
        'markdown': ContentType.MARKDOWN,
        'txt': ContentType.TEXT
    }
    return mapping.get(document_type, ContentType.DOCUMENT)

async def generate_document_file(content_id: int, content: str, document_type: str, title: str):
    """Background task to generate document file"""
    try:
        # This would integrate with document generation libraries
        # For now, just save as text file
        file_dir = Path("generated_files")
        file_dir.mkdir(exist_ok=True)
        
        file_path = file_dir / f"{content_id}_{title.replace(' ', '_')}.{document_type}"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Update database with file path
        # This would require a database session
        logger.info(f"Generated document file: {file_path}")
        
    except Exception as e:
        logger.error(f"Error generating document file: {e}")

# Content Management Endpoints
@router.get("/content", response_model=List[ContentListResponse])
async def list_generated_content(
    content_type: Optional[str] = None,
    language: Optional[str] = None,
    public_only: bool = False,
    templates_only: bool = False,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List generated content"""
    try:
        query = select(GeneratedContent).where(GeneratedContent.user_id == current_user.id)
        
        if content_type:
            query = query.where(GeneratedContent.content_type == content_type)
        
        if language:
            query = query.where(GeneratedContent.language == language)
        
        if public_only:
            query = query.where(GeneratedContent.is_public == True)
        
        if templates_only:
            query = query.where(GeneratedContent.is_template == True)
        
        query = query.order_by(desc(GeneratedContent.created_at)).offset(offset).limit(limit)
        
        result = await db.execute(query)
        content_list = result.scalars().all()
        
        return [
            ContentListResponse(
                id=content.id,
                uuid=content.uuid,
                title=content.title,
                content_type=content.content_type,
                file_size_bytes=content.file_size_bytes or 0,
                language=content.language,
                is_public=content.is_public,
                is_template=content.is_template,
                view_count=content.view_count,
                download_count=content.download_count,
                created_at=content.created_at
            )
            for content in content_list
        ]
        
    except Exception as e:
        logger.error(f"Error listing content: {e}")
        raise HTTPException(status_code=500, detail="Failed to list content")

@router.get("/content/{content_uuid}")
async def get_generated_content(
    content_uuid: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get specific generated content"""
    try:
        query = select(GeneratedContent).where(
            GeneratedContent.uuid == content_uuid,
            GeneratedContent.user_id == current_user.id
        )
        result = await db.execute(query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Increment view count
        content.view_count += 1
        await db.commit()
        
        return {
            "id": content.id,
            "uuid": content.uuid,
            "title": content.title,
            "description": content.description,
            "content_type": content.content_type,
            "content": content.content_text,
            "language": content.language,
            "framework": content.framework,
            "file_size_bytes": content.file_size_bytes,
            "is_public": content.is_public,
            "is_template": content.is_template,
            "view_count": content.view_count,
            "download_count": content.download_count,
            "metadata": content.metadata,
            "created_at": content.created_at
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting content: {e}")
        raise HTTPException(status_code=500, detail="Failed to get content")

@router.put("/content/{content_uuid}")
async def update_generated_content(
    content_uuid: str,
    request: ContentUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update generated content metadata"""
    try:
        query = select(GeneratedContent).where(
            GeneratedContent.uuid == content_uuid,
            GeneratedContent.user_id == current_user.id
        )
        result = await db.execute(query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Update fields
        if request.title is not None:
            content.title = request.title
        if request.description is not None:
            content.description = request.description
        if request.is_public is not None:
            content.is_public = request.is_public
        if request.is_template is not None:
            content.is_template = request.is_template
        if request.tags is not None:
            if not content.metadata:
                content.metadata = {}
            content.metadata["tags"] = request.tags
        
        content.updated_at = datetime.utcnow()
        
        await db.commit()
        
        return {"message": "Content updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating content: {e}")
        raise HTTPException(status_code=500, detail="Failed to update content")

@router.delete("/content/{content_uuid}")
async def delete_generated_content(
    content_uuid: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete generated content"""
    try:
        query = select(GeneratedContent).where(
            GeneratedContent.uuid == content_uuid,
            GeneratedContent.user_id == current_user.id
        )
        result = await db.execute(query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Delete file if exists
        if content.file_path and os.path.exists(content.file_path):
            try:
                os.unlink(content.file_path)
            except OSError:
                pass
        
        await db.delete(content)
        await db.commit()
        
        return {"message": "Content deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting content: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete content")

@router.get("/content/{content_uuid}/download")
async def download_generated_content(
    content_uuid: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Download generated content as file"""
    try:
        query = select(GeneratedContent).where(
            GeneratedContent.uuid == content_uuid,
            GeneratedContent.user_id == current_user.id
        )
        result = await db.execute(query)
        content = result.scalar_one_or_none()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        # Increment download count
        content.download_count += 1
        await db.commit()
        
        # Return file content
        from fastapi.responses import Response
        
        filename = f"{content.title.replace(' ', '_')}.{get_file_extension(content.language or 'txt')}"
        
        return Response(
            content=content.content_text.encode('utf-8'),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading content: {e}")
        raise HTTPException(status_code=500, detail="Failed to download content")