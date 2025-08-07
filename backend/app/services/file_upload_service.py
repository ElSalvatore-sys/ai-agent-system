"""
File Upload Service for WebSocket
Handles file uploads through WebSocket connections with support for
chunked uploads, file validation, virus scanning, and cloud storage
"""
import asyncio
import base64
import hashlib
import logging
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, BinaryIO
from uuid import uuid4

import aiofiles
from pydantic import BaseModel, Field

from app.core.config import settings
from app.services.websocket_manager import websocket_manager, WSMessage, MessageType

logger = logging.getLogger(__name__)

class FileUploadConfig:
    """File upload configuration"""
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_CHUNK_SIZE = 1024 * 1024  # 1MB
    ALLOWED_EXTENSIONS = {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
        # Documents
        '.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt',
        # Spreadsheets
        '.xls', '.xlsx', '.csv', '.ods',
        # Presentations
        '.ppt', '.pptx', '.odp',
        # Archives
        '.zip', '.rar', '.7z', '.tar', '.gz',
        # Code
        '.py', '.js', '.html', '.css', '.json', '.xml', '.md',
        # Audio
        '.mp3', '.wav', '.ogg', '.m4a', '.flac',
        # Video (small files only)
        '.mp4', '.avi', '.mov', '.wmv', '.webm'
    }
    VIRUS_SCAN_ENABLED = False  # Set to True in production with ClamAV
    UPLOAD_DIR = Path("uploads")
    TEMP_DIR = Path("uploads/temp")

class FileMetadata(BaseModel):
    """File metadata model"""
    file_id: str
    original_name: str
    file_size: int
    mime_type: str
    extension: str
    upload_started: datetime
    upload_completed: Optional[datetime] = None
    checksum: Optional[str] = None
    is_virus_scanned: bool = False
    is_safe: bool = True
    user_id: str
    connection_id: str
    room_id: Optional[str] = None

class FileUploadChunk(BaseModel):
    """File upload chunk model"""
    file_id: str
    chunk_index: int
    total_chunks: int
    chunk_data: str  # Base64 encoded
    is_final: bool = False

class FileUploadStatus(BaseModel):
    """File upload status model"""
    file_id: str
    status: str  # uploading, processing, completed, failed
    progress: float  # 0.0 to 1.0
    uploaded_chunks: int
    total_chunks: int
    error_message: Optional[str] = None

class WebSocketFileUploadService:
    """WebSocket file upload service"""
    
    def __init__(self):
        self.active_uploads: Dict[str, FileMetadata] = {}
        self.upload_chunks: Dict[str, Dict[int, bytes]] = {}  # file_id -> {chunk_index: data}
        self.upload_locks: Dict[str, asyncio.Lock] = {}
        
        # Ensure upload directories exist
        FileUploadConfig.UPLOAD_DIR.mkdir(exist_ok=True)
        FileUploadConfig.TEMP_DIR.mkdir(exist_ok=True)

    async def handle_file_upload_message(self, connection_id: str, ws_message: WSMessage) -> bool:
        """Handle file upload WebSocket message"""
        try:
            message_type = ws_message.data.get("upload_type", "chunk")
            
            if message_type == "init":
                return await self._handle_upload_init(connection_id, ws_message)
            elif message_type == "chunk":
                return await self._handle_upload_chunk(connection_id, ws_message)
            elif message_type == "finalize":
                return await self._handle_upload_finalize(connection_id, ws_message)
            elif message_type == "cancel":
                return await self._handle_upload_cancel(connection_id, ws_message)
            else:
                await self._send_upload_error(connection_id, "Unknown upload type", ws_message.data.get("file_id"))
                return False
                
        except Exception as e:
            logger.error(f"Error handling file upload: {e}")
            await self._send_upload_error(connection_id, f"Upload error: {str(e)}", ws_message.data.get("file_id"))
            return False

    async def _handle_upload_init(self, connection_id: str, ws_message: WSMessage) -> bool:
        """Initialize file upload"""
        try:
            data = ws_message.data
            file_name = data.get("file_name", "unknown")
            file_size = data.get("file_size", 0)
            mime_type = data.get("mime_type", "application/octet-stream")
            user_id = ws_message.user_id
            room_id = ws_message.room_id
            
            # Validate file
            validation_result = await self._validate_file(file_name, file_size, mime_type)
            if not validation_result["valid"]:
                await self._send_upload_error(connection_id, validation_result["error"])
                return False
            
            # Generate file ID
            file_id = str(uuid4())
            
            # Create file metadata
            file_metadata = FileMetadata(
                file_id=file_id,
                original_name=file_name,
                file_size=file_size,
                mime_type=mime_type,
                extension=Path(file_name).suffix.lower(),
                upload_started=datetime.utcnow(),
                user_id=user_id,
                connection_id=connection_id,
                room_id=room_id
            )
            
            # Store metadata
            self.active_uploads[file_id] = file_metadata
            self.upload_chunks[file_id] = {}
            self.upload_locks[file_id] = asyncio.Lock()
            
            # Send initialization response
            await websocket_manager._send_to_connection(connection_id, WSMessage(
                type=MessageType.FILE_UPLOAD,
                data={
                    "upload_type": "init_response",
                    "file_id": file_id,
                    "status": "initialized",
                    "max_chunk_size": FileUploadConfig.MAX_CHUNK_SIZE
                }
            ))
            
            logger.info(f"File upload initialized: {file_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing upload: {e}")
            return False

    async def _handle_upload_chunk(self, connection_id: str, ws_message: WSMessage) -> bool:
        """Handle file upload chunk"""
        try:
            data = ws_message.data
            file_id = data.get("file_id")
            chunk_index = data.get("chunk_index", 0)
            total_chunks = data.get("total_chunks", 1)
            chunk_data_b64 = data.get("chunk_data", "")
            
            if not file_id or file_id not in self.active_uploads:
                await self._send_upload_error(connection_id, "Invalid file ID", file_id)
                return False
            
            # Get upload lock
            async with self.upload_locks[file_id]:
                # Decode chunk data
                try:
                    chunk_data = base64.b64decode(chunk_data_b64)
                except Exception as e:
                    await self._send_upload_error(connection_id, f"Invalid chunk data: {e}", file_id)
                    return False
                
                # Validate chunk size
                if len(chunk_data) > FileUploadConfig.MAX_CHUNK_SIZE:
                    await self._send_upload_error(connection_id, "Chunk too large", file_id)
                    return False
                
                # Store chunk
                self.upload_chunks[file_id][chunk_index] = chunk_data
                
                # Calculate progress
                uploaded_chunks = len(self.upload_chunks[file_id])
                progress = uploaded_chunks / total_chunks
                
                # Send progress update
                await websocket_manager._send_to_connection(connection_id, WSMessage(
                    type=MessageType.FILE_UPLOAD,
                    data={
                        "upload_type": "progress",
                        "file_id": file_id,
                        "status": "uploading",
                        "progress": progress,
                        "uploaded_chunks": uploaded_chunks,
                        "total_chunks": total_chunks
                    }
                ))
                
                # Broadcast progress to room if applicable
                if ws_message.room_id:
                    await websocket_manager._broadcast_to_room(
                        ws_message.room_id,
                        WSMessage(
                            type=MessageType.FILE_UPLOAD,
                            data={
                                "upload_type": "progress",
                                "file_id": file_id,
                                "user_id": ws_message.user_id,
                                "file_name": self.active_uploads[file_id].original_name,
                                "progress": progress
                            },
                            room_id=ws_message.room_id
                        ),
                        exclude_user=ws_message.user_id
                    )
                
                logger.debug(f"Received chunk {chunk_index}/{total_chunks} for file {file_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error handling upload chunk: {e}")
            return False

    async def _handle_upload_finalize(self, connection_id: str, ws_message: WSMessage) -> bool:
        """Finalize file upload"""
        try:
            data = ws_message.data
            file_id = data.get("file_id")
            expected_checksum = data.get("checksum")
            
            if not file_id or file_id not in self.active_uploads:
                await self._send_upload_error(connection_id, "Invalid file ID", file_id)
                return False
            
            async with self.upload_locks[file_id]:
                file_metadata = self.active_uploads[file_id]
                chunks = self.upload_chunks[file_id]
                
                # Verify all chunks received
                expected_chunks = -(-file_metadata.file_size // FileUploadConfig.MAX_CHUNK_SIZE)  # Ceiling division
                if len(chunks) != expected_chunks:
                    await self._send_upload_error(
                        connection_id, 
                        f"Missing chunks: expected {expected_chunks}, got {len(chunks)}", 
                        file_id
                    )
                    return False
                
                # Reassemble file
                temp_file_path = FileUploadConfig.TEMP_DIR / f"{file_id}.tmp"
                final_file_path = FileUploadConfig.UPLOAD_DIR / f"{file_id}{file_metadata.extension}"
                
                try:
                    async with aiofiles.open(temp_file_path, 'wb') as f:
                        for chunk_index in sorted(chunks.keys()):
                            await f.write(chunks[chunk_index])
                    
                    # Verify file size
                    actual_size = temp_file_path.stat().st_size
                    if actual_size != file_metadata.file_size:
                        await self._send_upload_error(
                            connection_id,
                            f"File size mismatch: expected {file_metadata.file_size}, got {actual_size}",
                            file_id
                        )
                        temp_file_path.unlink(missing_ok=True)
                        return False
                    
                    # Calculate checksum
                    actual_checksum = await self._calculate_file_checksum(temp_file_path)
                    if expected_checksum and actual_checksum != expected_checksum:
                        await self._send_upload_error(
                            connection_id,
                            "Checksum verification failed",
                            file_id
                        )
                        temp_file_path.unlink(missing_ok=True)
                        return False
                    
                    # Virus scan (if enabled)
                    if FileUploadConfig.VIRUS_SCAN_ENABLED:
                        is_safe = await self._virus_scan_file(temp_file_path)
                        if not is_safe:
                            await self._send_upload_error(
                                connection_id,
                                "File failed virus scan",
                                file_id
                            )
                            temp_file_path.unlink(missing_ok=True)
                            return False
                        file_metadata.is_virus_scanned = True
                    
                    # Move to final location
                    temp_file_path.rename(final_file_path)
                    
                    # Update metadata
                    file_metadata.upload_completed = datetime.utcnow()
                    file_metadata.checksum = actual_checksum
                    
                    # Send completion message
                    await websocket_manager._send_to_connection(connection_id, WSMessage(
                        type=MessageType.FILE_UPLOAD,
                        data={
                            "upload_type": "completed",
                            "file_id": file_id,
                            "status": "completed",
                            "file_url": f"/api/v1/files/{file_id}",
                            "file_name": file_metadata.original_name,
                            "file_size": file_metadata.file_size,
                            "mime_type": file_metadata.mime_type,
                            "checksum": actual_checksum,
                            "upload_time": (file_metadata.upload_completed - file_metadata.upload_started).total_seconds()
                        }
                    ))
                    
                    # Broadcast completion to room
                    if ws_message.room_id:
                        await websocket_manager._broadcast_to_room(
                            ws_message.room_id,
                            WSMessage(
                                type=MessageType.FILE_UPLOAD,
                                data={
                                    "upload_type": "completed",
                                    "file_id": file_id,
                                    "user_id": ws_message.user_id,
                                    "file_name": file_metadata.original_name,
                                    "file_size": file_metadata.file_size,
                                    "mime_type": file_metadata.mime_type,
                                    "file_url": f"/api/v1/files/{file_id}"
                                },
                                room_id=ws_message.room_id
                            ),
                            exclude_user=ws_message.user_id
                        )
                    
                    # Cleanup
                    del self.upload_chunks[file_id]
                    del self.upload_locks[file_id]
                    
                    logger.info(f"File upload completed: {file_id}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error finalizing upload: {e}")
                    temp_file_path.unlink(missing_ok=True)
                    await self._send_upload_error(connection_id, f"Finalization error: {str(e)}", file_id)
                    return False
                
        except Exception as e:
            logger.error(f"Error in upload finalization: {e}")
            return False

    async def _handle_upload_cancel(self, connection_id: str, ws_message: WSMessage) -> bool:
        """Cancel file upload"""
        try:
            file_id = ws_message.data.get("file_id")
            
            if file_id and file_id in self.active_uploads:
                # Cleanup
                self.active_uploads.pop(file_id, None)
                self.upload_chunks.pop(file_id, None)
                self.upload_locks.pop(file_id, None)
                
                # Remove temp file if exists
                temp_file_path = FileUploadConfig.TEMP_DIR / f"{file_id}.tmp"
                temp_file_path.unlink(missing_ok=True)
                
                # Send cancellation confirmation
                await websocket_manager._send_to_connection(connection_id, WSMessage(
                    type=MessageType.FILE_UPLOAD,
                    data={
                        "upload_type": "cancelled",
                        "file_id": file_id,
                        "status": "cancelled"
                    }
                ))
                
                logger.info(f"File upload cancelled: {file_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling upload: {e}")
            return False

    async def _validate_file(self, file_name: str, file_size: int, mime_type: str) -> Dict[str, Any]:
        """Validate file upload"""
        try:
            # Check file size
            if file_size > FileUploadConfig.MAX_FILE_SIZE:
                return {
                    "valid": False,
                    "error": f"File too large. Maximum size: {FileUploadConfig.MAX_FILE_SIZE // (1024*1024)}MB"
                }
            
            if file_size <= 0:
                return {"valid": False, "error": "Invalid file size"}
            
            # Check file extension
            extension = Path(file_name).suffix.lower()
            if extension not in FileUploadConfig.ALLOWED_EXTENSIONS:
                return {
                    "valid": False,
                    "error": f"File type not allowed. Allowed types: {', '.join(FileUploadConfig.ALLOWED_EXTENSIONS)}"
                }
            
            # Check MIME type (basic validation)
            expected_mime = mimetypes.guess_type(file_name)[0]
            if expected_mime and not mime_type.startswith(expected_mime.split('/')[0]):
                logger.warning(f"MIME type mismatch: expected {expected_mime}, got {mime_type}")
            
            return {"valid": True, "error": None}
            
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {str(e)}"}

    async def _calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA-256 checksum of file"""
        hash_sha256 = hashlib.sha256()
        
        async with aiofiles.open(file_path, 'rb') as f:
            while chunk := await f.read(8192):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()

    async def _virus_scan_file(self, file_path: Path) -> bool:
        """Scan file for viruses (placeholder implementation)"""
        try:
            # This is a placeholder - in production, integrate with ClamAV or similar
            # For now, just check file size and basic patterns
            
            # Basic checks
            if file_path.stat().st_size == 0:
                return False
            
            # Check for suspicious file patterns (very basic)
            async with aiofiles.open(file_path, 'rb') as f:
                header = await f.read(1024)
                
                # Check for some basic malware signatures (this is not comprehensive)
                suspicious_patterns = [
                    b'This program cannot be run in DOS mode',
                    b'CreateFileA',
                    b'GetProcAddress',
                    b'LoadLibraryA'
                ]
                
                for pattern in suspicious_patterns:
                    if pattern in header:
                        logger.warning(f"Suspicious pattern found in file: {file_path}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during virus scan: {e}")
            return False

    async def _send_upload_error(self, connection_id: str, error_message: str, file_id: str = None):
        """Send upload error message"""
        await websocket_manager._send_to_connection(connection_id, WSMessage(
            type=MessageType.FILE_UPLOAD,
            data={
                "upload_type": "error",
                "file_id": file_id,
                "status": "failed",
                "error": error_message
            }
        ))

    def get_file_metadata(self, file_id: str) -> Optional[FileMetadata]:
        """Get file metadata"""
        return self.active_uploads.get(file_id)

    def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics"""
        return {
            "active_uploads": len(self.active_uploads),
            "total_chunks": sum(len(chunks) for chunks in self.upload_chunks.values()),
            "upload_dir_size": sum(f.stat().st_size for f in FileUploadConfig.UPLOAD_DIR.glob('*') if f.is_file()),
            "temp_dir_size": sum(f.stat().st_size for f in FileUploadConfig.TEMP_DIR.glob('*') if f.is_file())
        }

    async def cleanup_expired_uploads(self, max_age_hours: int = 24):
        """Cleanup expired incomplete uploads"""
        try:
            cutoff_time = datetime.utcnow().replace(hour=datetime.utcnow().hour - max_age_hours)
            
            expired_uploads = [
                file_id for file_id, metadata in self.active_uploads.items()
                if metadata.upload_started < cutoff_time and not metadata.upload_completed
            ]
            
            for file_id in expired_uploads:
                # Cleanup resources
                self.active_uploads.pop(file_id, None)
                self.upload_chunks.pop(file_id, None)
                self.upload_locks.pop(file_id, None)
                
                # Remove temp file
                temp_file_path = FileUploadConfig.TEMP_DIR / f"{file_id}.tmp"
                temp_file_path.unlink(missing_ok=True)
                
                logger.info(f"Cleaned up expired upload: {file_id}")
            
            return len(expired_uploads)
            
        except Exception as e:
            logger.error(f"Error during upload cleanup: {e}")
            return 0

# Global file upload service
file_upload_service = WebSocketFileUploadService()