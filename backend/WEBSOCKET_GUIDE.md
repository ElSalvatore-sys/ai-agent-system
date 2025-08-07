# WebSocket Chat System Guide

## Overview

The AI Agent System includes a comprehensive WebSocket chat system that provides real-time bidirectional communication with advanced features including room-based chat, typing indicators, file uploads, message queuing, delivery confirmation, and sophisticated connection management.

## Features

### Core WebSocket Features
- **Real-time bidirectional communication** - Instant messaging between clients and AI
- **Multiple concurrent AI conversations** - Support for multiple active chat sessions
- **Connection management** - Automatic reconnection, heartbeat monitoring, and cleanup
- **Message queuing** - Queue messages for offline users
- **Delivery confirmation** - Confirm message delivery and read receipts

### Room-Based Chat
- **Private rooms** - One-on-one conversations
- **Shared rooms** - Multiple users in a single room
- **Group rooms** - Organized group conversations with membership control
- **Room management** - Create, join, leave, and delete rooms

### Advanced Features
- **Typing indicators** - Real-time typing status for users and AI
- **File uploads** - Chunked file uploads through WebSocket with virus scanning
- **Authentication** - JWT-based WebSocket authentication
- **Rate limiting** - Per-connection rate limiting with sliding window algorithm
- **Error handling** - Comprehensive error handling and recovery

## Quick Start

### 1. Authentication

First, obtain a WebSocket token:

```bash
curl -X POST "http://localhost:8000/api/v1/ws/token" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

Response:
```json
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "expires_in": 86400
}
```

### 2. Connect to WebSocket

Connect to the WebSocket endpoint with authentication:

```javascript
const wsUrl = `ws://localhost:8000/api/v1/ws/ws?token=${wsToken}`;
const socket = new WebSocket(wsUrl);

socket.onopen = function(event) {
    console.log('Connected to WebSocket');
};

socket.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};
```

### 3. Send Messages

Send a chat message:

```javascript
const chatMessage = {
    type: "chat",
    data: {
        message: "Hello, AI!",
        conversation_id: null, // Will create new conversation
        model_preference: "gpt-4",
        temperature: 0.7
    },
    room_id: "room-123", // Optional: broadcast to room
    message_id: "msg-" + Date.now()
};

socket.send(JSON.stringify(chatMessage));
```

## WebSocket Message Types

### Connection Messages

#### Connect Confirmation
```json
{
    "type": "connect",
    "data": {
        "connection_id": "uuid-string",
        "status": "connected",
        "server_time": "2024-01-01T12:00:00Z"
    }
}
```

#### Heartbeat
```json
{
    "type": "system",
    "data": {
        "heartbeat": true,
        "server_time": "2024-01-01T12:00:00Z"
    }
}
```

### Chat Messages

#### Send Chat Message
```json
{
    "type": "chat",
    "data": {
        "message": "Your message here",
        "conversation_id": 123,
        "model_preference": "gpt-4",
        "system_prompt": "You are a helpful assistant",
        "temperature": 0.7,
        "max_tokens": 2048
    },
    "room_id": "room-123",
    "message_id": "unique-id"
}
```

#### Receive Chat Response (Streaming)
```json
{
    "type": "chat",
    "data": {
        "role": "assistant",
        "content": "Partial response...",
        "streaming": true,
        "conversation_id": 123,
        "timestamp": "2024-01-01T12:00:00Z"
    }
}
```

#### Chat Completion
```json
{
    "type": "chat",
    "data": {
        "role": "assistant",
        "content": "",
        "streaming": false,
        "done": true,
        "message_id": 456,
        "conversation_id": 123,
        "tokens_used": 150,
        "model_used": "gpt-4",
        "cost_usd": 0.003,
        "response_time": 2.5
    }
}
```

### Room Management

#### Join Room
```json
{
    "type": "join_room",
    "data": {
        "room_id": "room-123",
        "room_name": "My Chat Room",
        "room_type": "private"
    }
}
```

#### Leave Room
```json
{
    "type": "leave_room",
    "data": {
        "room_id": "room-123"
    }
}
```

#### Room Joined Confirmation
```json
{
    "type": "join_room",
    "data": {
        "room_id": "room-123",
        "room_name": "My Chat Room",
        "room_type": "private",
        "participants": ["user1", "user2"],
        "joined_at": "2024-01-01T12:00:00Z"
    }
}
```

### Typing Indicators

#### Send Typing Status
```json
{
    "type": "typing",
    "data": {
        "is_typing": true,
        "room_id": "room-123"
    }
}
```

#### Receive Typing Status
```json
{
    "type": "typing",
    "data": {
        "user_id": "user123",
        "username": "john_doe",
        "is_typing": true,
        "room_id": "room-123"
    }
}
```

### File Uploads

#### Initialize Upload
```json
{
    "type": "file_upload",
    "data": {
        "upload_type": "init",
        "file_name": "document.pdf",
        "file_size": 1048576,
        "mime_type": "application/pdf"
    },
    "room_id": "room-123"
}
```

#### Upload Chunk
```json
{
    "type": "file_upload",
    "data": {
        "upload_type": "chunk",
        "file_id": "file-uuid",
        "chunk_index": 0,
        "total_chunks": 10,
        "chunk_data": "base64-encoded-data"
    }
}
```

#### Finalize Upload
```json
{
    "type": "file_upload",
    "data": {
        "upload_type": "finalize",
        "file_id": "file-uuid",
        "checksum": "sha256-hash"
    }
}
```

### System Messages

#### Delivery Confirmation
```json
{
    "type": "delivery_confirmation",
    "data": {
        "message_id": "msg-123",
        "status": "delivered",
        "timestamp": "2024-01-01T12:00:00Z"
    }
}
```

#### Error Messages
```json
{
    "type": "error",
    "data": {
        "error": "Rate limit exceeded",
        "code": "RATE_LIMIT",
        "retry_after": 60
    }
}
```

## Room Types

### Private Rooms
- Default room type
- Accessible to any authenticated user
- Suitable for personal AI conversations

### Shared Rooms
- Multi-user rooms
- All authenticated users can join
- Good for public discussions

### Group Rooms
- Membership-controlled rooms
- Require specific permissions to join
- Ideal for team or organization chats

## File Upload System

### Supported File Types
- **Images**: jpg, jpeg, png, gif, bmp, webp, svg
- **Documents**: pdf, doc, docx, txt, rtf, odt
- **Spreadsheets**: xls, xlsx, csv, ods
- **Presentations**: ppt, pptx, odp
- **Archives**: zip, rar, 7z, tar, gz
- **Code**: py, js, html, css, json, xml, md
- **Audio**: mp3, wav, ogg, m4a, flac
- **Video**: mp4, avi, mov, wmv, webm (small files)

### Upload Limits
- **Maximum file size**: 50MB
- **Maximum chunk size**: 1MB
- **Virus scanning**: Enabled in production
- **Checksum verification**: SHA-256

### Upload Process
1. Initialize upload with file metadata
2. Upload file in chunks (base64 encoded)
3. Finalize upload with checksum verification
4. File is processed and made available

## API Endpoints

### HTTP Endpoints

#### Get WebSocket Stats
```bash
GET /api/v1/ws/stats
Authorization: Bearer <admin-token>
```

#### Get Room Information
```bash
GET /api/v1/ws/rooms/{room_id}
Authorization: Bearer <token>
```

#### Generate WebSocket Token
```bash
POST /api/v1/ws/token
Authorization: Bearer <token>
```

#### Create Room
```bash
POST /api/v1/ws/rooms
Content-Type: application/json
Authorization: Bearer <token>

{
    "room_id": "my-room",
    "room_name": "My Chat Room",
    "room_type": "private"
}
```

### WebSocket Endpoints

#### Main Chat Endpoint
```
ws://localhost:8000/api/v1/ws/ws?token=<ws-token>&room_id=<room>&room_type=<type>
```

Query Parameters:
- `token` (required): WebSocket authentication token
- `room_id` (optional): Auto-join room on connection
- `room_type` (optional): Room type (private, shared, group)
- `room_name` (optional): Room display name

## Client Implementation Examples

### JavaScript Client

```javascript
class WebSocketChatClient {
    constructor(token, roomId = null) {
        this.token = token;
        this.roomId = roomId;
        this.socket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
    }

    connect() {
        const params = new URLSearchParams({ token: this.token });
        if (this.roomId) params.append('room_id', this.roomId);
        
        const wsUrl = `ws://localhost:8000/api/v1/ws/ws?${params}`;
        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = this.onOpen.bind(this);
        this.socket.onmessage = this.onMessage.bind(this);
        this.socket.onclose = this.onClose.bind(this);
        this.socket.onerror = this.onError.bind(this);
    }

    onOpen(event) {
        console.log('Connected to WebSocket');
        this.reconnectAttempts = 0;
    }

    onMessage(event) {
        const message = JSON.parse(event.data);
        this.handleMessage(message);
    }

    onClose(event) {
        console.log('WebSocket connection closed');
        this.attemptReconnect();
    }

    onError(event) {
        console.error('WebSocket error:', event);
    }

    handleMessage(message) {
        switch (message.type) {
            case 'chat':
                this.handleChatMessage(message);
                break;
            case 'typing':
                this.handleTypingIndicator(message);
                break;
            case 'file_upload':
                this.handleFileUpload(message);
                break;
            case 'error':
                this.handleError(message);
                break;
        }
    }

    sendChatMessage(text, conversationId = null) {
        const message = {
            type: 'chat',
            data: {
                message: text,
                conversation_id: conversationId,
                model_preference: 'gpt-4'
            },
            room_id: this.roomId,
            message_id: 'msg-' + Date.now()
        };
        this.send(message);
    }

    sendTypingIndicator(isTyping) {
        const message = {
            type: 'typing',
            data: {
                is_typing: isTyping,
                room_id: this.roomId
            }
        };
        this.send(message);
    }

    send(message) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(message));
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = Math.pow(2, this.reconnectAttempts) * 1000; // Exponential backoff
            setTimeout(() => this.connect(), delay);
        }
    }
}

// Usage
const client = new WebSocketChatClient('your-ws-token', 'room-123');
client.connect();
```

### Python Client

```python
import asyncio
import json
import websockets
import logging

class WebSocketChatClient:
    def __init__(self, token, room_id=None):
        self.token = token
        self.room_id = room_id
        self.websocket = None
        self.running = False

    async def connect(self):
        uri = f"ws://localhost:8000/api/v1/ws/ws?token={self.token}"
        if self.room_id:
            uri += f"&room_id={self.room_id}"
        
        try:
            self.websocket = await websockets.connect(uri)
            self.running = True
            logging.info("Connected to WebSocket")
            
            # Start message handler
            await self.handle_messages()
            
        except Exception as e:
            logging.error(f"Connection error: {e}")

    async def handle_messages(self):
        try:
            async for message in self.websocket:
                data = json.loads(message)
                await self.process_message(data)
        except websockets.exceptions.ConnectionClosed:
            logging.info("WebSocket connection closed")
        except Exception as e:
            logging.error(f"Message handling error: {e}")

    async def process_message(self, message):
        msg_type = message.get('type')
        
        if msg_type == 'chat':
            await self.handle_chat_message(message)
        elif msg_type == 'typing':
            await self.handle_typing_indicator(message)
        elif msg_type == 'error':
            logging.error(f"WebSocket error: {message['data']}")

    async def send_chat_message(self, text, conversation_id=None):
        message = {
            'type': 'chat',
            'data': {
                'message': text,
                'conversation_id': conversation_id,
                'model_preference': 'gpt-4'
            },
            'room_id': self.room_id,
            'message_id': f'msg-{asyncio.get_event_loop().time()}'
        }
        await self.send(message)

    async def send(self, message):
        if self.websocket and not self.websocket.closed:
            await self.websocket.send(json.dumps(message))

    async def close(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()

# Usage
async def main():
    client = WebSocketChatClient('your-ws-token', 'room-123')
    await client.connect()

asyncio.run(main())
```

## Configuration

### Environment Variables

```env
# WebSocket Configuration
WEBSOCKET_MAX_CONNECTIONS=1000
WEBSOCKET_HEARTBEAT_INTERVAL=30
WEBSOCKET_CLEANUP_INTERVAL=300

# File Upload Configuration
MAX_FILE_SIZE=52428800  # 50MB
MAX_CHUNK_SIZE=1048576  # 1MB
VIRUS_SCAN_ENABLED=false
UPLOAD_DIR=uploads
TEMP_DIR=uploads/temp

# Rate Limiting
WS_RATE_LIMIT_REQUESTS=100
WS_RATE_LIMIT_WINDOW=60
```

### Security Considerations

1. **Authentication**: Always use JWT tokens for WebSocket authentication
2. **Rate Limiting**: Implement rate limiting to prevent abuse
3. **File Upload Security**: Enable virus scanning in production
4. **Input Validation**: Validate all incoming messages
5. **Connection Limits**: Set maximum connections per user
6. **CORS**: Configure CORS properly for cross-origin WebSocket connections

## Monitoring and Debugging

### WebSocket Statistics

Get real-time statistics about WebSocket connections:

```bash
curl -H "Authorization: Bearer <admin-token>" \
  http://localhost:8000/api/v1/ws/stats
```

### Logging

WebSocket activities are logged with structured logging:

```python
# Enable debug logging for WebSocket components
import logging
logging.getLogger('app.services.websocket_manager').setLevel(logging.DEBUG)
logging.getLogger('app.services.websocket_auth').setLevel(logging.DEBUG)
```

### Health Checks

WebSocket health is included in the main health check endpoint:

```bash
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if WebSocket token is valid
   - Verify server is running on correct port
   - Check firewall settings

2. **Authentication Failed**
   - Ensure JWT token is not expired
   - Verify token has correct permissions
   - Check token format (should be valid JWT)

3. **Messages Not Received**
   - Check WebSocket connection status
   - Verify message format is correct
   - Check rate limiting status

4. **File Upload Fails**
   - Verify file size is within limits
   - Check file type is allowed
   - Ensure sufficient disk space

### Debug Mode

Enable debug mode for detailed logging:

```python
# In your environment
DEBUG=true
LOG_LEVEL=debug
```

This comprehensive WebSocket system provides enterprise-grade real-time communication capabilities with full feature parity to modern chat applications while integrating seamlessly with AI models for intelligent conversations.