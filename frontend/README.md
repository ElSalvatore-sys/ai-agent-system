# Modern React TypeScript AI Chat Frontend

A modern React TypeScript frontend application with essential components for AI chat interactions, built with the latest technologies and best practices.

## 🚀 Technology Stack

- **⚛️ React 18** - Latest React with concurrent features
- **📘 TypeScript** - Full type safety with strict mode
- **⚡ Vite** - Lightning fast build tool and dev server
- **🎨 Tailwind CSS** - Utility-first CSS framework
- **🔄 React Query** - Powerful data synchronization
- **🧭 React Router** - Client-side routing
- **🔌 Socket.io** - Real-time communication
- **🪝 Custom Hooks** - Reusable stateful logic
- **🛡️ Error Boundaries** - Comprehensive error handling
- **🍞 Toast Notifications** - User feedback system

## 🏗️ Essential Components Created

### 1. AI Model Selector
- **Dropdown selection** between o3, Claude, Gemini models
- **Cost indicators** showing price per 1K tokens
- **Performance metrics** display (response time, token limits)
- **Auto-selection** based on task type
- **Visual indicators** for model status

**Features:**
- Real-time cost calculations
- Performance comparison
- Quick task-based selection
- Model availability status

### 2. Enhanced Chat Interface
- **Real-time messaging** with WebSocket integration
- **File upload capability** (PDFs, images, documents)
- **Dark/light theme support** with toggle
- **Export conversations** (TXT, JSON, PDF coming soon)
- **Typing indicators** and loading states
- **Drag & drop file support**

**Features:**
- Theme persistence
- File attachment previews
- Message timestamps
- Auto-scroll to latest messages
- Export functionality

### 3. Code Viewer Component
- **Syntax highlighting** for multiple languages
- **Copy to clipboard** functionality
- **Download individual files** or all files
- **Live preview** for web applications
- **Desktop/mobile preview modes**
- **File management** with sidebar navigation

**Features:**
- Language detection
- File size display
- Search within files
- Preview in modal
- External link opening

### 4. Enhanced Dashboard
- **Real-time cost tracking** with charts
- **Recent conversations** list with metrics
- **Quick action buttons** (Generate Code, Create PDF, Deploy App)
- **System status indicators** for all services
- **Auto-refresh capabilities** with manual override
- **Time range selection** (24h, 7d, 30d)

**Features:**
- Interactive cost charts
- System health monitoring
- Performance metrics
- Conversation history
- Quick navigation to actions

## 📁 Project Structure

```
src/
├── components/
│   ├── ui/                    # Reusable UI components
│   │   ├── Button.tsx         # Button with variants
│   │   ├── Input.tsx          # Form input component
│   │   ├── Modal.tsx          # Modal dialog
│   │   ├── Card.tsx           # Card layouts
│   │   ├── Avatar.tsx         # User avatars
│   │   ├── Loading.tsx        # Loading states
│   │   ├── ErrorBoundary.tsx  # Error handling
│   │   ├── Toast.tsx          # Notification system
│   │   └── index.ts           # Component exports
│   ├── features/              # Feature-specific components
│   │   ├── AIModelSelector.tsx      # AI model selection
│   │   ├── EnhancedChatInterface.tsx # Complete chat UI
│   │   ├── CodeViewer.tsx           # Code display & management
│   │   ├── EnhancedDashboard.tsx    # Advanced dashboard
│   │   ├── ChatMessage.tsx          # Individual messages
│   │   ├── ChatInput.tsx            # Message input
│   │   ├── Sidebar.tsx              # Navigation
│   │   ├── StatsCard.tsx            # Metrics display
│   │   └── index.ts                 # Feature exports
│   └── demo/                  # Demo & showcase
│       └── ComponentShowcase.tsx   # Live component demos
├── pages/
│   ├── EnhancedDashboardPage.tsx   # Main dashboard page
│   ├── EnhancedChat.tsx            # Advanced chat page
│   ├── Dashboard.tsx               # Basic dashboard
│   ├── Chat.tsx                    # Basic chat
│   ├── Analytics.tsx               # Analytics page
│   ├── Settings.tsx                # Settings page
│   └── index.ts                    # Page exports
├── hooks/
│   ├── useAI.ts               # AI interaction hooks
│   ├── useWebSocket.ts        # WebSocket management
│   └── index.ts               # Hook exports
├── services/
│   ├── api.ts                 # REST API service
│   ├── websocket.ts           # WebSocket service
│   └── index.ts               # Service exports
├── types/
│   └── index.ts               # TypeScript definitions
├── utils/
│   └── helpers.ts             # Utility functions
├── App.tsx                    # Main app component
├── main.tsx                   # App entry point
└── index.css                  # Global styles
```

## 🎯 Key Features

### 💬 Advanced Chat System
- Real-time messaging with WebSocket
- AI model selection and switching
- File attachments (images, PDFs, docs)
- Code generation with syntax highlighting
- Export conversations in multiple formats
- Dark/light theme support
- Typing indicators and presence

### 🤖 AI Model Management
- Multiple AI provider support (OpenAI, Anthropic, Google)
- Cost tracking and optimization
- Performance metrics and comparison
- Auto-selection based on task requirements
- Real-time model availability status

### 📊 Enhanced Analytics Dashboard
- Real-time cost and usage tracking
- Interactive charts and visualizations
- System health monitoring
- Recent activity overview
- Quick action shortcuts
- Auto-refresh capabilities

### 🛠️ Developer Experience
- **Type Safety**: Full TypeScript coverage with strict mode
- **Error Handling**: Comprehensive error boundaries
- **Performance**: Optimized with React Query caching
- **Accessibility**: ARIA compliant components
- **Testing**: Built for testability
- **Modern Patterns**: Hooks, functional components, custom hooks

## 🚀 Getting Started

### Prerequisites
- Node.js 20+
- npm or yarn

### Installation
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Type checking
npm run type-check

# Linting
npm run lint
```

### Environment Variables
Create a `.env` file in the frontend directory:

```env
VITE_API_URL=http://localhost:8000/api/v1
VITE_WS_URL=http://localhost:8000
```

## 🔗 Available Routes

- `/` - Enhanced Dashboard (default)
- `/enhanced-chat` - Advanced chat interface with AI models
- `/showcase` - Live component demonstrations
- `/analytics` - Analytics and metrics
- `/settings` - Application settings
- `/dashboard` - Basic dashboard (legacy)
- `/chat` - Basic chat (legacy)

## 🎨 Component Usage Examples

### AI Model Selector
```tsx
import { AIModelSelector } from '@/components/features';

<AIModelSelector
  selectedModel={currentModel}
  onModelSelect={handleModelChange}
  showCosts={true}
  showMetrics={true}
  onAutoSelect={handleAutoSelect}
/>
```

### Enhanced Chat Interface
```tsx
import { EnhancedChatInterface } from '@/components/features';

<EnhancedChatInterface
  session={chatSession}
  messages={messages}
  onSendMessage={handleSendMessage}
  onFileUpload={handleFileUpload}
  theme="light"
  onThemeChange={setTheme}
/>
```

### Code Viewer
```tsx
import { CodeViewer } from '@/components/features';

<CodeViewer
  files={codeFiles}
  selectedFileId={selectedFile}
  onFileSelect={setSelectedFile}
  onDownload={handleDownload}
  showPreview={true}
/>
```

### Enhanced Dashboard
```tsx
import { EnhancedDashboard } from '@/components/features';

<EnhancedDashboard
  costData={costHistory}
  recentConversations={conversations}
  systemStatus={status}
  onQuickAction={handleQuickAction}
  onRefreshData={refreshData}
/>
```

## 🛡️ Error Handling & Toast Notifications

The application includes comprehensive error handling and user feedback:

```tsx
import { ErrorBoundary, useSuccessToast, useErrorToast } from '@/components/ui';

// Wrap components with error boundaries
<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>

// Use toast notifications
const successToast = useSuccessToast();
const errorToast = useErrorToast();

successToast('Operation completed successfully!');
errorToast('Something went wrong. Please try again.');
```

## 🧪 Component Showcase

Visit `/showcase` to see all components in action with live demos and interactive examples.

## 📦 Building & Deployment

```bash
# Production build
npm run build

# Preview production build
npm run preview

# Docker build
docker build -f Dockerfile.prod -t ai-chat-frontend .
```

## 🔧 Integration with Backend

This frontend is designed to work with the FastAPI backend located in `../backend/`. Make sure the backend is running on `http://localhost:8000` for full functionality.

## 📚 Additional Resources

- [React Documentation](https://react.dev/)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Query Documentation](https://tanstack.com/query/latest)
- [Socket.io Documentation](https://socket.io/docs/v4/)