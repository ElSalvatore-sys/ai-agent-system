import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, Button, Modal } from '@/components/ui';
import { 
  MultiAIModelSelector, 
  EnhancedChatInterface, 
  CodeViewer, 
  EnhancedDashboard 
} from '@/components/features';
import { generateId } from '@/utils/helpers';
import type { AIModel, Message, ChatSession } from '@/types';

// Mock data for demo purposes
const mockModels: AIModel[] = [
  {
    id: 'o3',
    name: 'OpenAI o3',
    description: 'Latest reasoning model',
    provider: 'OpenAI',
    hostType: 'cloud',
    status: 'online',
    capabilities: ['coding', 'reasoning'],
    maxTokens: 128000,
    isAvailable: true,
  },
  {
    id: 'claude-3.5',
    name: 'Claude 3.5 Sonnet',
    description: 'Balanced general purpose model',
    provider: 'Claude',
    hostType: 'cloud',
    status: 'online',
    capabilities: ['chat', 'reasoning'],
    maxTokens: 200000,
    isAvailable: true,
  },
];

const mockMessages: Message[] = [
  {
    id: '1',
    content: 'Hello! Can you help me create a React component?',
    senderId: 'user',
    timestamp: new Date(Date.now() - 120000),
    type: 'text',
  },
  {
    id: '2',
    content: `I'll help you create a React component. Here's a modern TypeScript component example:

\`\`\`tsx
import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const CustomButton: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
}) => {
  const baseClasses = 'px-4 py-2 rounded-md font-medium transition-colors';
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={\`\${baseClasses} \${variantClasses[variant]} \${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }\`}
    >
      {children}
    </button>
  );
};
\`\`\`

This component includes:
- TypeScript interfaces for type safety
- Variant system for different styles
- Disabled state handling
- Modern React patterns with functional components
- Tailwind CSS for styling`,
    senderId: 'ai',
    timestamp: new Date(Date.now() - 60000),
    type: 'ai',
  },
];

const mockCodeFiles = [
  {
    id: '1',
    name: 'CustomButton.tsx',
    language: 'tsx',
    content: `import React from 'react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary';
  disabled?: boolean;
}

export const CustomButton: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  disabled = false,
}) => {
  const baseClasses = 'px-4 py-2 rounded-md font-medium transition-colors';
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={\`\${baseClasses} \${variantClasses[variant]} \${
        disabled ? 'opacity-50 cursor-not-allowed' : ''
      }\`}
    >
      {children}
    </button>
  );
};`,
    size: 1024,
    lastModified: new Date(),
  },
];

export const ComponentShowcase: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<AIModel>(mockModels[0]);
  const [showModal, setShowModal] = useState(false);
  const [selectedCodeFile, setSelectedCodeFile] = useState(mockCodeFiles[0].id);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');

  const mockSession: ChatSession = {
    id: 'demo-session',
    title: 'Component Demo Chat',
    userId: 'demo-user',
    messages: mockMessages,
    createdAt: new Date(),
    updatedAt: new Date(),
    isActive: true,
  };

  const mockCostData = [
    { date: '2024-01-01', cost: 2.5, tokens: 25000, model: 'o3' },
    { date: '2024-01-02', cost: 3.2, tokens: 32000, model: 'claude-3.5' },
    { date: '2024-01-03', cost: 1.8, tokens: 18000, model: 'gemini-pro' },
    { date: '2024-01-04', cost: 4.1, tokens: 41000, model: 'o3' },
    { date: '2024-01-05', cost: 2.9, tokens: 29000, model: 'claude-3.5' },
    { date: '2024-01-06', cost: 3.7, tokens: 37000, model: 'o3' },
    { date: '2024-01-07', cost: 2.3, tokens: 23000, model: 'gemini-pro' },
  ];

  const mockConversations = [
    {
      id: '1',
      title: 'React Component Help',
      lastMessage: 'Thanks for the component example!',
      timestamp: new Date(),
      messageCount: 15,
      cost: 0.45,
    },
    {
      id: '2',
      title: 'API Integration',
      lastMessage: 'The API is working perfectly now.',
      timestamp: new Date(),
      messageCount: 23,
      cost: 0.67,
    },
  ];

  const mockSystemStatus = {
    api: 'operational' as const,
    database: 'operational' as const,
    ai_models: 'operational' as const,
    websocket: 'operational' as const,
  };

  const handleSendMessage = (content: string) => {
    console.log('Demo: Message sent:', content);
  };

  const handleModelSelect = (model: AIModel) => {
    setSelectedModel(model);
    console.log('Demo: Model selected:', model.name);
  };

  const handleFileUpload = async (files: FileList) => {
    console.log('Demo: Files uploaded:', Array.from(files).map(f => f.name));
    return [];
  };

  const handleQuickAction = (action: string) => {
    console.log('Demo: Quick action:', action);
  };

  return (
    <div className="space-y-8 p-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-4">
          Essential React Components Showcase
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          This showcase demonstrates all the essential components created for the AI agent system, 
          including the chat interface, AI model selector, code viewer, and enhanced dashboard.
        </p>
      </div>

      {/* AI Model Selector Demo */}
      <Card>
        <CardHeader>
          <CardTitle>AI Model Selector Component</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-w-md">
            <MultiAIModelSelector
              models={mockModels}
              selectedModels={[selectedModel]}
              onModelToggle={handleModelSelect}
            />
          </div>
          <div className="mt-4 p-4 bg-gray-50 rounded-lg">
            <p className="text-sm text-gray-600">
              <strong>Selected Model:</strong> {selectedModel.name} - {selectedModel.description}
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Chat Interface Demo */}
      <Card>
        <CardHeader>
          <CardTitle>Enhanced Chat Interface Component</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 border border-gray-200 rounded-lg overflow-hidden">
            <EnhancedChatInterface
              session={mockSession}
              messages={mockMessages}
              onSendMessage={handleSendMessage}
              onFileUpload={handleFileUpload}
              theme={theme}
              onThemeChange={setTheme}
              model={selectedModel}
              models={mockModels}
              onModelSelect={handleModelSelect}
            />
          </div>
          <div className="mt-4 flex gap-2">
            <Button onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}>
              Toggle Theme
            </Button>
            <Button variant="outline" onClick={() => setShowModal(true)}>
              Show in Modal
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Code Viewer Demo */}
      <Card>
        <CardHeader>
          <CardTitle>Code Viewer Component</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80 border border-gray-200 rounded-lg overflow-hidden">
            <CodeViewer
              files={mockCodeFiles}
              selectedFileId={selectedCodeFile}
              onFileSelect={setSelectedCodeFile}
              showPreview={true}
            />
          </div>
        </CardContent>
      </Card>

      {/* Enhanced Dashboard Demo */}
      <Card>
        <CardHeader>
          <CardTitle>Enhanced Dashboard Component</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="border border-gray-200 rounded-lg overflow-hidden">
            <EnhancedDashboard
              costData={mockCostData}
              recentConversations={mockConversations}
              systemStatus={mockSystemStatus}
              onQuickAction={handleQuickAction}
              onRefreshData={() => console.log('Demo: Data refreshed')}
            />
          </div>
        </CardContent>
      </Card>

      {/* Modal Demo */}
      <Modal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
        title="Chat Interface in Modal"
        size="xl"
      >
        <div className="h-96">
          <EnhancedChatInterface
            session={mockSession}
            messages={mockMessages}
            onSendMessage={handleSendMessage}
            onFileUpload={handleFileUpload}
            theme={theme}
            onThemeChange={setTheme}
            model={selectedModel}
            models={mockModels}
            onModelSelect={handleModelSelect}
          />
        </div>
      </Modal>
    </div>
  );
};