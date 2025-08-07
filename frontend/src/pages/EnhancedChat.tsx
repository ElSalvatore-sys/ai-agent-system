import React, { useState, useMemo, useEffect } from 'react';
import { Settings } from 'lucide-react';
import { Button, ErrorBoundary, useSuccessToast, useErrorToast } from '@/components/ui';
import { ConversationsProvider, useConversations } from '@/context/ConversationsContext';
import { 
  SmartModelSelector, 
  EnhancedChatInterface, 
  CodeViewer,
  PerformanceIndicators,
  AdvancedConfigPanel
} from '@/components/features';
import { useAI, useChatWebSocket } from '@/hooks';
import { generateId } from '@/utils/helpers';
import type { Message, AIModel } from '@/types';

interface SmartAIModel extends AIModel {
  costPerToken: number;
  averageResponseTime: number;
  performance: 'excellent' | 'good' | 'fair';
  bestFor: string[];
  color: string;
  status: 'online' | 'offline' | 'loading';
}

interface FileUpload {
  id: string;
  file: File;
  type: 'pdf' | 'image' | 'document';
  status: 'uploading' | 'uploaded' | 'error';
  url?: string;
}

interface CodeFile {
  id:string;
  name: string;
  language: string;
  content: string;
  size: number;
  lastModified: Date;
}

const sampleModels: SmartAIModel[] = [
    { id: '1', name: 'Llama 3 8B', provider: 'Local', description: 'Fast and efficient for most tasks.', hostType: 'local', size: '8GB', costPerToken: 0, averageResponseTime: 50, performance: 'excellent', bestFor: ['coding', 'chat'], color: 'green', status: 'online' },
    { id: '2', name: 'Mistral 7B', provider: 'Local', description: 'A popular and powerful open-source model.', hostType: 'local', size: '7GB', costPerToken: 0, averageResponseTime: 60, performance: 'good', bestFor: ['writing', 'summarization'], color: 'blue', status: 'online' },
    { id: '3', name: 'GPT-4 Omni', provider: 'OpenAI', description: 'The latest and greatest from OpenAI.', hostType: 'cloud', size: 'N/A', costPerToken: 0.00015, averageResponseTime: 200, performance: 'excellent', bestFor: ['reasoning', 'complex tasks'], color: 'purple', status: 'online' },
    { id: '4', name: 'Phi-3 Mini', provider: 'Local', description: 'A small but surprisingly capable model.', hostType: 'local', size: '3.8GB', costPerToken: 0, averageResponseTime: 40, performance: 'good', bestFor: ['chat', 'fast responses'], color: 'orange', status: 'offline' },
];


const EnhancedChatContent: React.FC = () => {
  const { state, dispatch } = useConversations();
  const { activeSessionId } = state;
  const currentSession = activeSessionId ? state.sessions[activeSessionId] : null;

  const [models, setModels] = useState<SmartAIModel[]>(sampleModels);
  const [selectedModel, setSelectedModel] = useState<SmartAIModel>(sampleModels[0]);
  const [theme, setTheme] = useState<'light' | 'dark'>('light');
  const [codeFiles, setCodeFiles] = useState<CodeFile[]>([]);
  const [selectedCodeFileId, setSelectedCodeFileId] = useState<string>('');
  const [showCodeViewer, setShowCodeViewer] = useState(false);
  const [performanceMetrics, setPerformanceMetrics] = useState({ latency: 55, throughput: 120, cpuUsage: 30, memoryUsage: 60 });
  const [isConfigPanelOpen, setIsConfigPanelOpen] = useState(false);
  const [advancedConfig, setAdvancedConfig] = useState({
    temperature: 0.7,
    topP: 1,
    quantization: '8-bit',
    streamResponse: true,
  });
  
  const successToast = useSuccessToast();
  const errorToast = useErrorToast();

  const { sendMessage: sendAIMessage, isLoading: aiLoading } = useAI();
  const { typingUsers } = useChatWebSocket(activeSessionId || undefined);

  useEffect(() => {
    const interval = setInterval(() => {
      setPerformanceMetrics({
        latency: Math.floor(40 + Math.random() * 30),
        throughput: Math.floor(100 + Math.random() * 50),
        cpuUsage: Math.floor(20 + Math.random() * 40),
        memoryUsage: Math.floor(50 + Math.random() * 20)
      });
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (currentSession?.modelId) {
        const newSelectedModel = models.find(m => m.id === currentSession.modelId);
        if (newSelectedModel) {
            setSelectedModel(newSelectedModel);
        }
    }
  }, [currentSession, models]);

  const totalCost = useMemo(() => {
    return (currentSession?.messages ?? []).reduce((acc: number, msg: Message) => acc + (msg.cost || 0), 0);
  }, [currentSession]);

  const handleModelSelect = (model: SmartAIModel) => {
    if (activeSessionId) {
        dispatch({ type: 'UPDATE_SESSION_MODEL', payload: { sessionId: activeSessionId, modelId: model.id } });
    }
    setSelectedModel(model);
    successToast(`Switched to ${model.name}`);
  };

  const handleModelInstall = (model: SmartAIModel) => {
    setModels(prev => prev.map(m => m.id === model.id ? { ...m, status: 'loading' } : m));
    successToast(`Installing ${model.name}...`);

    setTimeout(() => {
        setModels(prev => prev.map(m => m.id === model.id ? { ...m, status: 'online' } : m));
        successToast(`${model.name} installed successfully!`);
        setSelectedModel(models.find(m => m.id === model.id)!);
    }, 3000);
  };

  const handleFileUpload = async (files: FileList): Promise<FileUpload[]> => {
    const uploads: FileUpload[] = [];
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const upload: FileUpload = {
        id: generateId(),
        file,
        type: file.type.startsWith('image/') ? 'image' : 
              file.type === 'application/pdf' ? 'pdf' : 'document',
        status: 'uploading',
      };
      
      uploads.push(upload);
      
      try {
        await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
        upload.status = 'uploaded';
        upload.url = URL.createObjectURL(file);
        
        successToast(`${file.name} uploaded successfully`);
      } catch (error) {
        upload.status = 'error';
        errorToast(`Failed to upload ${file.name}`);
      }
    }
    
    return uploads;
  };

  const handleSendMessage = async (content: string) => {
    if (!activeSessionId) {
      dispatch({ type: 'CREATE_SESSION', payload: { title: `Chat ${new Date().toLocaleString()}` } });
      return;
    }

    try {
      const userMessage: Message = {
        id: generateId(),
        content,
        senderId: 'local-user',
        timestamp: new Date(),
        type: 'text',
      };
      dispatch({ type: 'ADD_MESSAGE', payload: { sessionId: activeSessionId, message: userMessage } });
      
      const response = await sendAIMessage(content, selectedModel?.id, activeSessionId, messages, {
        temperature: advancedConfig.temperature,
        maxTokens: 2048
      });
      
      const aiMessage: Message = {
        id: generateId(),
        content: response.content,
        senderId: 'ai',
        timestamp: new Date(),
        type: 'ai',
        cost: response.tokens * (selectedModel?.costPerToken || 0),
      };
      dispatch({ type: 'ADD_MESSAGE', payload: { sessionId: activeSessionId, message: aiMessage } });

      if (response.content.includes('```')) {
        const codeBlocks = extractCodeBlocks(response.content);
        if (codeBlocks.length > 0) {
          setCodeFiles(prev => [...prev, ...codeBlocks]);
          setShowCodeViewer(true);
          if (codeBlocks.length === 1) {
            setSelectedCodeFileId(codeBlocks[0].id);
          }
        }
      }
      
    } catch (error) {
      console.error('Failed to send message:', error);
      errorToast('Failed to send message');
    }
  };

  const extractCodeBlocks = (content: string): CodeFile[] => {
    const codeBlockRegex = /```(\w+)?\n([\s\S]*?)```/g;
    const blocks: CodeFile[] = [];
    let match;
    let index = 0;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      const language = match[1] || 'text';
      const code = match[2].trim();
      const fileName = `generated_${index}_${Date.now()}.${getFileExtension(language)}`;
      
      blocks.push({
        id: generateId(),
        name: fileName,
        language,
        content: code,
        size: new Blob([code]).size,
        lastModified: new Date(),
      });
      
      index++;
    }

    return blocks;
  };

  const getFileExtension = (language: string): string => {
    const extensions: Record<string, string> = {
      javascript: 'js',
      typescript: 'ts',
      python: 'py',
      html: 'html',
      css: 'css',
      json: 'json',
      jsx: 'jsx',
      tsx: 'tsx',
    };
    return extensions[language] || 'txt';
  };

  const handleCodeFileDownload = (fileId: string) => {
    const file = codeFiles.find(f => f.id === fileId);
    if (file) {
      const blob = new Blob([file.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      successToast(`Downloaded ${file.name}`);
    }
  };

  const handleDownloadAllFiles = () => {
    codeFiles.forEach(file => {
      setTimeout(() => handleCodeFileDownload(file.id), 100);
    });
  };

  const handleQuickAction = (action: 'generate-code' | 'create-pdf' | 'deploy-app') => {
    const prompts = {
      'generate-code': 'Generate a complete React TypeScript component with modern patterns',
      'create-pdf': 'Create a PDF document with professional formatting',
      'deploy-app': 'Help me deploy this application to a cloud platform',
    };
    
    handleSendMessage(prompts[action]);
  };

  const messages = currentSession?.messages || [];

  return (
    <ErrorBoundary>
      <div className="flex h-[calc(100vh-4rem)] bg-gray-50">
        <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
          <div className="p-4 border-b border-gray-200">
            <SmartModelSelector
              models={models}
              selectedModel={selectedModel}
              onModelSelect={handleModelSelect}
              onModelInstall={handleModelInstall}
            />
          </div>
          <div className="p-4 border-b border-gray-200">
            <h3 className="text-sm font-medium text-gray-900 mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction('generate-code')}
                className="w-full justify-start"
              >
                Generate Code
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction('create-pdf')}
                className="w-full justify-start"
              >
                Create PDF
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handleQuickAction('deploy-app')}
                className="w-full justify-start"
              >
                Deploy App
              </Button>
            </div>
          </div>
          <div className="p-4 border-b border-gray-200">
            <PerformanceIndicators metrics={performanceMetrics} />
          </div>
          {codeFiles.length > 0 && (
            <div className="p-4 border-b border-gray-200">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-sm font-medium text-gray-900">Generated Files</h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowCodeViewer(!showCodeViewer)}
                >
                  {showCodeViewer ? 'Hide' : 'Show'} Code
                </Button>
              </div>
              <div className="space-y-1">
                {codeFiles.slice(-5).map((file) => (
                  <button
                    key={file.id}
                    onClick={() => {
                      setSelectedCodeFileId(file.id);
                      setShowCodeViewer(true);
                    }}
                    className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded-md"
                  >
                    {file.name}
                  </button>
                ))}
              </div>
            </div>
          )}
          <div className="mt-auto p-4 border-t border-gray-200">
            <Button
              variant="ghost"
              size="sm"
              className="w-full justify-start"
              onClick={() => setIsConfigPanelOpen(true)}
            >
              <Settings className="h-4 w-4 mr-2" />
              Settings
            </Button>
          </div>
        </div>
        <div className="flex-1 flex">
          <div className={`${showCodeViewer && codeFiles.length > 0 ? 'w-1/2' : 'w-full'}`}>
            <EnhancedChatInterface
              session={currentSession}
              messages={messages}
              onSendMessage={handleSendMessage}
              onFileUpload={handleFileUpload}
              isLoading={aiLoading}
              typingUsers={typingUsers}
              theme={theme}
              onThemeChange={setTheme}
              model={selectedModel}
              models={models}
              onModelSelect={handleModelSelect}
              totalCost={totalCost}
            />
          </div>
          {showCodeViewer && codeFiles.length > 0 && (
            <div className="w-1/2 border-l border-gray-200">
              <CodeViewer
                files={codeFiles}
                selectedFileId={selectedCodeFileId}
                onFileSelect={setSelectedCodeFileId}
                onDownload={handleCodeFileDownload}
                onDownloadAll={handleDownloadAllFiles}
                showPreview={true}
                previewUrl="about:blank"
              />
            </div>
          )}
        </div>
      </div>
      <AdvancedConfigPanel
        isOpen={isConfigPanelOpen}
        onClose={() => setIsConfigPanelOpen(false)}
        config={advancedConfig}
        onConfigChange={setAdvancedConfig}
      />
    </ErrorBoundary>
  );
};

export const EnhancedChat: React.FC = () => (
  <ConversationsProvider>
    <EnhancedChatContent />
  </ConversationsProvider>
);
