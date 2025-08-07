import React, { useState, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  Send, 
  Paperclip, 
  Download, 
  Sun, 
  Moon, 
  FileText,
  Image,
  X,
  Check,
  Settings,
} from 'lucide-react';
import { Button, Modal, Loading } from '@/components/ui';
import { ChatMessage, ChatInput, CostTracker, ContextPanel } from '@/components/features';
import { cn, formatDate } from '@/utils/helpers';
import type { Message, ChatSession, AIModel } from '@/types';

interface FileUpload {
  id: string;
  file: File;
  type: 'pdf' | 'image' | 'document';
  status: 'uploading' | 'uploaded' | 'error';
  url?: string;
}

interface EnhancedChatInterfaceProps {
  session?: ChatSession;
  messages: Message[];
  onSendMessage: (content: string, attachments?: FileUpload[]) => void;
  onFileUpload?: (files: FileList) => Promise<FileUpload[]>;
  isLoading?: boolean;
  typingUsers?: string[];
  theme?: 'light' | 'dark';
  onThemeChange?: (theme: 'light' | 'dark') => void;
  className?: string;
  model?: AIModel;
  models?: AIModel[];
  onModelSelect?: (model: AIModel) => void;
  /** Total cost accumulated so far (optional). */
  totalCost?: number;
}

export const EnhancedChatInterface: React.FC<EnhancedChatInterfaceProps> = ({
  session,
  messages,
  onSendMessage,
  onFileUpload,
  isLoading = false,
  typingUsers = [],
  theme = 'light',
  onThemeChange,
  className,
  model,
  models = [],
  onModelSelect,
  totalCost = 0,
}) => {
  const [uploadedFiles, setUploadedFiles] = useState<FileUpload[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [showExportModal, setShowExportModal] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [showContextPanel, setShowContextPanel] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleFileUpload = async (files: FileList) => {
    if (!onFileUpload) return;

    setIsUploading(true);
    try {
      const newFiles = await onFileUpload(files);
      setUploadedFiles(prev => [...prev, ...newFiles]);
    } catch (error) {
      console.error('File upload failed:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleFileDrop = (acceptedFiles: File[]) => {
    const fileList = new DataTransfer();
    acceptedFiles.forEach(file => fileList.items.add(file));
    handleFileUpload(fileList.files);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleFileDrop,
    noClick: true,
  });

  const handleFileSelect = () => {
    fileInputRef.current?.click();
  };

  const removeFile = (fileId: string) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const handleSendMessage = (content: string) => {
    onSendMessage(content, uploadedFiles);
    setUploadedFiles([]); // Clear files after sending
  };

  const exportConversation = async (format: 'txt' | 'json' | 'pdf') => {
    setIsExporting(true);
    try {
      let content = '';
      const timestamp = formatDate(new Date());

      if (format === 'txt') {
        content = `Chat Conversation - ${session?.title || 'Untitled'}\n`;
        content += `Exported: ${timestamp}\n\n`;
        content += messages.map(msg => 
          `[${formatDate(msg.timestamp)}] ${msg.type === 'ai' ? 'AI' : 'User'}: ${msg.content}`
        ).join('\n\n');
      } else if (format === 'json') {
        content = JSON.stringify({
          session: {
            title: session?.title || 'Untitled',
            exportedAt: timestamp,
          },
          messages: messages.map(msg => ({
            id: msg.id,
            content: msg.content,
            type: msg.type,
            timestamp: msg.timestamp,
          })),
        }, null, 2);
      }

      // Create and download file
      const blob = new Blob([content], { 
        type: format === 'json' ? 'application/json' : 'text/plain' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `chat-${session?.title || 'conversation'}-${Date.now()}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setShowExportModal(false);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  const getFileIcon = (type: string) => {
    switch (type) {
      case 'pdf':
      case 'document':
        return <FileText className="h-4 w-4" />;
      case 'image':
        return <Image className="h-4 w-4" />;
      default:
        return <Paperclip className="h-4 w-4" />;
    }
  };

  const isDarkTheme = theme === 'dark';

  return (
    <div className={cn(
      'flex flex-col h-full',
      isDarkTheme ? 'bg-gray-900 text-white' : 'bg-white text-gray-900',
      className
    )}>
      {/* Header */}
      <div className={cn(
        'flex items-center justify-between p-4 border-b',
        isDarkTheme ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'
      )}>
        <div>
          <h2 className="text-lg font-semibold">
            {session?.title || 'New Conversation'}
          </h2>
          <p className={cn(
            'text-sm',
            isDarkTheme ? 'text-gray-400' : 'text-gray-600'
          )}>
            {messages.length} messages
          </p>
          <CostTracker cost={totalCost} />
        </div>

        <div className="flex items-center gap-2">
          {/* Theme Toggle */}
          {onThemeChange && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onThemeChange(isDarkTheme ? 'light' : 'dark')}
              className="p-2"
            >
              {isDarkTheme ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </Button>
          )}

          {/* Export Button */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowExportModal(true)}
            className="p-2"
          >
            <Download className="h-4 w-4" />
          </Button>

          {/* Context Panel Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowContextPanel(!showContextPanel)}
            className="p-2"
          >
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4" {...getRootProps()}>
        <input {...getInputProps()} />
        {isDragActive && (
          <div className="absolute inset-0 bg-blue-500 bg-opacity-20 flex items-center justify-center">
            <p className="text-blue-700 font-semibold">Drop files to upload</p>
          </div>
        )}
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className={cn(
                'w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4',
                isDarkTheme ? 'bg-gray-800' : 'bg-gray-200'
              )}>
                <Send className={cn(
                  'h-8 w-8',
                  isDarkTheme ? 'text-gray-400' : 'text-gray-500'
                )} />
              </div>
              <h3 className={cn(
                'text-lg font-medium mb-2',
                isDarkTheme ? 'text-gray-200' : 'text-gray-900'
              )}>
                Start a conversation
              </h3>
              <p className={cn(
                'text-sm',
                isDarkTheme ? 'text-gray-400' : 'text-gray-600'
              )}>
                Send a message or upload files to begin
              </p>
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <ChatMessage
                key={message.id}
                message={message}
                isOwn={message.type !== 'ai'}
                className={isDarkTheme ? 'dark' : ''}
              />
            ))}

            {/* Typing Indicators */}
            {typingUsers.length > 0 && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                </div>
                <span>
                  {typingUsers.join(', ')} {typingUsers.length === 1 ? 'is' : 'are'} typing...
                </span>
              </div>
            )}

            {isLoading && model && (
              <div className="flex items-center gap-2 text-sm text-gray-500">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
                </div>
                <span>{model.name} is typing...</span>
              </div>
            )}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex justify-center">
                <Loading size="sm" text="AI is thinking..." />
              </div>
            )}

            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      {/* File Upload Area */}
      {uploadedFiles.length > 0 && (
        <div className={cn(
          'p-4 border-t',
          isDarkTheme ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'
        )}>
          <div className="flex flex-wrap gap-2">
            {uploadedFiles.map((file) => (
              <div
                key={file.id}
                className={cn(
                  'flex items-center gap-2 px-3 py-2 rounded-lg border',
                  isDarkTheme 
                    ? 'bg-gray-700 border-gray-600 text-gray-200' 
                    : 'bg-gray-100 border-gray-300 text-gray-900'
                )}
              >
                {getFileIcon(file.type)}
                <span className="text-sm truncate max-w-[150px]">
                  {file.file.name}
                </span>
                {file.status === 'uploading' && (
                  <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
                )}
                {file.status === 'uploaded' && (
                  <Check className="h-4 w-4 text-green-500" />
                )}
                <button
                  onClick={() => removeFile(file.id)}
                  className={cn(
                    'p-1 rounded hover:bg-gray-200',
                    isDarkTheme && 'hover:bg-gray-600'
                  )}
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className={cn(
        'border-t',
        isDarkTheme ? 'border-gray-700 bg-gray-800' : 'border-gray-200 bg-white'
      )}>
        <div className="p-4">
          <div className="flex items-end gap-2">
            {/* File Upload Button */}
            <Button
              variant="ghost"
              size="sm"
              onClick={handleFileSelect}
              disabled={isUploading}
              className="flex-shrink-0 p-2"
            >
              {isUploading ? (
                <div className="w-5 h-5 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
              ) : (
                <Paperclip className="h-5 w-5" />
              )}
            </Button>

            {/* Chat Input */}
            <div className="flex-1">
              <ChatInput
                onSendMessage={handleSendMessage}
                disabled={isLoading}
                placeholder="Type your message..."
                className={cn(
                  'rounded-lg',
                  isDarkTheme ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'
                )}
              />
            </div>
          </div>

          {/* File Input */}
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.png,.jpg,.jpeg,.gif,.doc,.docx,.txt"
            onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
            className="hidden"
          />
        </div>
      </div>

      {/* Export Modal */}
      <Modal
        isOpen={showExportModal}
        onClose={() => setShowExportModal(false)}
        title="Export Conversation"
        size="sm"
      >
        <div className="space-y-4">
          <p className="text-gray-600">
            Choose a format to export your conversation:
          </p>
          
          <div className="space-y-2">
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => exportConversation('txt')}
              disabled={isExporting}
            >
              <FileText className="h-4 w-4 mr-2" />
              Text File (.txt)
            </Button>
            
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => exportConversation('json')}
              disabled={isExporting}
            >
              <FileText className="h-4 w-4 mr-2" />
              JSON File (.json)
            </Button>
            
            <Button
              variant="outline"
              className="w-full justify-start"
              onClick={() => exportConversation('pdf')}
              disabled={true} // TODO: Implement PDF export
            >
              <FileText className="h-4 w-4 mr-2" />
              PDF File (.pdf) - Coming Soon
            </Button>
          </div>

          {isExporting && (
            <div className="flex items-center justify-center py-4">
              <Loading size="sm" text="Exporting..." />
            </div>
          )}
        </div>
      </Modal>

      {/* Context Panel */}
      <ContextPanel
        isOpen={showContextPanel}
        model={model}
        models={models}
        onModelSelect={onModelSelect}
        totalCost={totalCost}
        className="fixed top-0 right-0 h-full z-30 shadow-lg"
      />
    </div>
  );
};