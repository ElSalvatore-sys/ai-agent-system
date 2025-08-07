import React, { useState, useRef } from 'react';
import { 
  Copy, 
  Download, 
  Eye, 
  Code, 
  Play, 
  ExternalLink,
  Check,
  FileText,
  Globe,
  Smartphone
} from 'lucide-react';
import { Button, Card, CardHeader, CardTitle, CardContent, Modal } from '@/components/ui';
import { cn, copyToClipboard } from '@/utils/helpers';

interface CodeFile {
  id: string;
  name: string;
  language: string;
  content: string;
  size: number;
  lastModified: Date;
}

interface CodeViewerProps {
  files: CodeFile[];
  selectedFileId?: string;
  onFileSelect: (fileId: string) => void;
  onDownload?: (fileId: string) => void;
  onDownloadAll?: () => void;
  showPreview?: boolean;
  previewUrl?: string;
  className?: string;
}

const languageColors = {
  javascript: 'text-yellow-600 bg-yellow-50',
  typescript: 'text-blue-600 bg-blue-50',
  python: 'text-green-600 bg-green-50',
  html: 'text-orange-600 bg-orange-50',
  css: 'text-purple-600 bg-purple-50',
  json: 'text-gray-600 bg-gray-50',
  jsx: 'text-cyan-600 bg-cyan-50',
  tsx: 'text-indigo-600 bg-indigo-50',
} as const;

const getLanguageFromFileName = (fileName: string): string => {
  const ext = fileName.split('.').pop()?.toLowerCase();
  switch (ext) {
    case 'js':
      return 'javascript';
    case 'ts':
      return 'typescript';
    case 'py':
      return 'python';
    case 'html':
      return 'html';
    case 'css':
      return 'css';
    case 'json':
      return 'json';
    case 'jsx':
      return 'jsx';
    case 'tsx':
      return 'tsx';
    default:
      return 'text';
  }
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

export const CodeViewer: React.FC<CodeViewerProps> = ({
  files,
  selectedFileId,
  onFileSelect,
  onDownload,
  onDownloadAll,
  showPreview = false,
  previewUrl,
  className,
}) => {
  const [copiedFileId, setCopiedFileId] = useState<string | null>(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewMode, setPreviewMode] = useState<'desktop' | 'mobile'>('desktop');
  const codeRef = useRef<HTMLPreElement>(null);

  const selectedFile = files.find(f => f.id === selectedFileId) || files[0];

  const handleCopy = async (content: string, fileId: string) => {
    const success = await copyToClipboard(content);
    if (success) {
      setCopiedFileId(fileId);
      setTimeout(() => setCopiedFileId(null), 2000);
    }
  };

  const handleDownload = (file: CodeFile) => {
    if (onDownload) {
      onDownload(file.id);
    } else {
      // Default download behavior
      const blob = new Blob([file.content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const getLanguageColor = (language: string) => {
    return languageColors[language as keyof typeof languageColors] || 'text-gray-600 bg-gray-50';
  };

  if (!selectedFile) {
    return (
      <Card className={cn('h-full flex items-center justify-center', className)}>
        <div className="text-center text-gray-500">
          <Code className="h-12 w-12 mx-auto mb-4 text-gray-300" />
          <p>No code files to display</p>
        </div>
      </Card>
    );
  }

  return (
    <div className={cn('flex flex-col h-full', className)}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <Code className="h-5 w-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Code Viewer</h3>
          <span className="text-sm text-gray-500">
            {files.length} file{files.length !== 1 ? 's' : ''}
          </span>
        </div>

        <div className="flex items-center gap-2">
          {/* Preview Button */}
          {showPreview && previewUrl && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowPreviewModal(true)}
              className="flex items-center gap-2"
            >
              <Eye className="h-4 w-4" />
              Preview
            </Button>
          )}

          {/* Download All Button */}
          {files.length > 1 && (
            <Button
              variant="outline"
              size="sm"
              onClick={onDownloadAll}
              className="flex items-center gap-2"
            >
              <Download className="h-4 w-4" />
              Download All
            </Button>
          )}
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* File List Sidebar */}
        <div className="w-64 border-r border-gray-200 bg-gray-50">
          <div className="p-4">
            <h4 className="text-sm font-medium text-gray-900 mb-3">Files</h4>
            <div className="space-y-1">
              {files.map((file) => {
                const isSelected = file.id === selectedFile.id;
                return (
                  <button
                    key={file.id}
                    onClick={() => onFileSelect(file.id)}
                    className={cn(
                      'w-full text-left p-2 rounded-lg transition-colors',
                      isSelected
                        ? 'bg-blue-50 border border-blue-200 text-blue-900'
                        : 'hover:bg-gray-100 text-gray-700'
                    )}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <FileText className="h-4 w-4 flex-shrink-0" />
                      <span className="text-sm font-medium truncate">{file.name}</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <span className={cn(
                        'px-2 py-1 rounded-full',
                        getLanguageColor(getLanguageFromFileName(file.name))
                      )}>
                        {getLanguageFromFileName(file.name)}
                      </span>
                      <span>{formatFileSize(file.size)}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Code Display */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* File Actions Bar */}
          <div className="flex items-center justify-between p-3 border-b border-gray-200 bg-white">
            <div className="flex items-center gap-3">
              <span className="font-medium text-gray-900">{selectedFile.name}</span>
              <span className={cn(
                'px-2 py-1 rounded-full text-xs',
                getLanguageColor(selectedFile.language)
              )}>
                {selectedFile.language}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleCopy(selectedFile.content, selectedFile.id)}
                className="flex items-center gap-2"
              >
                {copiedFileId === selectedFile.id ? (
                  <>
                    <Check className="h-4 w-4 text-green-600" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    Copy
                  </>
                )}
              </Button>

              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleDownload(selectedFile)}
                className="flex items-center gap-2"
              >
                <Download className="h-4 w-4" />
                Download
              </Button>
            </div>
          </div>

          {/* Code Content */}
          <div className="flex-1 overflow-auto bg-gray-800 text-white">
            <SyntaxHighlighter
              language={selectedFile.language}
              style={vscDarkPlus}
              showLineNumbers
              wrapLines
              customStyle={{
                margin: 0,
                padding: '1rem',
                backgroundColor: 'transparent',
                height: '100%',
              }}
              codeTagProps={{
                style: {
                  fontFamily: '"Fira Code", monospace',
                },
              }}
            >
              {selectedFile.content}
            </SyntaxHighlighter>
          </div>
        </div>
      </div>

      {/* Preview Modal */}
      {showPreview && previewUrl && (
        <Modal
          isOpen={showPreviewModal}
          onClose={() => setShowPreviewModal(false)}
          title="Live Preview"
          size="xl"
        >
          <div className="space-y-4">
            {/* Preview Controls */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Button
                  variant={previewMode === 'desktop' ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setPreviewMode('desktop')}
                  className="flex items-center gap-2"
                >
                  <Globe className="h-4 w-4" />
                  Desktop
                </Button>
                <Button
                  variant={previewMode === 'mobile' ? 'primary' : 'outline'}
                  size="sm"
                  onClick={() => setPreviewMode('mobile')}
                  className="flex items-center gap-2"
                >
                  <Smartphone className="h-4 w-4" />
                  Mobile
                </Button>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => window.open(previewUrl, '_blank')}
                className="flex items-center gap-2"
              >
                <ExternalLink className="h-4 w-4" />
                Open in New Tab
              </Button>
            </div>

            {/* Preview Frame */}
            <div className={cn(
              'border border-gray-200 rounded-lg overflow-hidden bg-white',
              previewMode === 'mobile' ? 'max-w-sm mx-auto' : 'w-full'
            )}>
              <iframe
                src={previewUrl}
                className={cn(
                  'w-full border-0',
                  previewMode === 'mobile' ? 'h-[600px]' : 'h-[500px]'
                )}
                title="Live Preview"
              />
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};