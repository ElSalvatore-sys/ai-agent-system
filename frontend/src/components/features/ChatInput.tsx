import React, { useState, useRef, useEffect } from 'react';
import { Send, Mic, MicOff, Paperclip } from 'lucide-react';
import { Button } from '@/components/ui';
import { cn } from '@/utils/helpers';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  onTyping?: (isTyping: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
  maxLength?: number;
  className?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  onTyping,
  disabled = false,
  placeholder = 'Type your message...',
  maxLength = 2000,
  className,
}) => {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const typingTimeoutRef = useRef<number | null>(null);
  const recognitionRef = useRef<any | null>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  useEffect(() => {
    if (onTyping) {
      if (message.length > 0) {
        onTyping(true);
        if (typingTimeoutRef.current) {
          clearTimeout(typingTimeoutRef.current);
        }
        typingTimeoutRef.current = window.setTimeout(() => {
          onTyping(false);
        }, 1000);
      } else {
        onTyping(false);
      }
    }
    return () => {
      if (typingTimeoutRef.current) {
        clearTimeout(typingTimeoutRef.current);
      }
    };
  }, [message, onTyping]);

  useEffect(() => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (SpeechRecognition) {
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.interimResults = true;
      recognitionRef.current.continuous = true;

      recognitionRef.current.onresult = (event: any) => {
        let interimTranscript = '';
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }
        setMessage(finalTranscript + interimTranscript);
      };

      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };

      recognitionRef.current.onerror = (event: any) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
      };
    }
  }, []);

  const handleSend = () => {
    const trimmedMessage = message.trim();
    if (!trimmedMessage || disabled) return;
    onSendMessage(trimmedMessage);
    setMessage('');
    if (onTyping) onTyping(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = () => {
    console.log('File upload not implemented yet');
  };

  const toggleRecording = () => {
    if (isRecording) {
      recognitionRef.current?.stop();
    } else {
      recognitionRef.current?.start();
    }
    setIsRecording(!isRecording);
  };

  const canSend = message.trim().length > 0 && !disabled;

  return (
    <div className={cn('border-t border-gray-200 bg-white p-4', className)}>
      <div className="flex items-end gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={handleFileUpload}
          disabled={disabled}
          className="flex-shrink-0 p-2"
        >
          <Paperclip className="h-5 w-5" />
        </Button>
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            maxLength={maxLength}
            rows={1}
            className={cn(
              'w-full resize-none rounded-lg border border-gray-300 px-4 py-2 pr-12',
              'focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20',
              'disabled:bg-gray-50 disabled:text-gray-500',
              'max-h-32 min-h-[44px]'
            )}
          />
          {maxLength && (
            <div className="absolute bottom-1 right-2 text-xs text-gray-400">
              {message.length}/{maxLength}
            </div>
          )}
        </div>
        <Button
          variant={isRecording ? "destructive" : "ghost"}
          size="sm"
          onClick={toggleRecording}
          disabled={disabled}
          className="flex-shrink-0 p-2"
        >
          {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
        </Button>
        <Button
          onClick={handleSend}
          disabled={!canSend}
          size="sm"
          className="flex-shrink-0 p-2"
        >
          <Send className="h-5 w-5" />
        </Button>
      </div>
      {isRecording && (
        <div className="mt-2 flex items-center gap-2 text-sm text-red-600">
          <div className="flex space-x-1">
            <div className="h-2 w-2 bg-red-600 rounded-full animate-pulse"></div>
            <div className="h-2 w-2 bg-red-600 rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
            <div className="h-2 w-2 bg-red-600 rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
          </div>
          Recording...
        </div>
      )}
    </div>
  );
};
