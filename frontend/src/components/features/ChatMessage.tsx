import React from 'react';
import { motion } from 'framer-motion';
import { Copy, Edit2, Trash2, Bot, User, Volume2 } from 'lucide-react';
import { formatRelativeTime, copyToClipboard } from '@/utils/helpers';
import { Avatar, Button } from '@/components/ui';
import type { Message } from '@/types';

interface ChatMessageProps {
  message: Message;
  isOwn?: boolean;
  showAvatar?: boolean;
  onEdit?: (messageId: string) => void;
  onDelete?: (messageId: string) => void;
  className?: string;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({
  message,
  isOwn = false,
  showAvatar = true,
  onEdit,
  onDelete,
  className,
}) => {
  const handleCopy = async () => {
    const success = await copyToClipboard(message.content);
    if (success) {
      // You might want to show a toast notification here
      console.log('Message copied to clipboard');
    }
  };

  const handleSpeak = () => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(message.content);
      window.speechSynthesis.speak(utterance);
    } else {
      alert('Text-to-speech is not supported in your browser.');
    }
  };

  const isAI = message.type === 'ai';
  const isSystem = message.type === 'system';

  if (isSystem) {
    return (
      <div className="flex justify-center my-4">
        <div className="bg-gray-100 rounded-full px-4 py-2 text-sm text-gray-600">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`flex gap-3 group ${isOwn ? 'flex-row-reverse' : ''} ${className}`}
    >
      {showAvatar && (
        <Avatar
          size="sm"
          name={isAI ? 'AI Assistant' : 'User'}
          className="flex-shrink-0"
        >
          {isAI ? (
            <Bot className="h-4 w-4 text-blue-600" />
          ) : (
            <User className="h-4 w-4 text-gray-600" />
          )}
        </Avatar>
      )}

      <div className={`flex-1 max-w-lg ${isOwn ? 'text-right' : ''}`}>
        <div
          className={`
            inline-block px-4 py-2 rounded-2xl text-sm
            ${isOwn 
              ? 'bg-blue-600 text-white rounded-br-md' 
              : isAI
                ? 'bg-gray-100 text-gray-900 rounded-bl-md'
                : 'bg-gray-100 text-gray-900 rounded-bl-md'
            }
          `}
        >
          <div className="whitespace-pre-wrap break-words">
            {message.content}
          </div>
          
          {message.isEdited && (
            <div className="text-xs opacity-70 mt-1">
              edited
            </div>
          )}
        </div>

        <div className={`
          flex items-center gap-2 mt-1 text-xs text-gray-500
          ${isOwn ? 'justify-end' : 'justify-start'}
        `}>
          <span>{formatRelativeTime(message.timestamp)}</span>
          
          {/* Action buttons - only show on hover */}
          <div className="opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
            <button
              onClick={handleCopy}
              className="p-1 rounded hover:bg-gray-200 transition-colors"
              title="Copy message"
            >
              <Copy className="h-3 w-3" />
            </button>
            
            <button
              onClick={handleSpeak}
              className="p-1 rounded hover:bg-gray-200 transition-colors"
              title="Read message aloud"
            >
              <Volume2 className="h-3 w-3" />
            </button>
            
            {isOwn && onEdit && (
              <button
                onClick={() => onEdit(message.id)}
                className="p-1 rounded hover:bg-gray-200 transition-colors"
                title="Edit message"
              >
                <Edit2 className="h-3 w-3" />
              </button>
            )}
            
            {isOwn && onDelete && (
              <button
                onClick={() => onDelete(message.id)}
                className="p-1 rounded hover:bg-red-200 text-red-600 transition-colors"
                title="Delete message"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
};