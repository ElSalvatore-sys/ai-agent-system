import React, { useState, useCallback } from 'react';
import { Message, AIModel, ProcessingType } from '../../types';
import { Button } from '../ui/Button';
import { Textarea } from '../ui/Textarea';
import { Card } from '../ui/Card';
import { Send, Cloud, Home, Loader, CornerDownRight } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import LocalModelSelector from './LocalModelSelector'; // To switch models

// Mock data
const mockUser = { id: 'user1', name: 'You' };
const mockModels: Record<string, Pick<AIModel, 'name' | 'hostType'>> = {
  'ollama-llama2': { name: 'Llama 2', hostType: 'local' },
  'openai-gpt4': { name: 'GPT-4', hostType: 'cloud' },
};


const initialMessages: Message[] = [
    { id: '1', content: 'What is the capital of France?', senderId: 'user1', timestamp: new Date(), type: 'text' },
    { id: '2', content: 'The capital of France is Paris.', senderId: 'ai', timestamp: new Date(), type: 'ai', modelId: 'ollama-llama2', processingType: 'local' },
];


const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
    const { theme } = useTheme();
    const isUser = message.senderId === mockUser.id;
    const modelInfo = message.modelId ? mockModels[message.modelId] : null;

    return (
        <div className={`flex items-end gap-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-xl p-3 rounded-lg ${isUser ? 'bg-blue-500 text-white' : (theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200')}`}>
                <p>{message.content}</p>
                {!isUser && modelInfo && (
                    <div className="flex items-center text-xs mt-2 opacity-70">
                        {message.processingType === 'local' ? <Home size={12} className="mr-1"/> : <Cloud size={12} className="mr-1"/>}
                        <span>{modelInfo.name} ({message.processingType})</span>
                    </div>
                )}
            </div>
        </div>
    );
};


export const SmartChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentModelId, setCurrentModelId] = useState('ollama-llama2');
  const [isSwitching, setIsSwitching] = useState(false);
  const { theme } = useTheme();

  const handleSendMessage = useCallback(async () => {
    if (!input.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      senderId: mockUser.id,
      timestamp: new Date(),
      type: 'text',
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    // Mock AI response
    await new Promise(res => setTimeout(res, 1000));
    
    const model = mockModels[currentModelId];
    const aiResponse: Message = {
      id: Date.now().toString() + 'ai',
      content: `This is a mock response from ${model.name}. You asked: "${input}"`,
      senderId: 'ai',
      timestamp: new Date(),
      type: 'ai',
      modelId: currentModelId,
      processingType: model.hostType as ProcessingType,
    };
    
    setMessages(prev => [...prev, aiResponse]);
    setIsLoading(false);
  }, [input, currentModelId]);

  const handleModelSwitch = (newModelId: string) => {
    if (newModelId === currentModelId) return;

    setIsSwitching(true);
    // Simulate context transfer
    setTimeout(() => {
        setCurrentModelId(newModelId);
        setIsSwitching(false);
        const switchNotification: Message = {
            id: Date.now().toString(),
            type: 'system',
            senderId: 'system',
            timestamp: new Date(),
            content: `Switched to ${mockModels[newModelId].name}. Context has been transferred.`
        };
        setMessages(prev => [...prev, switchNotification]);
    }, 1500);
  }

  return (
    <div className={`flex flex-col h-full p-4 ${theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      <div className="flex-grow space-y-4 overflow-y-auto p-4 rounded-lg bg-white dark:bg-gray-800 shadow-inner">
        {messages.map(msg => 
            msg.type === 'system' ? 
            (<div key={msg.id} className="text-center text-xs text-gray-500 my-2">{msg.content}</div>) :
            (<MessageBubble key={msg.id} message={msg} />)
        )}
        {isLoading && <div className="flex justify-start"><Loader className="animate-spin" /></div>}
        {isSwitching && (
            <div className="text-center text-sm text-blue-500 my-2 flex items-center justify-center">
                <Loader className="animate-spin mr-2"/> Transferring context to new model...
            </div>
        )}
      </div>

      <div className="mt-4">
        <Card className={`p-2 rounded-lg ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}>
             {/* Simplified model switcher */}
            <div className="p-2 flex items-center justify-between">
                <span className="text-sm font-semibold">Current Model: {mockModels[currentModelId].name}</span>
                <div className="flex gap-2">
                    <Button size="sm" variant="outline" onClick={() => handleModelSwitch('ollama-llama2')} disabled={currentModelId === 'ollama-llama2' || isSwitching}>Use Local</Button>
                    <Button size="sm" variant="outline" onClick={() => handleModelSwitch('openai-gpt4')} disabled={currentModelId === 'openai-gpt4' || isSwitching}>Use Cloud</Button>
                </div>
            </div>
             <div className="flex gap-2 p-2">
                <Textarea
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    placeholder="Type your message..."
                    className="flex-grow"
                    onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                            e.preventDefault();
                            handleSendMessage();
                        }
                    }}
                />
                <Button onClick={handleSendMessage} disabled={isLoading || isSwitching} size="lg">
                    <Send />
                </Button>
            </div>
        </Card>
      </div>
    </div>
  );
};

export default SmartChatInterface;
