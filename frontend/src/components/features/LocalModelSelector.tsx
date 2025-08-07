import React, { useState, useMemo } from 'react';
import { AIModel, ModelProvider, ModelStatus, ModelCapability, InstallationStatus } from '../../types';
import { Button } from '../ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Sun, Zap, Cpu, Code, BrainCircuit, MessageSquare, ChevronDown, Download, Power } from 'lucide-react';
import { useTheme } from '../../context/providers/ThemeContext';

// Mock Data - Replace with actual data from your API/WebSocket
const mockModels: AIModel[] = [
  {
    id: 'ollama-llama2',
    name: 'Llama 2',
    provider: 'Ollama',
    description: 'A foundational, pre-trained large language model by Meta.',
    hostType: 'local',
    status: 'online',
    installationStatus: 'installed',
    capabilities: ['chat', 'reasoning'],
    performance: { speed: 150, quality: 4.2, cost: 0 },
    isAvailable: true,
    maxTokens: 4096,
  },
  {
    id: 'huggingface-mistral',
    name: 'Mistral-7B',
    provider: 'Hugging Face',
    description: 'A powerful and efficient model, great for a variety of tasks.',
    hostType: 'local',
    status: 'offline',
    installationStatus: 'installed',
    capabilities: ['chat', 'coding'],
    performance: { speed: 180, quality: 4.5, cost: 0 },
    isAvailable: false,
    maxTokens: 8000,
  },
  {
    id: 'huggingface-zephyr',
    name: 'Zephyr-7B',
    provider: 'Hugging Face',
    description: 'A fine-tuned version of Mistral-7B, optimized for chat.',
    hostType: 'local',
    status: 'loading',
    installationStatus: 'not-installed',
    capabilities: ['chat', 'reasoning'],
    performance: { speed: 170, quality: 4.4, cost: 0 },
    isAvailable: false,
    maxTokens: 8000,
  },
  {
    id: 'openai-gpt4',
    name: 'GPT-4',
    provider: 'OpenAI',
    description: 'The latest and most capable model from OpenAI.',
    hostType: 'cloud',
    status: 'online',
    capabilities: ['chat', 'coding', 'reasoning', 'multimodal'],
    performance: { speed: 50, quality: 4.9, cost: 30 },
    isAvailable: true,
    maxTokens: 128000,
  },
  {
    id: 'claude-opus',
    name: 'Claude 3 Opus',
    provider: 'Claude',
    description: 'Anthropic\'s most powerful model, for highly complex tasks.',
    hostType: 'cloud',
    status: 'error',
    capabilities: ['chat', 'coding', 'reasoning'],
    performance: { speed: 45, quality: 4.8, cost: 25 },
    isAvailable: false,
    maxTokens: 200000,
  },
   {
    id: 'gemini-1.5-pro',
    name: 'Gemini 1.5 Pro',
    provider: 'Gemini',
    description: 'Google\'s next-generation model, with a breakthrough 1M token context window.',
    hostType: 'cloud',
    status: 'online',
    capabilities: ['chat', 'coding', 'reasoning', 'multimodal'],
    performance: { speed: 60, quality: 4.7, cost: 7 },
    isAvailable: true,
    maxTokens: 1000000,
  },
];

// Helper components for visual elements
const StatusIndicator: React.FC<{ status: ModelStatus }> = ({ status }) => {
  const statusConfig = {
    online: { color: 'bg-green-500', text: 'Online' },
    offline: { color: 'bg-gray-500', text: 'Offline' },
    loading: { color: 'bg-yellow-500', text: 'Loading' },
    error: { color: 'bg-red-500', text: 'Error' },
  };
  const { color, text } = statusConfig[status];
  return (
    <div className="flex items-center">
      <span className={`w-3 h-3 rounded-full ${color} mr-2`}></span>
      <span className="text-sm">{text}</span>
    </div>
  );
};

const CapabilityBadge: React.FC<{ capability: ModelCapability }> = ({ capability }) => {
    const capabilityConfig = {
        coding: { icon: <Code size={12} />, text: 'Coding' },
        reasoning: { icon: <BrainCircuit size={12} />, text: 'Reasoning' },
        multimodal: { icon: <Sun size={12} />, text: 'Multimodal' },
        chat: { icon: <MessageSquare size={12} />, text: 'Chat' },
    };
    const { icon, text } = capabilityConfig[capability];
    return (
        <span className="flex items-center bg-gray-200 dark:bg-gray-700 text-xs font-medium px-2 py-1 rounded-full">
            {icon}
            <span className="ml-1">{text}</span>
        </span>
    );
};


const ModelCard: React.FC<{ model: AIModel; onSelect: (id: string) => void; selected: boolean }> = ({ model, onSelect, selected }) => {
  const { theme } = useTheme();
  
  const handleInstall = (e: React.MouseEvent) => {
    e.stopPropagation();
    // Add installation logic here
    alert(`Installing ${model.name}`);
  };

  return (
    <Card 
      className={`cursor-pointer transition-all duration-300 ${selected ? 'ring-2 ring-blue-500 shadow-lg' : 'hover:shadow-md'} ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}
      onClick={() => onSelect(model.id)}
    >
      <CardHeader>
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-lg font-bold">{model.name}</CardTitle>
            <p className="text-sm text-gray-500 dark:text-gray-400">{model.provider} - {model.hostType}</p>
          </div>
          <StatusIndicator status={model.status} />
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm mb-4 h-10">{model.description}</p>
        <div className="flex flex-wrap gap-2 mb-4">
          {model.capabilities.map(c => <CapabilityBadge key={c} capability={c} />)}
        </div>
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="font-semibold flex items-center"><Zap size={14} className="mr-1"/>Speed:</span>
            <span>{model.performance?.speed} t/s</span>
          </div>
          <div className="flex justify-between text-sm mb-2">
            <span className="font-semibold flex items-center"><BrainCircuit size={14} className="mr-1"/>Quality:</span>
            <span>{model.performance?.quality}/5.0</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="font-semibold flex items-center"><Cpu size={14} className="mr-1"/>Cost:</span>
            <span>${model.performance?.cost}/1M tokens</span>
          </div>
        </div>
        {model.hostType === 'local' && (
          <div className="mt-4">
            {model.installationStatus === 'not-installed' && (
              <Button onClick={handleInstall} className="w-full" size="sm" variant="outline">
                <Download size={16} className="mr-2" />
                Install Model
              </Button>
            )}
             {model.installationStatus === 'installed' && model.status === 'offline' && (
              <Button onClick={(e) => { e.stopPropagation(); alert('Loading model...');}} className="w-full" size="sm" variant="outline">
                <Power size={16} className="mr-2" />
                Load Model
              </Button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};


export const LocalModelSelector: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(mockModels.find(m => m.status === 'online')?.id || null);
  const [filter, setFilter] = useState<'all' | 'local' | 'cloud'>('all');
  
  const filteredModels = useMemo(() => {
    if (filter === 'all') return mockModels;
    return mockModels.filter(m => m.hostType === filter);
  }, [filter]);

  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <div className="mb-4">
        <h2 className="text-2xl font-bold mb-2">Smart Model Selector</h2>
        <div className="flex space-x-2">
          <Button variant={filter === 'all' ? 'primary' : 'outline'} onClick={() => setFilter('all')}>All</Button>
          <Button variant={filter === 'local' ? 'primary' : 'outline'} onClick={() => setFilter('local')}>Local</Button>
          <Button variant={filter === 'cloud' ? 'primary' : 'outline'} onClick={() => setFilter('cloud')}>Cloud</Button>
        </div>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filteredModels.map(model => (
          <ModelCard 
            key={model.id} 
            model={model} 
            onSelect={setSelectedModel}
            selected={selectedModel === model.id}
          />
        ))}
      </div>
    </div>
  );
};

export default LocalModelSelector;
