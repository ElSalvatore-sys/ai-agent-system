import React, { useState, useCallback } from 'react';
import { AIModel, InstallationStatus, QuantizationLevel } from '../../types';
import { Button } from '../ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '../ui/Card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/Select'; // Assuming a Select component exists
import { ProgressBar } from '../ui/ProgressBar'; // Assuming a ProgressBar component exists
import { HardDrive, Power, Trash2, Zap } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';

// Mock data for local models - in a real app, this would come from a WebSocket or API
const mockLocalModels: (AIModel & { size: string; downloadProgress?: number })[] = [
  {
    id: 'ollama-llama2',
    name: 'Llama 2 7B',
    provider: 'Ollama',
    description: 'A foundational, pre-trained large language model by Meta.',
    hostType: 'local',
    status: 'online',
    installationStatus: 'installed',
    capabilities: ['chat', 'reasoning'],
    isAvailable: true,
    maxTokens: 4096,
    size: '3.8 GB',
  },
  {
    id: 'huggingface-mistral',
    name: 'Mistral 7B',
    provider: 'Hugging Face',
    description: 'A powerful and efficient model, great for a variety of tasks.',
    hostType: 'local',
    status: 'offline',
    installationStatus: 'installed',
    capabilities: ['chat', 'coding'],
    isAvailable: false,
    maxTokens: 8000,
    size: '4.1 GB',
  },
  {
    id: 'huggingface-codellama',
    name: 'CodeLlama 13B',
    provider: 'Hugging Face',
    description: 'A fine-tuned version of Llama 2, specialized for code generation.',
    hostType: 'local',
    status: 'offline',
    installationStatus: 'installing',
    capabilities: ['coding'],
    isAvailable: false,
    maxTokens: 16000,
    size: '7.5 GB',
    downloadProgress: 45, // percentage
  },
  {
    id: 'ollama-phi2',
    name: 'Phi-2',
    provider: 'Ollama',
    description: 'A small, powerful model by Microsoft Research.',
    hostType: 'local',
    status: 'offline',
    installationStatus: 'not-installed',
    capabilities: ['chat', 'coding'],
    isAvailable: false,
    maxTokens: 2048,
    size: '1.5 GB',
  },
];

interface ModelManagerCardProps {
  model: AIModel & { size: string; downloadProgress?: number };
  onAction: (action: 'install' | 'uninstall' | 'load' | 'unload', modelId: string, options?: any) => void;
}

const ModelManagerCard: React.FC<ModelManagerCardProps> = ({ model, onAction }) => {
  const { theme } = useTheme();
  const [quantization, setQuantization] = useState<QuantizationLevel>('4-bit');

  const handleInstallClick = () => {
    onAction('install', model.id, { quantization });
  };

  return (
    <Card className={theme === 'dark' ? 'bg-gray-800' : 'bg-white'}>
      <CardHeader>
        <CardTitle className="text-lg">{model.name}</CardTitle>
        <p className="text-sm text-gray-500 dark:text-gray-400">{model.provider}</p>
      </CardHeader>
      <CardContent>
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center text-sm">
            <HardDrive size={16} className="mr-2" />
            <span>{model.size}</span>
          </div>
          <span className={`text-xs font-semibold px-2 py-1 rounded-full ${
            model.installationStatus === 'installed' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
          }`}>
            {model.installationStatus}
          </span>
        </div>
        
        {model.installationStatus === 'installing' && model.downloadProgress !== undefined && (
          <div className="mb-4">
            <ProgressBar value={model.downloadProgress} />
            <p className="text-center text-sm mt-1">{model.downloadProgress}% complete</p>
          </div>
        )}

        {model.installationStatus !== 'installed' && model.installationStatus !== 'installing' && (
           <div className="mb-4">
              <label className="text-sm font-medium mb-2 block">Quantization</label>
              <Select onValueChange={(value) => setQuantization(value as QuantizationLevel)} defaultValue={quantization}>
                <SelectTrigger>
                  <SelectValue placeholder="Select quantization" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="4-bit">4-bit (Recommended)</SelectItem>
                  <SelectItem value="8-bit">8-bit</SelectItem>
                  <SelectItem value="full">Full Precision</SelectItem>
                </SelectContent>
              </Select>
            </div>
        )}
        
        <div className="flex flex-col space-y-2">
           {model.installationStatus === 'not-installed' && (
             <Button onClick={handleInstallClick} variant="primary">
                Install Model
             </Button>
           )}
           {model.installationStatus === 'installed' && (
             <>
                {model.status === 'offline' ? (
                   <Button onClick={() => onAction('load', model.id)} variant="outline" className="flex items-center">
                     <Power size={16} className="mr-2"/> Load Model
                   </Button>
                ) : (
                  <Button onClick={() => onAction('unload', model.id)} variant="outline" className="flex items-center">
                    <Zap size={16} className="mr-2"/> Unload Model
                  </Button>
                )}
               <Button onClick={() => onAction('uninstall', model.id)} variant="destructive" className="flex items-center">
                 <Trash2 size={16} className="mr-2"/> Uninstall
               </Button>
             </>
           )}
           {model.installationStatus === 'installing' && (
             <Button disabled>Installing...</Button>
           )}
        </div>
      </CardContent>
    </Card>
  )
}


export const ModelInstallationManager: React.FC = () => {
  // In a real app, this state would be managed by a global store or context
  const [localModels, setLocalModels] = useState(mockLocalModels);
  
  const handleModelAction = useCallback((action: string, modelId: string, options?: any) => {
      // Here you would emit WebSocket events or call an API
      console.log(`Action: ${action}, Model: ${modelId}`, options);
      alert(`Action: ${action}, Model: ${modelId}`);

      // MOCK state update logic
      setLocalModels(prevModels => 
        prevModels.map(m => {
          if (m.id === modelId) {
            switch(action) {
              case 'load': return {...m, status: 'online'};
              case 'unload': return {...m, status: 'offline'};
              // Add more complex logic for install/uninstall if needed
            }
          }
          return m;
        })
      );

  }, []);

  return (
    <div className="p-4 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      <h2 className="text-2xl font-bold mb-4">Local Model Management</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {localModels.map(model => (
          <ModelManagerCard key={model.id} model={model} onAction={handleModelAction} />
        ))}
      </div>
    </div>
  );
};

export default ModelInstallationManager;

// NOTE: This component assumes the existence of the following UI components:
// - ../ui/Select (a dropdown component)
// - ../ui/ProgressBar (a progress bar component)
// These would need to be created as part of your UI library.
