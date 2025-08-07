import React, { useState } from 'react';
import { Button } from '../ui/Button';
import { Textarea } from '../ui/Textarea'; // Assuming a Textarea component exists
import { Card, CardContent } from '../ui/Card';
import { ComparisonResponse, AIModel } from '../../types';
import { Send, Star, Zap, Loader } from 'lucide-react';
import { useTheme } from '../../hooks/useTheme';
import LocalModelSelector from './LocalModelSelector'; // To select models for comparison

// Mock API call
const getMockComparison = async (prompt: string, modelIds: string[]): Promise<ComparisonResponse[]> => {
  console.log('Comparing models:', modelIds);
  return new Promise(resolve => {
    setTimeout(() => {
      const responses: ComparisonResponse[] = modelIds.map(id => {
        const latency = id.includes('local') ? Math.random() * 150 + 50 : Math.random() * 500 + 300;
        return {
          modelId: id,
          modelName: id.split('-').slice(1).join(' ').replace(/\b\w/g, l => l.toUpperCase()), // e.g., 'openai-gpt4' -> 'Gpt4'
          content: `Response from ${id} for prompt "${prompt}". It has a latency of ${latency.toFixed(0)}ms. Local models are faster. This mock response is generated to showcase the comparison view.`,
          latency: parseFloat(latency.toFixed(0)),
          cost: id.includes('local') ? 0 : Math.random() * 0.05,
        };
      });
      resolve(responses);
    }, 1500);
  });
};


const ResponseCard: React.FC<{ response: ComparisonResponse }> = ({ response }) => {
  const [rating, setRating] = useState<number | undefined>(undefined);
  const { theme } = useTheme();

  return (
    <Card className={`flex flex-col h-full ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}>
      <CardContent className="p-4 flex-grow">
        <h3 className="font-bold text-lg mb-2">{response.modelName}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-300 flex-grow">{response.content}</p>
      </CardContent>
      <div className="border-t p-4 flex flex-col space-y-3 dark:border-gray-700">
        <div className="flex justify-between items-center text-sm">
          <span className="font-semibold flex items-center"><Zap size={14} className="mr-1"/>Latency:</span>
          <span>{response.latency} ms</span>
        </div>
        <div className="flex justify-between items-center text-sm">
           <span className="font-semibold">Quality:</span>
           <div className="flex space-x-1">
            {[1, 2, 3, 4, 5].map(star => (
              <Star
                key={star}
                size={18}
                className={`cursor-pointer ${rating && rating >= star ? 'text-yellow-400 fill-current' : 'text-gray-400'}`}
                onClick={() => setRating(star)}
              />
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
};


export const ModelComparisonView: React.FC = () => {
  const [prompt, setPrompt] = useState<string>('');
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>(['ollama-llama2', 'openai-gpt4']);
  const [responses, setResponses] = useState<ComparisonResponse[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const { theme } = useTheme();
  
  const handleCompare = async () => {
    if (!prompt || selectedModelIds.length < 2) {
      alert('Please enter a prompt and select at least two models to compare.');
      return;
    }
    setIsLoading(true);
    setResponses([]);
    const results = await getMockComparison(prompt, selectedModelIds);
    setResponses(results);
    setIsLoading(false);
  };

  // This is a placeholder for model selection logic.
  // In a real app, you'd integrate a more robust multi-select component.
  const toggleModelSelection = (modelId: string) => {
    setSelectedModelIds(prev =>
      prev.includes(modelId)
        ? prev.filter(id => id !== modelId)
        : [...prev, modelId]
    );
  };

  return (
    <div className={`p-4 ${theme === 'dark' ? 'bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      <h2 className="text-2xl font-bold mb-4">Model A/B Comparison</h2>
      
      {/* Placeholder for model selection UI */}
      <Card className={`mb-4 ${theme === 'dark' ? 'bg-gray-800' : 'bg-white'}`}>
        <CardContent className="p-4">
            <h3 className="font-semibold mb-2">Select models to compare (choose at least 2):</h3>
            {/* Using a simplified list for selection. Replace with a proper multi-select UI */}
            <div className="flex flex-wrap gap-2">
                {['ollama-llama2', 'huggingface-mistral', 'openai-gpt4', 'claude-opus'].map(id => (
                    <Button 
                        key={id} 
                        variant={selectedModelIds.includes(id) ? 'primary' : 'outline'}
                        onClick={() => toggleModelSelection(id)}
                    >
                        {id.split('-').slice(1).join(' ')}
                    </Button>
                ))}
            </div>
        </CardContent>
      </Card>

      <div className="flex flex-col md:flex-row gap-4 mb-4">
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your prompt here to compare model responses..."
          className="flex-grow text-base"
        />
        <Button onClick={handleCompare} disabled={isLoading || prompt.length === 0 || selectedModelIds.length < 2} size="lg">
          {isLoading ? <Loader className="animate-spin mr-2" /> : <Send className="mr-2" />}
          Compare
        </Button>
      </div>

      {isLoading && (
         <div className="text-center p-8">
            <Loader className="animate-spin mx-auto text-blue-500" size={48} />
            <p className="mt-4">Comparing models...</p>
         </div>
      )}

      {responses.length > 0 && (
        <div className={`grid grid-cols-1 md:grid-cols-2 lg:grid-cols-${responses.length} gap-4`}>
          {responses.map(response => (
            <ResponseCard key={response.modelId} response={response} />
          ))}
        </div>
      )}
    </div>
  );
};

export default ModelComparisonView;
