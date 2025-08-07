import React, { useState } from 'react';
import { MultiAIModelSelector, ChatMessage, ChatInput } from '@/components/features';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui';
import { AIModel, Message } from '@/types';
import { useAI } from '@/hooks';

export const MultiModelComparison: React.FC = () => {
  const [selectedModels, setSelectedModels] = useState<AIModel[]>([]);
  const [responses, setResponses] = useState<Record<string, Message[]>>({});
  const { sendParallelMessages, isLoading, availableModels } = useAI();

  const handleModelToggle = (model: AIModel) => {
    setSelectedModels(prev =>
      prev.some(m => m.id === model.id)
        ? prev.filter(m => m.id !== model.id)
        : [...prev, model]
    );
  };

  const handleSendMessage = async (prompt: string) => {
    if (selectedModels.length === 0) {
      alert('Please select at least one model to compare.');
      return;
    }

    setResponses({});
    const modelIds = selectedModels.map(m => m.id);
    const results = await sendParallelMessages({
      sessionId: 'multi-model-comparison',
      message: prompt,
      modelIds,
    });

    const newResponses: Record<string, Message[]> = {};
    results.forEach((res, index) => {
      const modelId = modelIds[index];
      newResponses[modelId] = [
        {
          id: `msg-${modelId}`,
          content: res.content,
          senderId: 'ai',
          timestamp: new Date(),
          type: 'ai',
        },
      ];
    });
    setResponses(newResponses);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Multi-Model Comparison</CardTitle>
      </CardHeader>
      <CardContent>
        <MultiAIModelSelector
          models={availableModels}
          selectedModels={selectedModels}
          onModelToggle={handleModelToggle}
        />
        <div className="mt-4">
          <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4">
          {selectedModels.map(model => (
            <div key={model.id} className="border rounded-lg p-4">
              <h3 className="font-semibold">{model.name}</h3>
              <div className="mt-2 space-y-2">
                {responses[model.id]?.map(msg => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}
                {isLoading && <p>Loading...</p>}
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};
