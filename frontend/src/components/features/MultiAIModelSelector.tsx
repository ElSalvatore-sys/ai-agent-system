import React from 'react';
import { AIModel } from '@/types';
import { Button } from '@/components/ui';
import { cn } from '@/utils/helpers';

interface MultiAIModelSelectorProps {
  models: AIModel[];
  selectedModels: AIModel[];
  onModelToggle: (model: AIModel) => void;
}

export const MultiAIModelSelector: React.FC<MultiAIModelSelectorProps> = ({ models, selectedModels, onModelToggle }) => {
  return (
    <div className="flex flex-wrap gap-2">
      {models.map(model => {
        const isSelected = selectedModels.some(m => m.id === model.id);
        return (
          <Button
            key={model.id}
            variant={isSelected ? 'primary' : 'outline'}
            onClick={() => onModelToggle(model)}
            className={cn('transition-all', isSelected && 'ring-2 ring-blue-500')}
          >
            {model.name}
          </Button>
        );
      })}
    </div>
  );
};
