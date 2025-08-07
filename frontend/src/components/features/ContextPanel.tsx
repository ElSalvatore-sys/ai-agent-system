import React from 'react';
import { cn } from '@/utils/helpers';
import { MultiAIModelSelector } from '@/components/features/MultiAIModelSelector';
import { CostTracker } from '@/components/features/CostTracker';
import type { AIModel } from '@/types';

interface ContextPanelProps {
  /** currently selected model */
  model?: AIModel;
  /** list of available models */
  models?: AIModel[];
  /** total cost for the current conversation */
  totalCost?: number;
  /** callback when the user selects a model */
  onModelSelect?: (model: AIModel) => void;
  /** additional css */
  className?: string;
  /** whether the panel is shown */
  isOpen?: boolean;
}

/**
 * Small, self-contained right-hand side panel that surfaces contextual
 * utilities (model picker, cost overview, etc.).
 * Part of the progressive-disclosure IA – can be toggled per user.
 */
export const ContextPanel: React.FC<ContextPanelProps> = ({
  model,
  models = [],
  totalCost = 0,
  onModelSelect,
  className,
  isOpen = true,
}) => {
  if (!isOpen) return null;

  return (
    <aside
      className={cn(
        'w-80 shrink-0 border-l border-card-border p-4 space-y-6 bg-card/60 backdrop-blur-xl',
        className,
      )}
    >
      {/* Model selector */}
      {models?.length > 0 && (
        <section>
          <h3 className="text-sm font-medium mb-2 text-muted-foreground uppercase tracking-wider">
            Model
          </h3>
          <MultiAIModelSelector
            models={models}
            selectedModel={model}
            onModelSelect={onModelSelect ?? (() => undefined)}
            showCosts
            showMetrics
          />
        </section>
      )}

      {/* Cost overview */}
      <section>
        <h3 className="text-sm font-medium mb-2 text-muted-foreground uppercase tracking-wider">
          Cost
        </h3>
        <CostTracker cost={totalCost} className="w-full" />
      </section>
    </aside>
  );
};
