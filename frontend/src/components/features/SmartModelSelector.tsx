
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown, Zap, DollarSign, Clock, Cpu, CheckCircle, Download, X } from 'lucide-react';
import { Button } from '@/components/ui';
import { cn } from '@/utils/helpers';
import type { AIModel } from '@/types';

interface SmartAIModel extends AIModel {
  costPerToken: number;
  averageResponseTime: number;
  performance: 'excellent' | 'good' | 'fair';
  bestFor: string[];
  color: string;
  status: 'online' | 'offline' | 'loading';
}

interface SmartAIModelSelectorProps {
  models: SmartAIModel[];
  selectedModel?: SmartAIModel;
  onModelSelect: (model: SmartAIModel) => void;
  onModelInstall: (model: SmartAIModel) => void;
  className?: string;
}

const ModelCard: React.FC<{ model: SmartAIModel, onSelect: () => void, onInstall: () => void, isSelected: boolean }> = ({ model, onSelect, onInstall, isSelected }) => {
  const isInstalled = model.status !== 'offline';

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.2 }}
      className={cn(
        'relative p-4 rounded-lg border cursor-pointer transition-all duration-300 ease-in-out',
        'bg-white/30 backdrop-blur-md shadow-lg',
        isSelected ? 'border-blue-500 ring-2 ring-blue-500/50' : 'border-gray-200/50 hover:border-blue-400/50',
      )}
      onClick={isInstalled ? onSelect : undefined}
    >
      <div className="flex justify-between items-start">
        <div>
          <h3 className="font-bold text-lg text-gray-900">{model.name}</h3>
          <p className="text-sm text-gray-600">{model.provider}</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={cn(
            'px-2 py-1 rounded-full text-xs font-semibold text-white',
            model.status === 'online' && 'bg-green-500',
            model.status === 'offline' && 'bg-gray-400',
            model.status === 'loading' && 'bg-yellow-500 animate-pulse',
          )}>
            {model.status}
          </span>
          {isSelected && <CheckCircle className="h-6 w-6 text-blue-500" />}
        </div>
      </div>

      <div className="mt-4 flex justify-between items-center text-sm">
        <div className="flex items-center gap-1 text-gray-700">
          <Zap className="h-4 w-4 text-yellow-500" />
          <span>{model.performance}</span>
        </div>
        <div className="flex items-center gap-1 text-gray-700">
          <DollarSign className="h-4 w-4 text-green-500" />
          <span>{`$${model.costPerToken.toFixed(5)}/1k`}</span>
        </div>
        <div className="flex items-center gap-1 text-gray-700">
          <Clock className="h-4 w-4 text-blue-500" />
          <span>{`${model.averageResponseTime}ms`}</span>
        </div>
      </div>

      <div className="mt-3">
          <p className="text-xs text-gray-500">Best for: {model.bestFor.join(', ')}</p>
      </div>

      {!isInstalled && (
        <div className="absolute inset-0 bg-black/30 rounded-lg flex flex-col items-center justify-center">
          <Button onClick={onInstall} size="sm" className="bg-blue-600 hover:bg-blue-700 text-white">
            <Download className="h-4 w-4 mr-2" />
            Install Model
          </Button>
          <p className="text-xs text-white/80 mt-2">Requires ~{model.size} of disk space.</p>
        </div>
      )}
    </motion.div>
  );
};


export const SmartModelSelector: React.FC<SmartAIModelSelectorProps> = ({
  models,
  selectedModel,
  onModelSelect,
  onModelInstall,
  className,
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleModelSelect = (model: SmartAIModel) => {
    onModelSelect(model);
    setIsOpen(false);
  };
  
  const handleInstallClick = (model: SmartAIModel, e: React.MouseEvent) => {
    e.stopPropagation();
    onModelInstall(model);
  };

  return (
    <div className={cn('relative w-full', className)}>
      <Button
        onClick={() => setIsOpen(true)}
        className="w-full flex justify-between items-center p-3 bg-white/50 border border-gray-300/50 rounded-lg shadow-sm"
      >
        <div className="text-left">
          <div className="font-semibold text-gray-800">{selectedModel ? selectedModel.name : 'Select a Model'}</div>
          <div className="text-sm text-gray-500">{selectedModel ? selectedModel.provider : ''}</div>
        </div>
        <ChevronDown className="h-5 w-5 text-gray-400" />
      </Button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="fixed inset-0 z-50 bg-black/40 backdrop-blur-sm flex items-center justify-center"
            onClick={() => setIsOpen(false)}
          >
            <div className="w-full max-w-4xl p-6 bg-white/70 backdrop-blur-xl rounded-2xl shadow-2xl border border-gray-200/50" onClick={(e) => e.stopPropagation()}>
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-800">Choose an AI Model</h2>
                <Button variant="ghost" size="icon" onClick={() => setIsOpen(false)}>
                  <X className="h-6 w-6" />
                </Button>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model) => (
                  <ModelCard 
                    key={model.id} 
                    model={model}
                    onSelect={() => handleModelSelect(model)}
                    onInstall={(e) => handleInstallClick(model, e)}
                    isSelected={selectedModel?.id === model.id} 
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
