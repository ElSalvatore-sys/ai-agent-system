
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import { Button, Input, Switch } from '@/components/ui';

interface AdvancedConfigPanelProps {
  isOpen: boolean;
  onClose: () => void;
  config: {
    temperature: number;
    topP: number;
    quantization: string;
    streamResponse: boolean;
  };
  onConfigChange: (newConfig: any) => void;
}

const Slider: React.FC<{ label: string, value: number, onChange: (value: number) => void, min?: number, max?: number, step?: number }> = 
  ({ label, value, onChange, min = 0, max = 1, step = 0.1 }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <label className="text-sm font-medium text-gray-700">{label}</label>
      <span className="text-sm font-semibold text-gray-900">{value.toFixed(1)}</span>
    </div>
    <Input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
    />
  </div>
);

const QuantizationSelect: React.FC<{ value: string, onChange: (value: string) => void }> = ({ value, onChange }) => (
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">Quantization</label>
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="w-full p-2 border border-gray-300 rounded-md"
    >
      <option value="FP16">FP16 (High Quality)</option>
      <option value="8-bit">8-bit (Balanced)</option>
      <option value="4-bit">4-bit (High Performance)</option>
    </select>
  </div>
);

export const AdvancedConfigPanel: React.FC<AdvancedConfigPanelProps> = ({ isOpen, onClose, config, onConfigChange }) => {
  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ x: '100%' }}
          animate={{ x: 0 }}
          exit={{ x: '100%' }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="fixed top-0 right-0 h-full w-96 bg-white/80 backdrop-blur-lg shadow-2xl z-50 p-6 border-l border-gray-200/50 flex flex-col"
        >
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-800">Advanced Configuration</h2>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-6 w-6" />
            </Button>
          </div>
          
          <div className="space-y-6 flex-1">
            <Slider 
              label="Temperature"
              value={config.temperature}
              onChange={(value) => onConfigChange({ ...config, temperature: value })}
            />
            <Slider 
              label="Top-P"
              value={config.topP}
              onChange={(value) => onConfigChange({ ...config, topP: value })}
            />
            <QuantizationSelect 
              value={config.quantization}
              onChange={(value) => onConfigChange({ ...config, quantization: value })}
            />
            <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700">Stream Response</label>
                <Switch 
                    checked={config.streamResponse}
                    onCheckedChange={(checked) => onConfigChange({ ...config, streamResponse: checked })}
                />
            </div>
          </div>

          <div className="mt-6">
            <Button onClick={onClose} className="w-full">
                Save and Close
            </Button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
