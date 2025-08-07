import React, { useState } from 'react';
import { Plus, Bot, Settings, Play, Pause, Trash2, Edit } from 'lucide-react';
import { Button, Card, Modal, Input, Textarea } from '@/components/ui';
import { cn } from '@/utils/helpers';

interface Agent {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'paused' | 'stopped';
  type: 'chat' | 'task' | 'analysis';
  model: string;
  createdAt: Date;
  lastActive: Date;
  totalRuns: number;
  successRate: number;
}

const mockAgents: Agent[] = [
  {
    id: '1',
    name: 'Code Assistant',
    description: 'Helps with coding tasks, debugging, and code reviews',
    status: 'active',
    type: 'chat',
    model: 'claude-3.5-sonnet',
    createdAt: new Date('2024-01-15'),
    lastActive: new Date(),
    totalRuns: 247,
    successRate: 96.8,
  },
  {
    id: '2',
    name: 'Data Analyzer',
    description: 'Analyzes datasets and generates insights',
    status: 'active',
    type: 'analysis',
    model: 'gpt-4',
    createdAt: new Date('2024-02-01'),
    lastActive: new Date(Date.now() - 1000 * 60 * 30),
    totalRuns: 89,
    successRate: 94.4,
  },
  {
    id: '3',
    name: 'Task Scheduler',
    description: 'Manages and schedules automated tasks',
    status: 'paused',
    type: 'task',
    model: 'gemini-pro',
    createdAt: new Date('2024-01-20'),
    lastActive: new Date(Date.now() - 1000 * 60 * 60 * 2),
    totalRuns: 156,
    successRate: 91.7,
  },
];

export const Agents: React.FC = () => {
  const [agents, setAgents] = useState<Agent[]>(mockAgents);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [newAgent, setNewAgent] = useState({
    name: '',
    description: '',
    type: 'chat' as Agent['type'],
    model: 'claude-3.5-sonnet',
  });

  const getStatusColor = (status: Agent['status']) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-100';
      case 'paused':
        return 'text-yellow-600 bg-yellow-100';
      case 'stopped':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: Agent['type']) => {
    switch (type) {
      case 'chat':
        return '💬';
      case 'task':
        return '⚙️';
      case 'analysis':
        return '📊';
      default:
        return '🤖';
    }
  };

  const toggleAgentStatus = (agentId: string) => {
    setAgents(prev => prev.map(agent => 
      agent.id === agentId 
        ? { ...agent, status: agent.status === 'active' ? 'paused' : 'active' }
        : agent
    ));
  };

  const deleteAgent = (agentId: string) => {
    setAgents(prev => prev.filter(agent => agent.id !== agentId));
  };

  const createAgent = () => {
    const agent: Agent = {
      id: Date.now().toString(),
      ...newAgent,
      status: 'active',
      createdAt: new Date(),
      lastActive: new Date(),
      totalRuns: 0,
      successRate: 100,
    };
    
    setAgents(prev => [...prev, agent]);
    setNewAgent({
      name: '',
      description: '',
      type: 'chat',
      model: 'claude-3.5-sonnet',
    });
    setShowCreateModal(false);
  };

  const formatTimeAgo = (date: Date) => {
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-foreground">AI Agents</h1>
          <p className="text-muted-foreground mt-2">
            Manage and monitor your AI agents
          </p>
        </div>
        <Button onClick={() => setShowCreateModal(true)}>
          <Plus className="h-4 w-4 mr-2" />
          Create Agent
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Agents</p>
              <p className="text-2xl font-bold text-foreground">{agents.length}</p>
            </div>
            <Bot className="h-8 w-8 text-primary" />
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Active</p>
              <p className="text-2xl font-bold text-green-600">
                {agents.filter(a => a.status === 'active').length}
              </p>
            </div>
            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
            </div>
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Total Runs</p>
              <p className="text-2xl font-bold text-foreground">
                {agents.reduce((sum, agent) => sum + agent.totalRuns, 0)}
              </p>
            </div>
            <Play className="h-8 w-8 text-blue-500" />
          </div>
        </Card>
        
        <Card className="p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-muted-foreground">Avg Success</p>
              <p className="text-2xl font-bold text-foreground">
                {(agents.reduce((sum, agent) => sum + agent.successRate, 0) / agents.length).toFixed(1)}%
              </p>
            </div>
            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
              <span className="text-xs font-bold text-green-600">✓</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {agents.map((agent) => (
          <Card key={agent.id} className="p-6 hover:shadow-lg transition-shadow">
            {/* Agent Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="text-2xl">{getTypeIcon(agent.type)}</div>
                <div>
                  <h3 className="font-semibold text-foreground">{agent.name}</h3>
                  <span className={cn(
                    'inline-flex items-center px-2 py-1 rounded-full text-xs font-medium',
                    getStatusColor(agent.status)
                  )}>
                    {agent.status}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleAgentStatus(agent.id)}
                  className="p-2"
                >
                  {agent.status === 'active' ? (
                    <Pause className="h-4 w-4" />
                  ) : (
                    <Play className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedAgent(agent)}
                  className="p-2"
                >
                  <Settings className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => deleteAgent(agent.id)}
                  className="p-2 text-red-500 hover:text-red-700"
                >
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>

            {/* Agent Description */}
            <p className="text-sm text-muted-foreground mb-4">
              {agent.description}
            </p>

            {/* Agent Details */}
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Model:</span>
                <span className="text-foreground font-medium">{agent.model}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Total Runs:</span>
                <span className="text-foreground font-medium">{agent.totalRuns}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Success Rate:</span>
                <span className="text-foreground font-medium">{agent.successRate}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Last Active:</span>
                <span className="text-foreground font-medium">
                  {formatTimeAgo(agent.lastActive)}
                </span>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Create Agent Modal */}
      <Modal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        title="Create New Agent"
      >
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Agent Name
            </label>
            <Input
              value={newAgent.name}
              onChange={(value) => setNewAgent(prev => ({ ...prev, name: value }))}
              placeholder="Enter agent name"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Description
            </label>
            <Textarea
              value={newAgent.description}
              onChange={(value) => setNewAgent(prev => ({ ...prev, description: value }))}
              placeholder="Describe what this agent does"
              rows={3}
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              Type
            </label>
            <select
              value={newAgent.type}
              onChange={(e) => setNewAgent(prev => ({ ...prev, type: e.target.value as Agent['type'] }))}
              className="w-full px-3 py-2 border border-input rounded-md bg-background"
            >
              <option value="chat">Chat Assistant</option>
              <option value="task">Task Automation</option>
              <option value="analysis">Data Analysis</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-foreground mb-2">
              AI Model
            </label>
            <select
              value={newAgent.model}
              onChange={(e) => setNewAgent(prev => ({ ...prev, model: e.target.value }))}
              className="w-full px-3 py-2 border border-input rounded-md bg-background"
            >
              <option value="claude-3.5-sonnet">Claude 3.5 Sonnet</option>
              <option value="gpt-4">GPT-4</option>
              <option value="gemini-pro">Gemini Pro</option>
              <option value="llama-2">Llama 2</option>
            </select>
          </div>
          
          <div className="flex gap-3 pt-4">
            <Button variant="outline" onClick={() => setShowCreateModal(false)}>
              Cancel
            </Button>
            <Button onClick={createAgent} disabled={!newAgent.name.trim()}>
              Create Agent
            </Button>
          </div>
        </div>
      </Modal>

      {/* Agent Settings Modal */}
      {selectedAgent && (
        <Modal
          isOpen={!!selectedAgent}
          onClose={() => setSelectedAgent(null)}
          title={`${selectedAgent.name} Settings`}
        >
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Created:</span>
                <p className="font-medium">{selectedAgent.createdAt.toLocaleDateString()}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Type:</span>
                <p className="font-medium capitalize">{selectedAgent.type}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Model:</span>
                <p className="font-medium">{selectedAgent.model}</p>
              </div>
              <div>
                <span className="text-muted-foreground">Status:</span>
                <p className="font-medium capitalize">{selectedAgent.status}</p>
              </div>
            </div>
            
            <div className="pt-4 border-t">
              <h4 className="font-medium mb-2">Performance Metrics</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-muted-foreground">Total Runs:</span>
                  <p className="font-bold text-lg">{selectedAgent.totalRuns}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Success Rate:</span>
                  <p className="font-bold text-lg text-green-600">{selectedAgent.successRate}%</p>
                </div>
              </div>
            </div>
            
            <div className="flex gap-3 pt-4">
              <Button variant="outline" onClick={() => setSelectedAgent(null)}>
                Close
              </Button>
              <Button>
                <Edit className="h-4 w-4 mr-2" />
                Edit Agent
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};