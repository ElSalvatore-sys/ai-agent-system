import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { 
  BarChart3, 
  MessageSquare, 
  Settings, 
  Home,
  ChevronLeft,
  ChevronRight,
  Plus,
  Search,
  Bot,
  Code
} from 'lucide-react';
import { cn } from '@/utils/helpers';
import { Button, Input, Avatar } from '@/components/ui';
import type { ChatSession } from '@/types';

interface SidebarProps {
  sessions?: ChatSession[];
  onNewChat?: () => void;
  onSelectSession?: (sessionId: string) => void;
  className?: string;
}

export const Sidebar: React.FC<SidebarProps> = ({
  sessions = [],
  onNewChat,
  onSelectSession,
  className,
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const location = useLocation();

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Enhanced Chat', href: '/enhanced-chat', icon: MessageSquare },
    { name: 'Monitoring', href: '/monitoring', icon: BarChart3 },
    { name: 'Showcase', href: '/showcase', icon: Code },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
    { name: 'Settings', href: '/settings', icon: Settings },
  ];

  const filteredSessions = sessions.filter(session =>
    session.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div
      className={cn(
        'flex flex-col bg-card/60 backdrop-blur-xl border-r border-card-border transition-all duration-300',
        collapsed ? 'w-20' : 'w-72',
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-card-border h-20">
        {!collapsed && (
          <div className="flex items-center gap-3">
            <Bot className="h-8 w-8 text-primary" />
            <span className="text-xl font-bold text-foreground">AI Agent</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setCollapsed(!collapsed)}
          className="p-2"
        >
          {collapsed ? (
            <ChevronRight className="h-5 w-5" />
          ) : (
            <ChevronLeft className="h-5 w-5" />
          )}
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4 space-y-2">
        {navigation.map((item) => {
          const Icon = item.icon;
          const isActive = location.pathname === item.href;
          
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                'flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                'hover:bg-secondary/80',
                isActive 
                  ? 'bg-secondary text-foreground' 
                  : 'text-muted-foreground hover:text-foreground'
              )}
              title={item.name}
            >
              <Icon className="h-5 w-5 flex-shrink-0" />
              {!collapsed && <span>{item.name}</span>}
            </Link>
          );
        })}

        {/* Chat Sessions Section */}
        {location.pathname === '/chat' && !collapsed && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Chats
              </h3>
              {onNewChat && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onNewChat}
                  className="p-1"
                >
                  <Plus className="h-4 w-4" />
                </Button>
              )}
            </div>

            {/* Search */}
            {sessions.length > 5 && (
              <div className="mb-4">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search chats..."
                    value={searchQuery}
                    onChange={(val) => setSearchQuery(val)}
                    className="pl-10 py-2 text-sm"
                  />
                </div>
              </div>
            )}

            {/* Sessions List */}
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {filteredSessions.length > 0 ? (
                filteredSessions.map((session) => (
                  <button
                    key={session.id}
                    onClick={() => onSelectSession?.(session.id)}
                    className={cn(
                      'w-full text-left px-3 py-2 rounded-lg text-sm transition-colors',
                      'hover:bg-secondary/80 group',
                      session.isActive ? 'bg-secondary text-foreground' : 'text-muted-foreground hover:text-foreground'
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <MessageSquare className="h-4 w-4 flex-shrink-0" />
                      <span className="truncate">{session.title}</span>
                    </div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {session.messages.length} messages
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-sm text-muted-foreground text-center py-4">
                  {searchQuery ? 'No chats found' : 'No chats yet'}
                </div>
              )}
            </div>
          </div>
        )}
      </nav>

      {/* User Profile */}
      <div className="p-4 border-t border-card-border mt-auto">
        {!collapsed ? (
          <div className="flex items-center gap-3">
            <Avatar size="sm" name="User" isOnline />
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-foreground truncate">
                John Doe
              </p>
              <p className="text-xs text-muted-foreground truncate">
                john@example.com
              </p>
            </div>
          </div>
        ) : (
          <Avatar size="sm" name="User" isOnline />
        )}
      </div>
    </div>
  );
};
