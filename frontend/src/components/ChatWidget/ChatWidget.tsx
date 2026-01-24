import React, { useState, useRef, useEffect } from 'react';
import './ChatWidget.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'auto' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      role: 'user',
      content: inputValue,
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(
  'https://shumailahassan-physical-ai-backend.hf.space/ask',
  {

        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: inputValue,
          history: messages,
        }),
      });

      if (!response.ok) {
        throw new Error(`API request failed with status ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response || 'No response received from backend.',
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error getting response from backend:', error);

      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const toggleChat = () => setIsOpen(!isOpen);

  return (
    <div className={`chat-widget ${isOpen ? 'chat-widget--open' : ''}`}>
      <button className="chat-widget__toggle-btn" onClick={toggleChat} aria-label={isOpen ? 'Close chat' : 'Open chat'}>
        {isOpen ? '✕' : '💬'}
      </button>

      {isOpen && (
        <div className="chat-widget__container">
          <div className="chat-widget__header">
            <h3>Physical AI Assistant</h3>
            <p>Ask me anything about Physical AI & Robotics</p>
          </div>

          <div className="chat-widget__messages">
            {messages.length === 0 && (
              <div className="chat-widget__welcome">
                <p>Hello! I'm your Physical AI assistant.</p>
                <p>Ask me about robotics, ROS2, NVIDIA Isaac, or any topic from the textbook.</p>
              </div>
            )}

            {messages.map((msg, index) => (
              <div
                key={index}
                className={`chat-widget__message ${msg.role === 'user' ? 'chat-widget__message--user' : 'chat-widget__message--assistant'}`}
              >
                <div className="chat-widget__message-content">{msg.content}</div>
              </div>
            ))}

            {isLoading && (
              <div className="chat-widget__message chat-widget__message--assistant">
                <div className="chat-widget__message-content">
                  <div className="chat-widget__typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <form onSubmit={handleSubmit} className="chat-widget__input-form">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about Physical AI..."
              disabled={isLoading}
              className="chat-widget__input"
            />
            <button
              type="submit"
              disabled={!inputValue.trim() || isLoading}
              className="chat-widget__send-btn"
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default ChatWidget;
