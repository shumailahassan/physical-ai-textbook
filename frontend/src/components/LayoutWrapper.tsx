import React from 'react';
import ChatWidget from '../components/ChatWidget/ChatWidget';

const LayoutWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
};

export default LayoutWrapper;