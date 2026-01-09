import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // Manual sidebar configuration for the textbook
  tutorialSidebar: [
    'intro',
    'navigation-guide',
    'module-0-intro',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS2)',
      items: [
        'module-1-ros2-architecture',
        'module-1-nodes-lifecycle',
        'module-1-topics-message-passing'
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      items: [
        'module-2-digital-twin-concepts',
        'module-2-gazebo-simulation',
        'module-2-unity-simulation',
        'module-2-robot-modeling-physics',
        'module-2-sensor-simulation',
        'module-2-virtual-testing',
        'module-2-ros2-integration',
        'module-2-exercises-assignments'
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      items: [
        'module-3-isaac-overview',
        'module-3-perception-systems',
        'module-3-control-systems',
        'module-3-decision-making',
        'module-3-sim-integration',
        'module-3-hardware-acceleration',
        'module-3-deployment-scenarios',
        'module-3-complete-system'
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA) Systems',
      items: [
        'module-4-vla-fundamentals',
        'module-4-multimodal-perception',
        'module-4-language-understanding',
        'module-4-action-planning',
        'module-4-vla-integration',
        'module-4-vla-training',
        'module-4-vla-applications',
        'module-4-vla-complete-system',
        'module-4-exercises-assignments',
        'module-4-technical-verification'
      ],
    },
    {
      type: 'category',
      label: 'Module 5: Complete Humanoid Robot Integration',
      items: [
        'module-5-integration-humanoid-control',
        'integration-validation-summary'
      ],
    },
  ],
};

export default sidebars;
