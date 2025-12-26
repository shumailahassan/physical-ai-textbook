<!--
SYNC IMPACT REPORT:
Version change: N/A → 1.0.0 (initial creation)
Added sections: All principles and sections as specified
Removed sections: None (new file)
Templates requiring updates: ⚠ pending - .specify/templates/plan-template.md, .specify/templates/spec-template.md, .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# AI/Spec-Driven Textbook on Physical AI & Humanoid Robotics Constitution

## Core Principles

### Accuracy
All technical concepts must be correct and verifiable. Every statement of fact, code example, and technical explanation must be accurate and based on verified sources or established practice in robotics and AI.

### Clarity
Content must be understandable by students with computer science background. Technical concepts should be explained with appropriate context, examples, and analogies that make complex topics accessible without sacrificing precision.

### Structured Learning
Each module must have chapters, headings, bullet points, and examples. Content must follow a logical progression with clear sectioning, hierarchical headings, and pedagogically sound organization that facilitates learning.

### Reusability
Content should be modular and easily updatable. Each module and chapter should be designed as a self-contained unit that can be updated, modified, or replaced without affecting other parts of the textbook.

### Consistency
Terminology, Markdown formatting, and style consistent across modules. All content must follow the same formatting standards, use consistent terminology, and maintain uniform style throughout the textbook.

### AI-Driven
Claude-generated content must follow Spec-Kit Plus workflow. All content generation must be traceable, follow the spec-plan-tasks workflow, and maintain compliance with the project's AI-driven development approach.

## Quality Standards

- Markdown output for each chapter (docs/moduleX_chapterY.md)
- Module structure:
  - Module Title
  - Chapter Number & Title
  - Introduction
  - Theory / Concepts
  - Examples / Exercises
  - Summary
- Module-wise headings: H2 for main sections (##), H3 for sub-sections (###)
- Code snippets in fenced blocks ```python or ```bash
- Reference format: Inline links or footnotes to sources (if any)
- Readability: Flesch-Kincaid grade 10-12

## Constraints

- Each module must correspond to the hackathon course modules:
  1. The Robotic Nervous System (ROS 2)
  2. The Digital Twin (Gazebo & Unity)
  3. The AI-Robot Brain (NVIDIA Isaac)
  4. Vision-Language-Action (VLA)
- Each chapter: 500-1000 words
- Markdown files stored in `docs/` folder
- Sidebar file to reflect module and chapter order
- Avoid research-paper style; focus on textbook-style teaching

## Development Workflow

- Content generation follows the Spec-Kit Plus workflow: /sp.specify → /sp.plan → /sp.tasks → /sp.implement
- Each module must have clear specifications before implementation
- Tasks must be testable and verifiable
- All changes must maintain Docusaurus compatibility
- Regular compliance reviews ensure adherence to constitution principles

## Governance

This constitution governs all aspects of the textbook development process. All content must comply with these principles and standards. Amendments to this constitution require explicit documentation of changes, approval from project stakeholders, and a migration plan for existing content. All pull requests and reviews must verify compliance with these principles. Content that does not meet these standards must be revised before acceptance.

**Version**: 1.0.0 | **Ratified**: 2025-12-26 | **Last Amended**: 2025-12-26