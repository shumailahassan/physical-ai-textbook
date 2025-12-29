# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a Docusaurus-based documentation website for a comprehensive textbook on Physical AI & Humanoid Robotics. The content is organized into 5 interconnected modules covering ROS2, Digital Twin, NVIDIA Isaac, Vision-Language-Action systems, and complete integration.

## Project Structure

- `docs/`: Contains all textbook content organized by modules
- `sidebars.ts`: Defines the navigation structure for the textbook
- `docusaurus.config.ts`: Main Docusaurus configuration
- `package.json`: Project dependencies and scripts
- `physical-ai-book/specs/`: Specification files for the textbook content
- `src/`: Custom source code (components, CSS, pages)
- `static/`: Static assets like images
- `blog/`: Blog posts (default Docusaurus content)

## Module Structure

The textbook is organized into 5 main modules:

1. **Module 1**: The Robotic Nervous System (ROS2) - Covers ROS2 architecture, nodes, topics, and messaging
2. **Module 2**: The Digital Twin (Gazebo & Unity) - Covers simulation environments, robot modeling, and virtual testing
3. **Module 3**: The AI-Robot Brain (NVIDIA Isaac) - Covers perception, control, decision-making, and deployment
4. **Module 4**: Vision-Language-Action (VLA) Systems - Covers multimodal perception, language understanding, and action planning
5. **Module 5**: Complete Humanoid Robot Integration - Brings all modules together

## Development Commands

### Building and Running

```bash
# Install dependencies
npm install

# Start local development server
npm run start

# Build static site
npm run build

# Serve built site locally
npm run serve

# Clean and rebuild
npm run clear

# Deploy the site
npm run deploy

# Type checking
npm run typecheck

# Additional Docusaurus commands
npm run docusaurus write-translations
npm run docusaurus write-heading-ids
npm run swizzle  # For customizing Docusaurus components
```

### File Naming Convention

- Documentation files follow the pattern: `module-X-[topic].md`
- IDs in frontmatter should match the filename without extension
- Sidebar entries reference these IDs for navigation
- Use lowercase with hyphens for file names (e.g., `module-1-ros2-architecture.md`)

## Architecture and Content Guidelines

### Documentation Structure

Each module follows this structure:
- Conceptual explanations with clear headings (H2 for sections, H3 for subsections)
- Code examples in appropriate languages (Python, C++, etc.)
- Practical exercises and assignments
- Integration with previous modules where applicable
- Cross-module learning paths and connections highlighted

### Technical Requirements

- All chapters should be 500-1000 words
- Code examples must be properly formatted with syntax highlighting
- Mathematical concepts and robotics-specific terminology should be clearly defined
- Cross-module integration points should be highlighted
- Include comparison tables and structured information where appropriate
- Use proper heading hierarchy (H1 for title, H2 for sections, H3 for subsections)

### Frontmatter Requirements

All documentation files must include:
```yaml
---
id: unique-identifier
title: Descriptive Title
sidebar_label: Short Sidebar Label
---
```

For introduction and navigation files, you may also use:
```yaml
---
sidebar_position: 1  # To control order in sidebar
---
```

## Key Technologies

- **Docusaurus 3.9.2**: Static site generator for documentation
- **Markdown/MDX**: Content format with React component support
- **React 19.0.0**: For interactive components (if needed)
- **TypeScript ~5.6.2**: For configuration files
- **Prism React Renderer**: For syntax highlighting
- **Node.js >=20.0**: Required runtime environment

## Common Development Tasks

### Adding New Content

1. Create a new markdown file in the `docs/` directory
2. Follow the naming convention `module-X-topic.md`
3. Add proper frontmatter with ID, title, and sidebar_label
4. Include the new file in `sidebars.ts` under the appropriate module
5. Use proper heading hierarchy (H1 for title, H2 for sections, H3 for subsections)
6. Ensure cross-module connections are highlighted where relevant

### Updating Navigation

Modify `sidebars.ts` to add/remove/reorganize documentation entries. The structure is hierarchical with categories and individual document references. Each module is organized as a category with its documents as items.

### Validating Changes

- Run `npm run build` to verify there are no build errors
- Use `npm run start` to preview changes locally
- Check that all internal links work correctly
- Verify proper rendering of code examples and diagrams
- Run `npm run typecheck` to ensure TypeScript configurations are valid

### Content Development Best Practices

- Maintain consistent terminology across modules
- Include cross-module references to show integration points
- Use tables for feature comparisons (as shown in the ROS2 architecture example)
- Include practical examples and use cases
- Follow the spec-driven development approach mentioned in the textbook
- Structure content to support different learning paths (beginners, AI specialists, robotics engineers)