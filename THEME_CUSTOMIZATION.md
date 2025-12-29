# Modern Theme for Physical AI & Humanoid Robotics Textbook

This theme provides a modern, visually appealing design for your Docusaurus-based textbook with the following features:

## Features

### 🎨 **Modern Design**
- Contemporary indigo/blue color scheme optimized for technical content
- Smooth animations and transitions
- Card-based layouts for modules and sections
- Modern typography with Inter and Fira Code fonts

### 🌗 **Dark/Light Mode**
- Automatic dark/light mode switching
- Respects user's system preference
- Beautiful color schemes for both modes
- Smooth transition between color schemes

### 📱 **Responsive Design**
- Fully responsive layout for mobile, tablet, and desktop
- Optimized navigation for all screen sizes
- Touch-friendly elements and interactions

### 💻 **Code Block Styling**
- Modern code block appearance with syntax highlighting
- Custom styling for inline code and code blocks
- Support for multiple programming languages

### 📚 **Navigation & Sidebar**
- Beautiful sidebar with active state indicators
- Hover effects and smooth transitions
- Collapsible sections for better organization

### 🎯 **Section Cards**
- Module cards with gradient headers
- Hover animations and depth effects
- Consistent styling across all sections

## How to Use

### Configuration
The theme is configured in `docusaurus.config.ts` with the following key settings:

- **Color Mode**: Configured with dark/light toggle enabled
- **Fonts**: Using Inter for body text and Fira Code for monospace
- **Prism**: Using vsLight for light mode and Dracula for dark mode
- **Additional Languages**: Bash, Python, C++, JSON, YAML

### Custom CSS
The `src/css/custom.css` file contains:

- Modern color palette variables
- Responsive design improvements
- Custom card and module styling
- Code block and typography enhancements
- Dark mode overrides
- Animation and transition effects

## Customization

You can customize the theme by modifying:

1. **Colors**: Update the CSS variables in `:root` and `[data-theme='dark']`
2. **Typography**: Change font families and sizes
3. **Spacing**: Adjust padding and margin variables
4. **Borders**: Modify border radius and shadow values

## Fonts Used

- **Inter**: Modern, highly readable sans-serif for body text
- **Fira Code**: Programming font with ligatures for code blocks
- Both fonts are loaded via Google Fonts CDN

## Best Practices

- Use consistent heading hierarchy (H1, H2, H3, etc.)
- Apply code blocks for technical examples
- Use admonitions (info, tip, warning, danger) appropriately
- Organize content in clear sections with descriptive titles