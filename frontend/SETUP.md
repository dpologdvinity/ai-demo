# Frontend Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
cd /home/kaitlyn/git/ai-demo/frontend
npm install
```

### 2. Environment Configuration

Create a `.env` file (optional - defaults to localhost:8000):

```bash
cp .env.example .env
```

Edit `.env` if you need to change the API URL:

```
VITE_API_URL=http://localhost:8000
```

### 3. Start Development Server

```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## Project Structure Overview

```
frontend/
├── public/                 # Static assets
├── src/
│   ├── components/
│   │   ├── common/        # Reusable UI components (Button, Card, etc.)
│   │   ├── layout/        # Layout components (Header, Layout)
│   │   ├── ui/            # Shadcn/UI components
│   │   ├── visualizations/ # Chart components (to be added)
│   │   └── algorithm-demos/ # Algorithm-specific demos (to be added)
│   ├── pages/             # Route pages (Home, ML, DL, NLP, CV, RL)
│   ├── services/          # API service layer
│   ├── hooks/             # Custom React hooks
│   ├── types/             # TypeScript type definitions
│   ├── lib/               # Utility functions
│   ├── App.tsx           # Root component with routing
│   ├── main.tsx          # Application entry point
│   └── index.css         # Global styles with Tailwind
├── index.html            # HTML entry point
├── vite.config.ts        # Vite configuration
├── tailwind.config.js    # Tailwind CSS configuration
├── tsconfig.json         # TypeScript configuration
└── package.json          # Dependencies and scripts
```

## Available Scripts

- `npm run dev` - Start development server on port 3000
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

## Key Features

### 1. Routing
- React Router v6 with the following routes:
  - `/` - Home page
  - `/ml` - Machine Learning algorithms
  - `/deep-learning` - Deep Learning algorithms
  - `/nlp` - Natural Language Processing
  - `/computer-vision` - Computer Vision
  - `/reinforcement-learning` - Reinforcement Learning

### 2. State Management
- **TanStack Query** - Server state management and caching
- **Zustand** - Client state management (to be implemented as needed)

### 3. Styling
- **Tailwind CSS** - Utility-first CSS framework
- **Shadcn/UI** - Accessible component primitives
- Custom theme with CSS variables for light/dark mode support

### 4. API Integration
- Axios-based API service with interceptors
- TypeScript types for all API requests/responses
- Automatic proxy to backend (port 8000) in development

### 5. Type Safety
- Full TypeScript support
- Comprehensive type definitions in `src/types/`
- Strict type checking enabled

## Development Tips

### Adding New Components

1. Create components in appropriate directories:
   - Common UI: `src/components/common/`
   - Layout: `src/components/layout/`
   - Visualizations: `src/components/visualizations/`
   - Algorithm demos: `src/components/algorithm-demos/`

2. Use existing components as templates:
   - `Button.tsx` - Button component
   - `Card.tsx` - Card component with variants
   - `LoadingSpinner.tsx` - Loading states
   - `ErrorMessage.tsx` - Error display

### Adding New Pages

1. Create page component in `src/pages/`
2. Add route in `src/App.tsx`
3. Add navigation link in `src/components/layout/Header.tsx`

### Working with API

1. Add new methods to `src/services/api.ts`
2. Create custom hooks in `src/hooks/` using TanStack Query
3. Use the hooks in your components

Example:
```typescript
const { data, isLoading, error } = useAlgorithms(AlgorithmCategory.ML);
```

### Styling Guidelines

- Use Tailwind utility classes
- Use `cn()` utility for conditional classes
- Follow the existing color scheme (primary, secondary, etc.)
- Ensure responsive design with Tailwind's responsive modifiers

## Next Steps

### Components to Add

1. **Visualizations**
   - Line charts (Recharts)
   - Scatter plots
   - Confusion matrices
   - Neural network diagrams

2. **Algorithm Demos**
   - Parameter controls
   - Training progress displays
   - Result visualizations
   - Code examples

3. **UI Enhancements**
   - Toast notifications
   - Modal dialogs
   - Tabs for different views
   - Tooltips for explanations

### Features to Implement

1. Algorithm execution interface
2. Real-time training visualization
3. Parameter tuning controls
4. Results comparison
5. Export functionality
6. Dark mode toggle
7. Responsive mobile layout

## Troubleshooting

### Port Already in Use
If port 3000 is already in use, modify `vite.config.ts`:
```typescript
server: {
  port: 3001, // Change to available port
}
```

### API Connection Issues
- Ensure backend is running on port 8000
- Check proxy configuration in `vite.config.ts`
- Verify CORS settings on backend

### TypeScript Errors
- Run `npm run build` to check for type errors
- Ensure all imports use the `@/` alias for src files

### Style Not Applying
- Check Tailwind config includes all file paths
- Verify `index.css` is imported in `main.tsx`
- Clear browser cache and restart dev server

## Resources

- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vitejs.dev/)
- [TanStack Query](https://tanstack.com/query/latest)
- [Tailwind CSS](https://tailwindcss.com/)
- [Shadcn/UI](https://ui.shadcn.com/)
- [TypeScript](https://www.typescriptlang.org/)
