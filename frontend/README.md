# AI Algorithms Demo - Frontend

Modern React + TypeScript frontend built with Vite for the AI Algorithms Demo platform.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **TanStack Query** - Server state management
- **Zustand** - Client state management
- **Tailwind CSS** - Styling
- **Shadcn/UI** - UI components
- **Recharts** - Data visualization
- **Axios** - HTTP client
- **Lucide React** - Icons

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

The app will be available at `http://localhost:3000`

### Build

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

## Project Structure

```
src/
├── components/
│   ├── common/          # Reusable UI components
│   ├── layout/          # Layout components (Header, Footer, etc.)
│   ├── visualizations/  # Chart and visualization components
│   └── algorithm-demos/ # Algorithm-specific demo components
├── pages/               # Route pages
├── services/            # API service layer
├── hooks/               # Custom React hooks
├── types/               # TypeScript type definitions
├── lib/                 # Utility functions
├── App.tsx             # Root component
└── main.tsx            # Entry point
```

## Features

- Interactive algorithm demonstrations
- Real-time parameter adjustment
- Data visualization with Recharts
- Responsive design with Tailwind CSS
- Type-safe API communication
- Optimistic UI updates
- Error handling and loading states

## API Integration

The frontend communicates with the backend API running on `http://localhost:8000`. The Vite dev server is configured to proxy `/api` requests to the backend.

## Available Routes

- `/` - Home page with category overview
- `/ml` - Machine Learning algorithms
- `/deep-learning` - Deep Learning algorithms
- `/nlp` - Natural Language Processing algorithms
- `/computer-vision` - Computer Vision algorithms
- `/reinforcement-learning` - Reinforcement Learning algorithms
