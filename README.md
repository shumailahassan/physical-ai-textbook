# Physical AI & Humanoid Robotics Textbook

This project implements a comprehensive textbook on Physical AI & Humanoid Robotics with a RAG (Retrieval Augmented Generation) system.

## Project Structure

```
my-website/
├── backend/                          # Backend services
│   └── main.py                      # Single-file RAG ingestion pipeline (Spec-1)
├── frontend/                         # Docusaurus frontend
│   ├── src/                         # Source code
│   ├── static/                      # Static assets
│   ├── i18n/                        # Internationalization files
│   ├── docusaurus.config.ts         # Docusaurus configuration
│   ├── package.json                 # Frontend dependencies
│   ├── sidebars.ts                  # Navigation sidebars
│   └── tsconfig.json                # TypeScript configuration
├── sp/specs/                         # Specification files
│   ├── spec-1-rag-ingestion/        # RAG ingestion specifications
│   └── spec-2-rag-retrieval/        # RAG retrieval specifications
├── docs/                            # Textbook content
├── build/                           # Built frontend files
├── rag_pipeline/                    # Optional modular RAG implementation
├── .env.example                     # Environment variables template
└── README.md                        # This file
```

## Backend (RAG Pipeline)

The backend contains the RAG ingestion pipeline:

- `backend/main.py`: Single-file implementation of Spec-1 RAG ingestion pipeline
- Processes textbook URLs, extracts content, generates embeddings with Cohere
- Stores vectors in Qdrant with metadata (URL, title, section, chunk index)
- Idempotent operation - re-running does not duplicate vectors

## Frontend

The frontend is built with Docusaurus and contains:

- All textbook content in the `docs/` directory
- Standard Docusaurus configuration and components
- Internationalization support
- Static assets

## Specifications

Specifications are organized under `sp/specs/`:

- `spec-1-rag-ingestion/`: RAG ingestion specifications
- `spec-2-rag-retrieval/`: RAG retrieval specifications

## Setup

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```

2. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. Run the RAG ingestion pipeline:
   ```bash
   cd backend
   python main.py
   ```

## Environment Variables

Required environment variables:

- `COHERE_API_KEY`: Cohere API key for embeddings
- `QDRANT_URL`: URL to Qdrant instance
- `QDRANT_API_KEY`: Qdrant API key (if using cloud)
- `QDRANT_COLLECTION_NAME`: Name of the collection to store vectors
- `TEXTBOOK_URLS`: Comma-separated list of textbook URLs to process

## Optional Modular Implementation

The `rag_pipeline/` directory contains an alternative modular implementation that can be used instead of the single-file approach in `backend/main.py`.

## Original Docusaurus Information

This website is built using [Docusaurus](https://docusaurus.io/), a modern static website generator.

### Installation

```bash
yarn
```

### Local Development

```bash
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

### Deployment

Using SSH:

```bash
USE_SSH=true yarn deploy
```

Not using SSH:

```bash
GIT_USER=<Your GitHub username> yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
