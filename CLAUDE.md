# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Research Assistant project built with NX workspace and NestJS backend. The project follows NX monorepo conventions for scalable development.

## Technology Stack

- **Workspace**: NX (v21.5.2) - Monorepo management and build system
- **Backend**:
  - NestJS - Node.js framework for web API and services
  - FastAPI - Python framework for AI components and ML services
- **Languages**: TypeScript, Python 3.10+
- **Package Managers**: npm (Node.js), pip/pyproject.toml (Python)
- **Testing**: Jest (TypeScript), pytest (Python)

## Project Structure

```
apps/
  api/                    # NestJS backend application
    src/
      app/                # Main application module
        app.controller.ts # Main controller
        app.module.ts     # Root module
        app.service.ts    # Main service
      main.ts             # Application entry point
  api-e2e/               # End-to-end tests for the API
  api-py/                # Python FastAPI for AI components
    src/
      controllers/       # API controllers (like NestJS)
        ai_controller.py # AI-related endpoints
        asr_controller.py # Speech recognition endpoints
        health_controller.py # Health check endpoints
      models/            # Pydantic data models
      services/          # AI service implementations
        asr_service.py   # Whisper ASR service
      main.py            # FastAPI application entry point
    tests/               # Python tests
    pyproject.toml       # Python project configuration
libs/                    # Shared libraries (empty for now)
```

## Common Commands

### Development
#### NestJS API (TypeScript)
- `nx serve api` - Start the NestJS API in development mode
- `nx build api` - Build the NestJS API for production
- `nx test api` - Run unit tests for the NestJS API
- `nx e2e api-e2e` - Run end-to-end tests

#### Python AI API
- `nx check-python api-py` - Verify Python 3.10+ installation
- `nx install api-py` - Install Python dependencies with AI packages
- `nx serve api-py` - Start the Python FastAPI server (port 8001)
- `nx test api-py` - Run Python tests with pytest
- `nx lint api-py` - Run flake8 linting
- `nx format api-py` - Format code with black
- `nx type-check api-py` - Run mypy type checking

### Project Management
- `nx show projects` - List all projects in the workspace
- `nx show project api` - Show details about the API project
- `nx graph` - Show project dependency graph

### Code Generation
#### NestJS
- `nx g @nx/nest:resource [name]` - Generate a new NestJS resource (controller, service, module)
- `nx g @nx/nest:service [name]` - Generate a new service
- `nx g @nx/nest:controller [name]` - Generate a new controller

#### Python
- Python files should be created manually following FastAPI patterns
- Use the controller structure in `apps/api-py/src/controllers/`
- Controllers include prefix and tags for automatic OpenAPI documentation

## Architecture Notes

- The workspace follows NX conventions with applications in `apps/` and shared libraries in `libs/`
- **NestJS API** (port 3000): Handles web services, user management, and general backend operations
- **Python FastAPI** (port 8001): Dedicated to AI/ML operations, research queries, and data processing
- Both APIs can communicate with each other and share data through HTTP calls
- Python API uses modern `pyproject.toml` for dependency management
- TypeScript API uses webpack with NX build system

## Development Workflow

### Initial Setup
1. **Ensure Python 3.10+ is installed**:
   - macOS: `brew install python`
   - Ubuntu: `sudo apt install python3 python3-pip`
   - Windows: Download from [python.org](https://www.python.org/downloads/)

2. **Verify Python installation**: `nx check-python api-py`

3. **Install Python dependencies**: `nx install api-py`

### Daily Development
1. Use NX generators to create new features/modules
2. Place shared functionality in `libs/` directory
3. Keep applications focused and lightweight in `apps/` directory
4. Run tests frequently using the NX commands
5. Use the NX cache for faster builds and tests

### Python Development Notes
- Uses system Python directly (no virtual environment complexity)
- Dependencies are managed via `pyproject.toml`
- AI packages (transformers, torch, etc.) are installed to system Python
- Docker will provide isolation for production deployment