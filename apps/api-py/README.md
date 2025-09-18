# AI Research Assistant Python API

FastAPI application for AI components of the research assistant.

## Requirements

### Python Installation
- **Python 3.10 or higher** is required
- **pip3** for package management

### macOS Installation
```bash
# Using Homebrew
brew install python

# Verify installation
python3 --version
pip3 --version
```

### Ubuntu/Debian Installation
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip

# Verify installation
python3 --version
pip3 --version
```

### Windows Installation
1. Download Python from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Verify in Command Prompt:
   ```cmd
   python --version
   pip --version
   ```

## Features

- AI research paper search
- Machine learning model endpoints
- **Automatic Speech Recognition (ASR)** using OpenAI Whisper
- Data processing services
- Direct system Python usage (no virtual environment complexity)

## Development

### Setup
1. **Check Python**: `nx check-python api-py` (verifies Python 3.10+ availability)
2. **Install dependencies**: `nx install api-py` (installs Python packages)
3. **Run server**: `nx serve api-py`
4. **Run tests**: `nx test api-py`

### Available Commands
- `nx check-python api-py` - Verify Python installation
- `nx install api-py` - Install dependencies with AI packages
- `nx install-prod api-py` - Install production dependencies only
- `nx serve api-py` - Start development server (port 8001)
- `nx test api-py` - Run tests with pytest
- `nx lint api-py` - Run flake8 linting
- `nx format api-py` - Format code with black
- `nx type-check api-py` - Run mypy type checking

### Project Structure
```
src/
  controllers/          # API controllers (similar to NestJS)
    ai_controller.py    # AI and ML endpoints
    asr_controller.py   # Speech recognition endpoints
    health_controller.py # Health check endpoints
  models/               # Pydantic data models
  services/             # Business logic and AI services
    asr_service.py      # Whisper ASR service
  main.py               # FastAPI application entry point
```

### ASR (Speech Recognition) Features
- **Multiple Whisper models**: tiny, base, small, medium, large
- **Multi-language support**: Auto-detection or specify language
- **Audio format support**: mp3, wav, m4a, flac, ogg, and more
- **Segmented transcription**: Get timestamped segments
- **Model caching**: Efficient memory usage with model reuse

### API Endpoints
- `POST /asr/transcribe` - Upload audio file for transcription
- `GET /asr/models` - List available Whisper models and info
- `GET /asr/health` - ASR service health check

### Deployment
This project is designed for Docker deployment where virtual environments are unnecessary. For production, use Docker containers which provide isolation without venv complexity.

### Troubleshooting

**"python3: command not found"**
- Install Python 3.10+ using the instructions above
- Ensure Python is in your system PATH

**Package installation issues**
- Use `sudo` on Linux/macOS if you get permission errors
- Consider using `--user` flag: `pip3 install --user -e .[ai,dev]`

## API Documentation

When running, visit http://localhost:8001/docs for interactive API documentation.