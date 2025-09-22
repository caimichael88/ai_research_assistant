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


-------------

Manula test:


  🚀 Manual Testing Guide

  Step 1: Start the Server

  cd apps/api-py

  # Use test configuration
  cp .env.test .env

  # Start the server
  uvicorn src.main:app --host 127.0.0.1 --port 8001 --reload

  You should see:
  INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
  INFO:     Started reloader process
  INFO:     Started server process
  INFO:     Application startup complete.

  Step 2: Test with Browser

  Open your browser and visit:

  1. API Documentation: http://127.0.0.1:8001/docs
    - Interactive Swagger UI to test all endpoints
    - Try the endpoints directly in the browser
  2. Health Check: http://127.0.0.1:8001/v1/tts/healthz
    - Should show: {"status":"healthy","version":"1.0.0","engines":["fake"]}
  3. Voice List: http://127.0.0.1:8001/v1/tts/voices
    - Should show available voices in JSON format

  Step 3: Test with curl Commands

  Open a new terminal and try these:

  Health Check

  curl http://127.0.0.1:8001/v1/tts/healthz
  Expected: {"status":"healthy","version":"1.0.0","engines":["fake"]}

  List Voices

  curl http://127.0.0.1:8001/v1/tts/voices
  Expected: JSON with 2 voices (en_female_1, en_male_1)

  Generate Speech (Save to File)

  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{
      "text": "Hello, this is a test of the TTS system!",
      "voice_id": "en_female_1",
      "sample_rate": 22050,
      "speed": 1.0
    }' \
    --output my_test_audio.wav

  Check the Generated Audio

  # Check file size (should be > 1000 bytes)
  ls -la my_test_audio.wav

  # Play the audio (macOS)
  afplay my_test_audio.wav

  # Or check file type
  file my_test_audio.wav
  Expected: my_test_audio.wav: RIFF (little-endian) data, WAVE audio

  Test Streaming

  curl "http://127.0.0.1:8001/v1/tts/stream?text=Streaming%20test&voice_id=en_female_1" \
    --output streaming_test.wav

  Step 4: Test Error Handling

  Invalid Request (Empty Text)

  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "", "voice_id": "en_female_1"}'
  Expected: HTTP 422 error

  Invalid Speed

  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "test", "voice_id": "en_female_1", "speed": 10.0}'
  Expected: HTTP 422 error

  Step 5: Interactive Testing with Swagger UI

  1. Go to http://127.0.0.1:8001/docs
  2. Click on any endpoint (e.g., "POST /v1/tts/synthesize")
  3. Click "Try it out"
  4. Fill in the parameters:
  {
    "text": "Hello from Swagger UI!",
    "voice_id": "en_female_1",
    "sample_rate": 22050,
    "speed": 1.0,
    "format": "wav"
  }
  5. Click "Execute"
  6. Download the generated audio file

  Step 6: Test Different Parameters

  Try different combinations:

  # Different voice
  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello", "voice_id": "en_male_1"}' \
    --output male_voice.wav

  # Different speed
  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "Fast speech test", "speed": 1.5}' \
    --output fast_speech.wav

  # Longer text
  curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a longer text to test how the TTS system handles more complex 
  sentences with multiple words and punctuation."}' \
    --output long_text.wav

  What to Look For:

  ✅ Success Indicators:
  - Server starts without errors
  - Health endpoint returns "healthy"
  - Audio files are generated (> 1KB)
  - Files are valid WAV format
  - Response times < 1 second (for fake TTS)

  ❌ Failure Indicators:
  - Server won't start
  - 500 errors from endpoints
  - Empty or tiny audio files
  - Invalid file formats
  - Very slow responses (> 5 seconds)

  Troubleshooting:

  If something doesn't work:

  1. Check server logs in the terminal where you started uvicorn
  2. Verify environment with env | grep -E "(FAKE_TTS|DEVICE)"
  3. Test imports with python quick_test.py
  4. Check port availability with lsof -i :8001

  This manual testing approach gives you full control and lets you understand exactly how each
  endpoint behaves!