# TTS Service Smoke Tests

This document explains how to run smoke tests for the TTS (Text-to-Speech) service to verify that all components are working correctly.

## What are Smoke Tests?

Smoke tests are basic tests that verify the fundamental functionality of a system. They're designed to catch major issues quickly and ensure the service is ready for more detailed testing or production use.

## Quick Start

### 1. Automated Smoke Tests (Recommended)

```bash
# Run all smoke tests automatically
./run_smoke_tests.sh

# Quick tests only (faster)
./run_smoke_tests.sh --quick

# Verbose output for debugging
./run_smoke_tests.sh --verbose

# Keep server running after tests
./run_smoke_tests.sh --keep-server
```

### 2. Manual Smoke Tests

```bash
# Start the service manually
export $(cat .env.test | grep -v '^#' | xargs)
uvicorn src.main:app --host 127.0.0.1 --port 8001

# In another terminal, run tests
python smoke_test.py --verbose
```

## Test Coverage

The smoke tests verify the following functionality:

### ✅ Health Check
- **Endpoint**: `GET /v1/tts/healthz`
- **Tests**: Service availability, engine status
- **Expected**: HTTP 200, health status "healthy"

### ✅ Voice Listing
- **Endpoint**: `GET /v1/tts/voices`
- **Tests**: Available voices, voice metadata
- **Expected**: HTTP 200, list of available voices

### ✅ Synchronous Synthesis
- **Endpoint**: `POST /v1/tts/synthesize`
- **Tests**: Text-to-speech conversion, audio generation
- **Expected**: HTTP 200, valid audio file (WAV)

### ✅ Streaming Synthesis
- **Endpoint**: `GET /v1/tts/stream`
- **Tests**: Chunked audio streaming
- **Expected**: HTTP 200, streaming audio chunks

### ✅ Error Handling
- **Tests**: Invalid requests, error responses
- **Expected**: Proper HTTP error codes (400, 422, 500)

## Test Configuration

### Test Environment (`.env.test`)
The smoke tests use a special configuration optimized for testing:

```bash
FAKE_TTS=1                 # Use fake TTS for reliable testing
DEVICE=cpu                 # Use CPU for consistent results
WARMUP_ON_STARTUP=0        # Skip warmup for faster startup
LOUDNESS=0                 # Disable processing for speed
DEFAULT_SAMPLE_RATE=16000  # Lower sample rate for speed
```

### Test Data
The tests use the following sample inputs:

- **Health Check**: No input required
- **Voice Listing**: No input required
- **Synthesis**: "Hello world, this is a test of the TTS system."
- **Streaming**: "This is a streaming test."
- **Error Tests**: Empty text, invalid parameters

## Running Individual Tests

### Health Check Test
```bash
curl http://127.0.0.1:8001/v1/tts/healthz
```

### Voice Listing Test
```bash
curl http://127.0.0.1:8001/v1/tts/voices
```

### Synthesis Test
```bash
curl -X POST http://127.0.0.1:8001/v1/tts/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "voice_id": "en_female_1",
    "sample_rate": 22050,
    "format": "wav"
  }' \
  --output test_output.wav
```

### Streaming Test
```bash
curl "http://127.0.0.1:8001/v1/tts/stream?text=Hello%20world&voice_id=en_female_1" \
  --output test_stream.wav
```

## Expected Results

### Successful Test Run
```
[INFO] Starting smoke tests for http://127.0.0.1:8001
============================================================

--- Health Check ---
[SUCCESS] Health check passed: healthy
[INFO] Available engines: ['fake']

--- Voices Endpoint ---
[SUCCESS] Found 2 voices
[INFO]   - en_female_1: English Female 1 (en-US)
[INFO]   - en_male_1: English Male 1 (en-US)

--- Synthesis Endpoint ---
[SUCCESS] Synthesis successful!
[INFO] Content-Type: audio/wav
[INFO] Audio size: 1024 bytes
[INFO] Latency: 45.2ms
[SUCCESS] Audio validation passed

--- Streaming Endpoint ---
[SUCCESS] Streaming successful!
[INFO] Received 3 chunks
[INFO] Total size: 1024 bytes
[INFO] Streaming latency: 52.1ms

--- Error Handling ---
[SUCCESS] ✓ Empty text: Expected response
[SUCCESS] ✓ Invalid voice: Expected response
[SUCCESS] ✓ Invalid speed: Expected response

============================================================
SMOKE TEST SUMMARY
============================================================
Health Check: PASS
Voices Endpoint: PASS
Synthesis Endpoint: PASS
Streaming Endpoint: PASS
Error Handling: PASS

Overall: 5/5 tests passed
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check if port is in use
lsof -i :8001

# Check service logs
tail -f /tmp/tts_service_test.log

# Verify environment variables
env | grep -E "(FAKE_TTS|DEVICE|PORT)"
```

#### 2. Tests Fail
```bash
# Run with verbose output for details
./run_smoke_tests.sh --verbose

# Check specific endpoint manually
curl -v http://127.0.0.1:8001/v1/tts/healthz

# Verify test environment
cat .env.test
```

#### 3. Audio Issues
```bash
# Check if audio file is valid
file test_output.wav

# Play audio file (if available)
afplay test_output.wav  # macOS
aplay test_output.wav   # Linux
```

### Debug Mode
For detailed debugging, set these environment variables:

```bash
export LOG_LEVEL=DEBUG
export DEBUG_REQUESTS=1
./run_smoke_tests.sh --verbose --keep-server
```

## Integration with CI/CD

### GitHub Actions Example
```yaml
- name: Run TTS Smoke Tests
  run: |
    cd apps/api-py
    ./run_smoke_tests.sh --quick
```

### Docker Testing
```bash
# Build test image
docker build -t tts-service-test .

# Run smoke tests in container
docker run --rm tts-service-test ./run_smoke_tests.sh
```

## Performance Benchmarks

The smoke tests also provide basic performance metrics:

- **Startup Time**: How long the service takes to become ready
- **Synthesis Latency**: Time to generate audio from text
- **Streaming Latency**: Time to first audio chunk
- **Audio Quality**: Basic validation of output format

### Expected Performance (Fake TTS)
- Startup: < 5 seconds
- Synthesis: < 100ms
- Streaming: < 150ms
- Audio Size: > 100 bytes for "Hello world"

### Expected Performance (Real Models)
- Startup: 10-30 seconds (with warmup)
- Synthesis: 200ms - 2s (depending on text length and hardware)
- Streaming: 500ms - 1s (time to first chunk)
- Audio Quality: Proper WAV format, audible speech

## Next Steps

After smoke tests pass:

1. **Load Testing**: Test with multiple concurrent requests
2. **Integration Testing**: Test with real frontend applications
3. **Model Testing**: Switch to real TTS models (`FAKE_TTS=0`)
4. **Performance Testing**: Measure latency under various conditions
5. **User Acceptance Testing**: Manual testing with real use cases

----------------------------------

Michael Manual test:

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