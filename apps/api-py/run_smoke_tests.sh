#!/bin/bash
# TTS Service Smoke Test Runner
#
# This script starts the TTS service and runs comprehensive smoke tests
# Usage: ./run_smoke_tests.sh [--quick] [--verbose] [--keep-server]

set -e

# Configuration
SERVICE_PORT=8001
SERVICE_HOST=127.0.0.1
PID_FILE="/tmp/tts_service_test.pid"
LOG_FILE="/tmp/tts_service_test.log"
WAIT_TIMEOUT=30

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=false
VERBOSE=false
KEEP_SERVER=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --keep-server)
            KEEP_SERVER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--quick] [--verbose] [--keep-server]"
            echo "  --quick      Run basic tests only"
            echo "  --verbose    Show detailed output"
            echo "  --keep-server Keep server running after tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Cleanup function
cleanup() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log "Stopping TTS service (PID: $PID)..."
            kill $PID
            sleep 2

            # Force kill if still running
            if ps -p $PID > /dev/null 2>&1; then
                log_warning "Force killing TTS service..."
                kill -9 $PID
            fi
        fi
        rm -f "$PID_FILE"
    fi

    rm -f "$LOG_FILE"
}

# Set up cleanup on exit
if [[ "$KEEP_SERVER" == "false" ]]; then
    trap cleanup EXIT
fi

# Check if port is already in use
if lsof -i :$SERVICE_PORT > /dev/null 2>&1; then
    log_error "Port $SERVICE_PORT is already in use!"
    log "Trying to stop existing service..."

    EXISTING_PID=$(lsof -ti :$SERVICE_PORT)
    if [[ -n "$EXISTING_PID" ]]; then
        kill $EXISTING_PID || true
        sleep 3
    fi

    if lsof -i :$SERVICE_PORT > /dev/null 2>&1; then
        log_error "Could not free port $SERVICE_PORT. Please stop the existing service."
        exit 1
    fi
fi

# Start TTS service
log "Starting TTS service for testing..."

# Set test environment
export $(cat .env.test | grep -v '^#' | xargs)

# Start the service in background
if command -v uvicorn > /dev/null 2>&1; then
    uvicorn src.main:app \
        --host $SERVICE_HOST \
        --port $SERVICE_PORT \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    SERVICE_PID=$!
elif command -v python > /dev/null 2>&1; then
    python -m uvicorn src.main:app \
        --host $SERVICE_HOST \
        --port $SERVICE_PORT \
        --log-level info \
        > "$LOG_FILE" 2>&1 &
    SERVICE_PID=$!
else
    log_error "Neither uvicorn nor python found! Please install uvicorn."
    exit 1
fi

echo $SERVICE_PID > "$PID_FILE"
log "TTS service started (PID: $SERVICE_PID)"

# Wait for service to be ready
log "Waiting for service to be ready..."
for i in $(seq 1 $WAIT_TIMEOUT); do
    if curl -s "http://$SERVICE_HOST:$SERVICE_PORT/v1/tts/healthz" > /dev/null 2>&1; then
        log_success "Service is ready!"
        break
    fi

    if [[ $i -eq $WAIT_TIMEOUT ]]; then
        log_error "Service failed to start within $WAIT_TIMEOUT seconds"
        log "Service logs:"
        cat "$LOG_FILE"
        exit 1
    fi

    sleep 1
done

# Run smoke tests
log "Running smoke tests..."

SMOKE_TEST_ARGS=""
if [[ "$VERBOSE" == "true" ]]; then
    SMOKE_TEST_ARGS="$SMOKE_TEST_ARGS --verbose"
fi

if python smoke_test.py --host $SERVICE_HOST --port $SERVICE_PORT $SMOKE_TEST_ARGS; then
    log_success "All smoke tests passed!"
    EXIT_CODE=0
else
    log_error "Some smoke tests failed!"
    EXIT_CODE=1
fi

# Additional quick tests if requested
if [[ "$QUICK_MODE" == "false" ]]; then
    log "Running additional integration tests..."

    # Test with curl commands
    log "Testing with curl..."

    # Health check
    if curl -s "http://$SERVICE_HOST:$SERVICE_PORT/v1/tts/healthz" | grep -q "healthy"; then
        log_success "✓ Health check via curl"
    else
        log_error "✗ Health check via curl failed"
        EXIT_CODE=1
    fi

    # Voices endpoint
    if curl -s "http://$SERVICE_HOST:$SERVICE_PORT/v1/tts/voices" | grep -q "voices"; then
        log_success "✓ Voices endpoint via curl"
    else
        log_error "✗ Voices endpoint via curl failed"
        EXIT_CODE=1
    fi

    # Test synthesis with curl
    if curl -s -X POST "http://$SERVICE_HOST:$SERVICE_PORT/v1/tts/synthesize" \
        -H "Content-Type: application/json" \
        -d '{"text":"Hello world","voice_id":"en_female_1"}' \
        --output /tmp/test_audio.wav; then

        if [[ -f /tmp/test_audio.wav ]] && [[ $(stat -f%z /tmp/test_audio.wav 2>/dev/null || stat -c%s /tmp/test_audio.wav) -gt 100 ]]; then
            log_success "✓ Synthesis endpoint via curl"
            rm -f /tmp/test_audio.wav
        else
            log_error "✗ Synthesis endpoint returned invalid audio"
            EXIT_CODE=1
        fi
    else
        log_error "✗ Synthesis endpoint via curl failed"
        EXIT_CODE=1
    fi
fi

# Show service logs if tests failed
if [[ $EXIT_CODE -ne 0 ]] && [[ "$VERBOSE" == "true" ]]; then
    log "Service logs (last 50 lines):"
    tail -50 "$LOG_FILE"
fi

# Keep server running if requested
if [[ "$KEEP_SERVER" == "true" ]]; then
    log_success "Smoke tests completed. Server is still running at http://$SERVICE_HOST:$SERVICE_PORT"
    log "API docs available at: http://$SERVICE_HOST:$SERVICE_PORT/docs"
    log "To stop the server: kill $(cat $PID_FILE)"

    # Don't cleanup on exit
    trap - EXIT
else
    log "Cleaning up..."
fi

exit $EXIT_CODE