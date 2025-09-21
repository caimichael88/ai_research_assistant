#!/usr/bin/env python3
"""
Quick smoke test without starting a full server
Tests the TTS components directly and via HTTP requests
"""
import os
import sys
import requests
import time
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set test environment
os.environ.update({
    'FAKE_TTS': '1',
    'DEVICE': 'cpu',
    'WARMUP_ON_STARTUP': '0',
    'LOUDNESS': '0',
    'LOG_LEVEL': 'INFO'
})

def test_imports():
    """Test that all components can be imported"""
    print("🔍 Testing imports...")

    try:
        from components.ports import SynthesisRequest, AudioFormat, TTSEngine
        print("✅ Components imported successfully")

        from services.tts_service import get_tts_service
        print("✅ TTS service imported successfully")

        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_service_directly():
    """Test TTS service components directly"""
    print("\n🧪 Testing TTS service directly...")

    try:
        from components.ports import SynthesisRequest, AudioFormat
        from services.tts_service import get_tts_service

        # Get service
        service = get_tts_service()
        print(f"✅ Service created: {service.id}")

        # Test synthesis
        req = SynthesisRequest(
            text="Hello world, this is a test",
            voice_id="en_female_1",
            sample_rate=22050,
            fmt=AudioFormat.wav
        )

        start_time = time.time()
        result = service.synthesize_sync(req)
        end_time = time.time()

        print(f"✅ Synthesis successful:")
        print(f"   - Audio size: {len(result.audio)} bytes")
        print(f"   - Media type: {result.media_type}")
        print(f"   - Model: {result.model_header}")
        print(f"   - Latency: {(end_time - start_time) * 1000:.1f}ms")

        # Save test audio
        test_file = Path(tempfile.gettempdir()) / "quick_test_output.wav"
        with open(test_file, 'wb') as f:
            f.write(result.audio)
        print(f"✅ Audio saved to: {test_file}")

        return True

    except Exception as e:
        print(f"❌ Service test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_simple_server():
    """Start a minimal server and test HTTP endpoints"""
    print("\n🌐 Testing with simple HTTP server...")

    try:
        from fastapi import FastAPI
        from components.ports import SynthesisRequest, AudioFormat
        from services.tts_service import get_tts_service
        import uvicorn
        import threading

        # Create minimal app
        app = FastAPI()

        @app.get("/healthz")
        async def health():
            service = get_tts_service()
            return {"status": "healthy", "engine": service.id}

        @app.post("/synthesize")
        async def synthesize(request: dict):
            service = get_tts_service()
            req = SynthesisRequest(
                text=request.get("text", "Hello test"),
                voice_id=request.get("voice_id", "en_female_1")
            )
            result = service.synthesize_sync(req)
            return {
                "success": True,
                "audio_size": len(result.audio),
                "media_type": result.media_type
            }

        # Start server in thread
        def run_server():
            uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

        # Wait for server to start
        time.sleep(3)

        # Test endpoints
        base_url = "http://127.0.0.1:8002"

        # Health check
        resp = requests.get(f"{base_url}/healthz", timeout=5)
        if resp.status_code == 200:
            print(f"✅ Health check: {resp.json()}")
        else:
            print(f"❌ Health check failed: {resp.status_code}")
            return False

        # Synthesis test
        resp = requests.post(f"{base_url}/synthesize",
                           json={"text": "Hello HTTP test"},
                           timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ HTTP synthesis: {data['audio_size']} bytes, {data['media_type']}")
        else:
            print(f"❌ HTTP synthesis failed: {resp.status_code}")
            return False

        return True

    except Exception as e:
        print(f"❌ HTTP test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run quick smoke tests"""
    print("🚀 TTS Quick Smoke Test")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Direct Service Test", test_service_directly),
        ("HTTP Server Test", test_with_simple_server),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! TTS service is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)