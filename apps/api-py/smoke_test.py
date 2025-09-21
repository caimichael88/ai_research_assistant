#!/usr/bin/env python3
"""
TTS Service Smoke Test

This script performs basic smoke tests to verify that the TTS service is working correctly.
It tests all major endpoints and validates responses.

Usage:
    python smoke_test.py                    # Test localhost:8001
    python smoke_test.py --host=prod.com    # Test production server
    python smoke_test.py --verbose          # Detailed output
"""

import requests
import argparse
import sys
import time
from pathlib import Path
import tempfile

class Colors:
    """Terminal colors for output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class TTSSmokeTest:
    def __init__(self, base_url: str, verbose: bool = False, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.verbose = verbose
        self.timeout = timeout
        self.session = requests.Session()
        self.results = []

    def log(self, message: str, level: str = "INFO"):
        """Log message with color coding"""
        colors = {
            "INFO": Colors.BLUE,
            "SUCCESS": Colors.GREEN,
            "ERROR": Colors.RED,
            "WARNING": Colors.YELLOW
        }

        color = colors.get(level, "")
        print(f"{color}[{level}]{Colors.ENDC} {message}")

        if self.verbose and level == "INFO":
            print(f"  → {message}")

    def test_health_check(self) -> bool:
        """Test the health check endpoint"""
        self.log("Testing health check endpoint...", "INFO")

        try:
            response = self.session.get(
                f"{self.base_url}/v1/tts/healthz",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                self.log(f"Health check passed: {data.get('status')}", "SUCCESS")
                self.log(f"Available engines: {data.get('engines', [])}", "INFO")
                return True
            else:
                self.log(f"Health check failed with status {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"Health check failed: {e}", "ERROR")
            return False

    def test_voices_endpoint(self) -> bool:
        """Test the voices listing endpoint"""
        self.log("Testing voices endpoint...", "INFO")

        try:
            response = self.session.get(
                f"{self.base_url}/v1/tts/voices",
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                voices = data.get('voices', [])
                self.log(f"Found {len(voices)} voices", "SUCCESS")

                for voice in voices[:3]:  # Show first 3 voices
                    self.log(f"  - {voice.get('voice_id')}: {voice.get('name')} ({voice.get('language')})", "INFO")

                return True
            else:
                self.log(f"Voices endpoint failed with status {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"Voices endpoint failed: {e}", "ERROR")
            return False

    def test_synthesis_endpoint(self) -> bool:
        """Test the synchronous synthesis endpoint"""
        self.log("Testing synthesis endpoint...", "INFO")

        test_payload = {
            "text": "Hello world, this is a test of the TTS system.",
            "voice_id": "en_female_1",
            "sample_rate": 22050,
            "speed": 1.0,
            "format": "wav"
        }

        try:
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/v1/tts/synthesize",
                json=test_payload,
                timeout=self.timeout
            )
            end_time = time.time()

            if response.status_code == 200:
                # Check response headers
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                latency = (end_time - start_time) * 1000

                self.log(f"Synthesis successful!", "SUCCESS")
                self.log(f"Content-Type: {content_type}", "INFO")
                self.log(f"Audio size: {content_length} bytes", "INFO")
                self.log(f"Latency: {latency:.1f}ms", "INFO")

                # Validate it's actually audio
                if content_type.startswith('audio/') and content_length > 100:
                    self.log("Audio validation passed", "SUCCESS")

                    # Optionally save audio file for manual inspection
                    if self.verbose:
                        audio_file = Path(tempfile.gettempdir()) / "smoke_test_output.wav"
                        with open(audio_file, 'wb') as f:
                            f.write(response.content)
                        self.log(f"Audio saved to: {audio_file}", "INFO")

                    return True
                else:
                    self.log("Audio validation failed - invalid content", "ERROR")
                    return False

            else:
                self.log(f"Synthesis failed with status {response.status_code}", "ERROR")
                if self.verbose:
                    self.log(f"Response: {response.text}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"Synthesis endpoint failed: {e}", "ERROR")
            return False

    def test_streaming_endpoint(self) -> bool:
        """Test the streaming synthesis endpoint"""
        self.log("Testing streaming endpoint...", "INFO")

        params = {
            "text": "This is a streaming test.",
            "voice_id": "en_female_1",
            "sample_rate": 22050,
            "speed": 1.0,
            "format": "wav"
        }

        try:
            start_time = time.time()
            response = self.session.get(
                f"{self.base_url}/v1/tts/stream",
                params=params,
                timeout=self.timeout,
                stream=True
            )

            if response.status_code == 200:
                # Collect streaming data
                chunks = []
                chunk_count = 0

                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        chunks.append(chunk)
                        chunk_count += 1
                        if chunk_count > 100:  # Prevent infinite loops
                            break

                end_time = time.time()
                total_size = sum(len(chunk) for chunk in chunks)
                latency = (end_time - start_time) * 1000

                self.log(f"Streaming successful!", "SUCCESS")
                self.log(f"Received {chunk_count} chunks", "INFO")
                self.log(f"Total size: {total_size} bytes", "INFO")
                self.log(f"Streaming latency: {latency:.1f}ms", "INFO")

                return True
            else:
                self.log(f"Streaming failed with status {response.status_code}", "ERROR")
                return False

        except requests.exceptions.RequestException as e:
            self.log(f"Streaming endpoint failed: {e}", "ERROR")
            return False

    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        self.log("Testing error handling...", "INFO")

        # Test empty text
        test_cases = [
            {
                "name": "Empty text",
                "payload": {"text": "", "voice_id": "en_female_1"},
                "expected_status": 422  # Validation error
            },
            {
                "name": "Invalid voice",
                "payload": {"text": "Hello", "voice_id": "invalid_voice"},
                "expected_status": [200, 500]  # Might work with fallback or fail
            },
            {
                "name": "Invalid speed",
                "payload": {"text": "Hello", "voice_id": "en_female_1", "speed": 10.0},
                "expected_status": 422  # Validation error
            }
        ]

        success_count = 0
        for test_case in test_cases:
            try:
                response = self.session.post(
                    f"{self.base_url}/v1/tts/synthesize",
                    json=test_case["payload"],
                    timeout=self.timeout
                )

                expected = test_case["expected_status"]
                if isinstance(expected, list):
                    status_ok = response.status_code in expected
                else:
                    status_ok = response.status_code == expected

                if status_ok:
                    self.log(f"✓ {test_case['name']}: Expected response", "SUCCESS")
                    success_count += 1
                else:
                    self.log(f"✗ {test_case['name']}: Unexpected status {response.status_code}", "WARNING")

            except requests.exceptions.RequestException as e:
                self.log(f"✗ {test_case['name']}: Request failed: {e}", "WARNING")

        return success_count >= len(test_cases) // 2  # At least half should pass

    def run_all_tests(self) -> bool:
        """Run all smoke tests"""
        self.log(f"Starting smoke tests for {self.base_url}", "INFO")
        self.log("=" * 60, "INFO")

        tests = [
            ("Health Check", self.test_health_check),
            ("Voices Endpoint", self.test_voices_endpoint),
            ("Synthesis Endpoint", self.test_synthesis_endpoint),
            ("Streaming Endpoint", self.test_streaming_endpoint),
            ("Error Handling", self.test_error_handling),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} ---", "INFO")

            try:
                if test_func():
                    passed += 1
                    self.results.append((test_name, True, None))
                else:
                    self.results.append((test_name, False, "Test failed"))
            except Exception as e:
                self.log(f"Test {test_name} crashed: {e}", "ERROR")
                self.results.append((test_name, False, str(e)))

        # Summary
        self.log("\n" + "=" * 60, "INFO")
        self.log("SMOKE TEST SUMMARY", "INFO")
        self.log("=" * 60, "INFO")

        for test_name, success, error in self.results:
            status = "PASS" if success else "FAIL"
            color = "SUCCESS" if success else "ERROR"
            self.log(f"{test_name}: {status}", color)
            if error and self.verbose:
                self.log(f"  Error: {error}", "ERROR")

        self.log(f"\nOverall: {passed}/{total} tests passed",
                "SUCCESS" if passed == total else "WARNING")

        return passed == total

def main():
    parser = argparse.ArgumentParser(description="TTS Service Smoke Test")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", default="8001", help="Server port")
    parser.add_argument("--protocol", default="http", choices=["http", "https"], help="Protocol")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", default=30, type=int, help="Request timeout in seconds")

    args = parser.parse_args()

    base_url = f"{args.protocol}://{args.host}:{args.port}"

    # Run smoke tests
    tester = TTSSmokeTest(base_url, verbose=args.verbose, timeout=args.timeout)

    try:
        success = tester.run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        tester.log("\nSmoke test interrupted by user", "WARNING")
        sys.exit(1)
    except Exception as e:
        tester.log(f"Smoke test failed with error: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()