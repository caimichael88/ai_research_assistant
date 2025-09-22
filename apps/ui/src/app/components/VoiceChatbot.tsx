import React, { useState, useRef, useCallback } from 'react';
import './VoiceChatbot.css';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

const VoiceChatbot: React.FC = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [useLangGraph, setUseLangGraph] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' });
        await processAudio(audioBlob);

        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      setError(null);
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to access microphone. Please ensure microphone permissions are granted.');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsProcessing(true);
    }
  }, [isRecording]);

  const processAudio = async (audioBlob: Blob) => {
    try {
      const formData = new FormData();
      const uniqueFilename = `recording_${Date.now()}_${Math.random().toString(36).substring(2, 11)}.wav`;
      formData.append('file', audioBlob, uniqueFilename);

      // Send audio to /voice/agent endpoint for LangGraph processing
      const voiceResponse = await fetch('http://localhost:8001/voice/agent', {
        method: 'POST',
        body: formData,
      });

      if (!voiceResponse.ok) {
        throw new Error('Failed to process voice conversation');
      }

      // The response should be an audio file
      const audioData = await voiceResponse.blob();

      // Add user message
      const userMessageObj: Message = {
        id: Date.now().toString(),
        type: 'user',
        content: "Voice message processed via LangGraph Agent",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, userMessageObj]);

      // Add AI response message
      const aiMessageObj: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "AI response generated via LangGraph Agent (audio)",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, aiMessageObj]);

      // Play the AI response audio
      const audioUrl = URL.createObjectURL(audioData);
      const audio = new Audio(audioUrl);

      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };

      audio.play().catch(err => {
        console.error('Error playing audio:', err);
        setError('Failed to play audio response');
      });

    } catch (err) {
      console.error('Error processing audio:', err);
      setError('Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const clearConversation = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="voice-chatbot">
      <div className="conversation-display">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <p>Welcome! Click the microphone button to start your conversation.</p>
          </div>
        ) : (
          <div className="conversation-history">
            {messages.map((message) => (
              <div key={message.id} className="conversation-turn">
                <div className={`${message.type}-message`}>
                  <strong>{message.type === 'user' ? 'You:' : 'Assistant:'}</strong> {message.content}
                </div>
                <div className="timestamp">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      <div className="controls">
        <button
          className={`record-button ${isRecording ? 'recording' : ''} ${isProcessing ? 'processing' : ''}`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={isProcessing}
        >
          {isProcessing ? (
            <span>Processing...</span>
          ) : isRecording ? (
            <span>🔴 Stop Recording</span>
          ) : (
            <span>🎤 Start Recording</span>
          )}
        </button>

        {messages.length > 0 && (
          <button
            className="clear-button"
            onClick={clearConversation}
            disabled={isRecording || isProcessing}
          >
            Clear Chat
          </button>
        )}
      </div>

      <div className="status">
        {isRecording && <p>🎙️ Recording... Click "Stop Recording" when done</p>}
        {isProcessing && <p>⏳ Processing your request...</p>}
        {!isRecording && !isProcessing && <p>💬 Ready for conversation</p>}
      </div>
    </div>
  );
};

export default VoiceChatbot;