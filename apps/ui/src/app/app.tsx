import VoiceChatbot from './components/VoiceChatbot';
import './app.module.css';

export function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Research Assistant</h1>
        <p>Click the microphone to start a voice conversation</p>
      </header>
      <main className="app-main">
        <VoiceChatbot />
      </main>
    </div>
  );
}

export default App;
