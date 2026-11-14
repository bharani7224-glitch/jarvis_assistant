# Jarvis Voice Assistant

An intelligent voice-controlled desktop AI assistant with support for wake word detection, Silero VAD, Speech-to-Text (STT), Text-to-Speech (TTS), and LLM integration (Gemini / Anthropic Claude) with tool execution capabilities.

## Features

- **Wake Word & Hotkey Trigger**: Activate using hotkey (`Ctrl+Space`) or optional wake word detection.
- **Voice Activity Detection (VAD)**: Intelligent recording cutoff using Silero VAD or energy-based detection.
- **Speech-to-Text (STT)**: Fast transcription via Whisper or Google Speech Recognition.
- **Text-to-Speech (TTS)**: Natural voice feedback powered by Edge TTS or offline TTS engines.
- **LLM Brain**: Supports Google Gemini or Anthropic Claude with function/tool calling.
- **Desktop Tools & Automation**: Web browsing, GUI operations, system control, and voice confirmations.

## Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/bharani7224-glitch/jarvis_assistant.git
   cd jarvis_assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to select your LLM provider (`gemini` or `claude`) and add your respective API key.

## Usage

Run the assistant:
```bash
python main.py
```
Press `Ctrl+Space` (default hotkey) or speak the wake word to activate Jarvis.
