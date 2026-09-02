import os
import assemblyai as aai

aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

def transcribe_audio(file_path: str) -> str:
    config = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
    )

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path, config=config)

    if transcript.status == aai.TranscriptStatus.error:
        raise Exception(f"Transcription failed: {transcript.error}")

    lines = []
    for utterance in transcript.utterances:
        speaker = utterance.speaker
        text = utterance.text
        start = utterance.start / 1000
        lines.append(f"[{start:07.2f}] Speaker {speaker}: {text}")

    return "\n".join(lines)
