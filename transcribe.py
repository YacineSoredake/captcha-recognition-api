from transformers import pipeline

print("Loading model... (first run downloads ~1.2GB, be patient)")

pipe = pipeline(
    "automatic-speech-recognition",
    model="nvidia/nemotron-3.5-asr-streaming-0.6b",
    device="cpu"
)

result = pipe("sample.flac")
print("\nTranscription:", result["text"])