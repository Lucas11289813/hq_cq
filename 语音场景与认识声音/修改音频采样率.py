import torchaudio

waveform, sample_rate = torchaudio.load('audios/376967__kathid__goodbye-high-quality.mp3')

print(sample_rate)

torchaudio.save('./example.mp3', waveform, sample_rate=32000)
