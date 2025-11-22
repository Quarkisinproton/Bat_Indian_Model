# Sample Data

This directory is for sample/demo data to test the pipeline.

## Expected Directory Structure

For raw audio data:
```
data/raw/
├── Species_1/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
├── Species_2/
│   ├── audio_001.wav
│   └── ...
└── Species_N/
```

For processed spectrogram data:
```
data/processed/
├── Species_1/
│   ├── audio_001.png
│   ├── audio_002.png
│   └── ...
├── Species_2/
│   ├── audio_001.png
│   └── ...
└── Species_N/
```

## Indian Bat Species Examples

Common Indian bat species that could be classified:
1. Pipistrellus coromandra (Indian Pipistrelle)
2. Rhinolophus rouxii (Rufous Horseshoe Bat)
3. Pteropus giganteus (Indian Flying Fox)
4. Hipposideros speoris (Schneider's Leaf-nosed Bat)
5. Megaderma lyra (Indian False Vampire)
6. Taphozous melanopogon (Black-bearded Tomb Bat)
7. Myotis muricola (Nepal Myotis)
8. Scotophilus heathii (Greater Asiatic Yellow Bat)
9. Miniopterus pusillus (Small Bent-wing Bat)
10. Eptesicus serotinus (Serotine Bat)

## Audio Requirements

- **Format**: WAV, MP3, FLAC, or OGG
- **Sample Rate**: Preferably 44.1 kHz or higher
- **Duration**: 3-10 seconds of bat calls
- **Quality**: Clear recordings with minimal background noise
- **Frequency Range**: Bat echolocation typically 20-120 kHz

## Recording Tips

1. Use high-quality bat detectors (e.g., frequency division, time expansion)
2. Record in areas with known bat activity
3. Capture multiple call sequences per individual
4. Note environmental conditions and location
5. Ensure proper species identification through expert verification
