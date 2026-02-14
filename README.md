# Voice Effect Studio (PyQt + Python)

Desktop app for real-time voice effects with:

- distortion, reverb, delay, echo (live sliders)
- selectable Windows input/output devices
- in-app input/output gain (software amplification)
- processed audio recording to WAV or M4A
- live waveform + recorded waveform view
- playback of the last saved recording with seek bar

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Usage

1. Select microphone and speaker devices.
2. Choose language (`한국어` / `English`) in the language combo.
3. Click `Start Processing`.
4. Adjust gain/effects in real time.
5. Set recording location:
   - `Browse File`: choose exact audio file path (`.wav` or `.m4a`).
   - `Choose Folder`: choose only folder, file name is auto-filled.
6. Choose monitor mode:
   - `Live Monitor (current mode)`: speaker output works during recording.
   - `Mute While Recording (playback only)`: no speaker output while recording, output only when playing saved voice.
7. Click `Start Recording` and then `Stop and Save` to write processed audio.
8. Use `Play/Pause` to play or pause, and `Stop` to return to `0s`.
9. On `Last Recorded Signal`, move the vertical cursor (`|`) to any position to seek playback.
10. While processing is running, playback will not auto-stop processing. Stop processing first if you want playback.

## External Apps (Discord / Windows Voice Recorder)

To capture processed output in another app, route the app output to a virtual audio cable device and select that virtual device as input in the external app.

If no virtual routing is set up, use the built-in recorder in this app.

## M4A Note

`imageio-ffmpeg` is used for M4A encode/decode. It is listed in `requirements.txt`.

## Build EXE (Windows)

1. Install PyInstaller:

```bash
pip install pyinstaller
```

2. Put icon file in `assets/icons` (recommended: `app.ico`).

3. Build `onedir` (recommended for this app):

```bash
pyinstaller --noconfirm --windowed --name VoiceEffectStudio ^
  --icon assets/icons/app.ico ^
  --add-data "assets;assets" ^
  main.py
```

Output folder:
- `dist/VoiceEffectStudio/VoiceEffectStudio.exe`

4. Optional: Build `onefile`:

```bash
pyinstaller --noconfirm --windowed --onefile --name VoiceEffectStudio ^
  --icon assets/icons/app.ico ^
  --add-data "assets;assets" ^
  main.py
```

Notes:
- Taskbar/title icon is set from `assets/icons` at runtime.
- If `app.ico` exists, it is preferred automatically.
