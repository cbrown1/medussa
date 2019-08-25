# Medussa: A cross-platform high-level audio library for Python

## Usage Examples

```python
import medussa

# Open default audio device
d = medussa.open_default_device()

# Load a sound file into memory, then play it looped:
data,fs = medussa.read_file('/path/to/file2.wav')
s1 = d.open_array(data,fs)
s1.is_looping = True
s1.play()

# Try out the blocking functions; be careful with long duration soundfiles
# Read a soundfile into a numpy array
buff,fs = medussa.read_file('/path/to/shortfile.wav')

# Now play it on the default output device
medussa.play_array(buff, fs)

# Stream the same file from disk; also blocks
medussa.play_file('/path/to/shortfile.wav')

# Create a 500-Hz tone stream
s2 = d.create_tone(500)

# mix_mat is a numpy array that allows you to route input (file or array channels) to output channels, and adjust channel volume
mm = s2.mix_mat
mm *= .1 # turn the volume down
s2.mix_mat = mm
s2.play()

# Pink noise in right channel
s3 = d.create_pink()
mm = s3.mix_mat
mm[0] = 0 # turn off left channel
mm[1] = .1 # turn on right channel
s3.mix_mat = mm
s3.is_playing = True # Same as s3.play()
s3.is_muted = True # mute; same as s3.mute(True)

# Stream a flac file, show off fading:
s4 = medussa.open_file('/path/to/file.flac')
s4.mix_mat_fade_duration = 1. # Set fade to 1 sec
s4.play()
mm = s4.mix_mat
mm = mm[::-1] # Swap left and right channels
s4.mix_mat = mm # Fade will start here, when mix_mat is assigned. Left channel will fade out, right will fade in

s1.stop()

```
