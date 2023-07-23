# 2121-PySpace
GLSL Fractal Ray Marcher in Python.  
Forked from https://github.com/HackerPoet/PySpace

## First Installation
- Get Python 3  
- [Download the Repository](https://github.com/Nighthater/2121-PySpace/archive/refs/heads/master.zip)
- Start a Terminal in the Folder you just unzipped
- Windows: Right click -> Open in Terminal
```bash
pip install -r requirements.txt
```
- Wait for the command to complete (~30s)

# Usage

There are two batchfiles to use. `Run Recorder.bat` and `Run Renderer.bat`. They are functionally the Same program, just running in different screen resolutions.

| Keys | Action|
|--|--|
| W & S | Dolly |
| A & D | Truck |
| R & F | Pedestal |
| Mouse | Pan & Tilt (& Rotation)|
| Space | Start/Stop Recording Camera Path |
| P | Playback and Render Recorded Camera Path |
| C | Screenshot |
| Esc | End Program |


Keyvars affect the Shape of the Fractal. Not every Keyvar affects the fractal, try to experiment a bit.  
The Fractals you can change with the keyvars are already animated.  

| Keys | Action|
|--|--|
|Insert & Del| Change Keyvar 0|
|Pos 1 & End| Change Keyvar 1|
|Pg Up & Pg Down| Change Keyvar 2|
|Numpad 7 & Numpad 4| Change Keyvar 3|
|Numpad 8 & Numpad 5| Change Keyvar 4|
|Numpad 9 & Numpad 6| Change Keyvar 5|
|Left Shift| Decrase Change by 10|
|Right Shift| Increase Change by 10

# Workflow 
## Recording

- Launch `Run Recorder.bat`

- Press a Number between 1 and 8 to select a Fractal to view.  
##### NOTE: You can only select the Fractal at startup.

- Position you camera however you want, press `SPACE` to record your camera movement and `SPACE` to stop recording
##### NOTE: Only one Recording at a time will currently be recorded! More to that below.

- Once you finished a Recording, close the program with `ESC`

- If you want to take further Recording you should rename the `rec_vars.npy` and `recording.npy` files in the Directory.

- Best practice would be to append the filename with: `_[#NO. OF FRACTAL]_[#NO. OF RECORDING].npy`.  
  **And to place them into a separate folder!**

- Once you Recorded every Camerapath, you can now Render the Scenes you have taken!

## Rendering

- Copy and rename two corresponding camera path files back to `rec_vars.npy` and `recording.npy` in the main directory.

- Launch `Run Renderer.bat`

- Select the Fractal you originally rendered (Press a Number between 1 and 8)

- Press P

- Wait for the Renderer to Finish. The Program closes automatically

- The rendered Images will be saved to `./playback`. Save these images somewhere safe! When Rendering a new set of images the previous ones get overwritten. The First two Images are currently not usable.

- Delete the Recording Files in the current directory, and repeat the process again until everything is rendered.
