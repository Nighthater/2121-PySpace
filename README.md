# 2121-PySpace
GLSL Fractal Ray Marcher in Python.  
Forked from https://github.com/HackerPoet/PySpace



## Installation
- Get Python 3  
- [Download the Repository](https://github.com/Nighthater/2121-PySpace/archive/refs/heads/master.zip)
- Start a Terminal in the Folder you just unzipped
- Windows: Right click -> Open in Terminal
```bash
pip install -r requirements.txt
```
- Wait for the command to complete (~30s)

## Usage

Launch Run.bat  

Press a Number between 1 and 8 to select a Fractal to view  

| Keys | Action|
|--|--|
| W & S | Dolly |
| A & D | Truck |
| R & F | Pedestal |
| Mouse | Pan & Tilt |
| Space | Start/Stop Recording Camera Path|
| P | Playback and Render Recorded Camera Path |
| C | Screenshot |
| Esc | End Program |

### Keyvars

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
|Right Shift| Increase Change by 10|

## Render

- After Recording a Camera Path and Rendering it, all Images are placed in the playback folder.  
- The First two Images are currently not usable.  
- When Rendering a new set of images the previous ones get overwritten.  

## Videos
Overview: https://youtu.be/svLzmFuSBhk

Examples: https://youtu.be/N8WWodGk9-g

ODS Demo: https://youtu.be/yJyp7zEGKaU
