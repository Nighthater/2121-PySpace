#!/usr/bin/env python

import pygame, sys, math, random, os
import numpy as np
import pyspace

from pyspace.coloring import *
from pyspace.fold import *
from pyspace.geo import *
from pyspace.object import *
from pyspace.shader import Shader
from pyspace.camera import Camera

from ctypes import *
from OpenGL.GL import *
from pygame.locals import *

from PIL import Image
from PIL import ImageOps

import time

import os
os.environ['SDL_VIDEO_CENTERED'] = '1'

#Size of the window and rendering
win_size = (1280, 720)
#win_size = (2560, 1440)
#win_size = (3840, 2160)

#Maximum frames per second
max_fps = 30

#Forces an 'up' orientation when True, free-camera when False
gimbal_lock = False

#Mouse look speed
look_speed = 0.003

#Use this avoids collisions with the fractal
auto_velocity = True
auto_multiplier = 2.0

#Maximum velocity of the camera
max_velocity = 2.0

#Amount of acceleration when moving
speed_accel = 2.0

#Velocity decay factor when keys are released
speed_decel = 0.6

clicking = False
mouse_pos = None
screen_center = (win_size[0]/2, win_size[1]/2)
start_pos = [0, 0, 12.0]
vel = np.zeros((3,), dtype=np.float32)
look_x = 0.0
look_y = 0.0

#----------------------------------------------
#    When building your own fractals, you can
# substitute numbers with string placeholders
# which can be tuned real-time with key bindings.
#
# In this example program:
#    '0'   +Insert  -Delete
#    '1'   +Home    -End
#    '2'   +PageUp  -PageDown
#    '3'   +NumPad7 -NumPad4
#    '4'   +NumPad8 -NumPad5
#    '5'   +NumPad9 -NumPad6
#
# Hold down left-shift to decrease rate 10x
# Hold down right-shift to increase rate 10x
#
# Set initial values of '0' through '6' below
#----------------------------------------------
keyvars = [1.0, 0.5, 5.0, 2.0, 1.57, 0.5]

#----------------------------------------------
#            Fractal Examples Below
#----------------------------------------------
def infinite_spheres():
    obj = Object()
    obj.add(FoldRepeatX('0'))
    obj.add(FoldRepeatY('1'))
    obj.add(FoldRepeatZ('2'))
    obj.add(Sphere(0.5, (1.0, 1.0, 1.0), color=(0.5,0.1,0.75)))
    return obj
    
def mandelbox():
    obj = Object()
    obj.add(OrbitInitInf())
    for _ in range(32):
        obj.add(FoldBox(1.0))
        obj.add(FoldSphere('1', '2'))
        obj.add(FoldScaleOrigin('3'))
        obj.add(OrbitMinAbs(1.0))
    obj.add(Box(6.0, color=(0.50,0.0,1.0)))
    return obj


def mausoleum():
    obj = Object()
    obj.add(OrbitInitZero())
    for _ in range(16):
        obj.add(FoldBox(0.5))
        obj.add(FoldMenger())
        obj.add(FoldScaleTranslate('2', (-5.27,-0.34,'1')))
        obj.add(FoldRotateX( math.pi / 2 ))
        obj.add(OrbitMax((0.42,0.38,0.19)))
    obj.add(Box(2.0, color='orbit'))
    return obj

def menger():
    obj = Object()
    for _ in range(8):
        obj.add(FoldAbs())
        obj.add(FoldMenger())
        obj.add(FoldScaleTranslate('3', (-2,-2,'1')))
        obj.add(FoldPlane((0,0,-1), -1))
    obj.add(Box(2.0, color=(.2,.5,1)))
    return obj

def tree_planet():
    obj = Object()
    obj.add(OrbitInitInf())
    for _ in range(30):
        obj.add(FoldRotateY(0.44))
        obj.add(FoldAbs())
        obj.add(FoldMenger())
        obj.add(OrbitMinAbs((0.24,2.28,7.6)))
        obj.add(FoldScaleTranslate(1.3, (-2,-4.8,0)))
        obj.add(FoldPlane((0,0,-1), 0))
    obj.add(Box(4.8, color='orbit'))
    return obj

def sierpinski_tetrahedron():
    obj = Object()
    obj.add(OrbitInitZero())
    for _ in range(50):
        obj.add(FoldSierpinski())
        obj.add(FoldScaleTranslate(2, -1))
    obj.add(Tetrahedron(color=(0.5,0.2,0.7)))
    return obj

def snow_stadium():
    obj = Object()
    obj.add(OrbitInitInf())
    for _ in range(30):
        obj.add(FoldRotateY(3.33))
        obj.add(FoldSierpinski())
        obj.add(FoldRotateX(0.15))
        obj.add(FoldMenger())
        obj.add(FoldScaleTranslate('4', (-6.61, -4.0, -2.42)))
        obj.add(OrbitMinAbs(1.0))
    obj.add(Box(4.8, color='orbit'))
    return obj

def test_fractal():
    obj = Object()
    obj.add(OrbitInitInf())
    for _ in range(20):
        obj.add(FoldSierpinski())
        obj.add(FoldMenger())
        obj.add(FoldRotateY(math.pi/2))
        obj.add(FoldAbs())
        obj.add(FoldRotateZ('0'))
        obj.add(FoldScaleTranslate(1.89, (-7.10, 0.396, -6.29)))
        obj.add(OrbitMinAbs((1,1,1)))
    obj.add(Box(6.0, color='orbit'))
    return obj

#----------------------------------------------
#             Helper Utilities
#----------------------------------------------
def interp_data(x, f=2.0):
    new_dim = int(x.shape[0]*f)
    output = np.empty((new_dim,) + x.shape[1:], dtype=np.float32)
    for i in range(new_dim):
        a, b1 = math.modf(float(i) / f)
        b2 = min(b1 + 1, x.shape[0] - 1)
        output[i] = x[int(b1)]*(1-a) + x[int(b2)]*a
    return output

def make_rot(angle, axis_ix):
    s = math.sin(angle)
    c = math.cos(angle)
    if axis_ix == 0:
        return np.array([[ 1,  0,  0],
                         [ 0,  c, -s],
                         [ 0,  s,  c]], dtype=np.float32)
    elif axis_ix == 1:
        return np.array([[ c,  0,  s],
                         [ 0,  1,  0],
                         [-s,  0,  c]], dtype=np.float32)
    elif axis_ix == 2:
        return np.array([[ c, -s,  0],
                         [ s,  c,  0],
                         [ 0,  0,  1]], dtype=np.float32)

def reorthogonalize(mat):
    u, s, v = np.linalg.svd(mat)
    return np.dot(u, v)

# move the cursor back , only if the window is focused
def center_mouse():
    if pygame.key.get_focused():
        pygame.mouse.set_pos(screen_center)

#--------------------------------------------------
#                  Video Recording
#
#    When you're ready to record a video, press 'r'
# to start recording, and then move around.  The
# camera's path and live '0' through '5' parameters
# are recorded to a file.  Press 'r' when finished.
#
#    Now you can exit the program and turn up the
# camera parameters for better rendering.  For
# example; window size, anti-aliasing, motion blur,
# and depth of field are great options.
#
#    When you're ready to playback and render, press
# 'p' and the recorded movements are played back.
# Images are saved to a './playback' folder.  You
# can import the image sequence to editing software
# to convert it to a video.
#
#    You can press 'c' anytime for a screenshot.
#---------------------------------------------------

if __name__ == '__main__':
    pygame.init()
    window = pygame.display.set_mode(win_size, OPENGL | DOUBLEBUF)
    pygame.mouse.set_visible(False)
    center_mouse()

    #======================================================
    #               Menu Screen
    #======================================================
    obj_render = None
    
    menu = {
        '1': ('Mandelbox', mandelbox),
        '2': ('Mausoleum', mausoleum),
        '3': ('Infinite Spheres', infinite_spheres),
        '4': ('Menger Sponge', menger),
        '5': ('Tree Planet', tree_planet),
        '6': ('Sierpinski Tetrahedron', sierpinski_tetrahedron),
        '7': ('Snow Stadium', snow_stadium),
        '8': ('Example Fractal', test_fractal)
    }
    
    font_size = 60
    my_font = pygame.font.Font(None, font_size) # None gets default font

    widest_title = max([my_font.size(title[0]) for title in menu.values()])[0]
    menu_left_margin = (win_size[0] - widest_title) / 2

    i = 2
    for k, v in menu.items():
        text = k + " : " + v[0]
        text_surface = my_font.render(text, True, ( 255, 255, 255 ))
        window.blit(text_surface, ( menu_left_margin, i * font_size ))
        i += 1
    text_surface = my_font.render('Fractal Menu', True, ( 255, 255, 255 ))
    window.blit(text_surface, ( (win_size[0] - text_surface.get_width()) / 2, 60 ))
    pygame.display.flip()

    while True:
        # sleep until a key is pressed
        event = pygame.event.wait()
        if event.type == pygame.QUIT: sys.exit(0)
        if event.type == pygame.KEYDOWN and event.unicode in '123456789':
            obj_render = menu[event.unicode][1]()
            break
    window = pygame.display.set_mode(win_size, OPENGL | DOUBLEBUF)
    #======================================================

    #======================================================
    #             Change camera settings here
    # See pyspace/camera.py for all camera options
    #======================================================
    camera = Camera()
    camera['ANTIALIASING_SAMPLES'] = 1
    camera['AMBIENT_OCCLUSION_STRENGTH'] = 0.00
    #======================================================

    shader = Shader(obj_render)
    program = shader.compile(camera)
    print("Compiled!")

    matID = glGetUniformLocation(program, "iMat")
    prevMatID = glGetUniformLocation(program, "iPrevMat")
    resID = glGetUniformLocation(program, "iResolution")
    ipdID = glGetUniformLocation(program, "iIPD")

    glUseProgram(program)
    glUniform2fv(resID, 1, win_size)
    glUniform1f(ipdID, 0.04)

    fullscreen_quad = np.array([-1.0, -1.0, 0.0, 1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, fullscreen_quad)
    glEnableVertexAttribArray(0)

    mat = np.identity(4, np.float32)
    mat[3,:3] = np.array(start_pos)
    prevMat = np.copy(mat)
    for i in range(len(keyvars)):
        shader.set(str(i), keyvars[i])

    recording = None
    rec_vars = None
    playback = None
    playback_vars = None
    playback_ix = -1
    frame_num = 0

    def start_playback():
        global playback
        global playback_vars
        global playback_ix
        global prevMat
        if not os.path.exists('playback'):
            os.makedirs('playback')
        playback = np.load('recording.npy')
        playback_vars = np.load('rec_vars.npy')
        playback = interp_data(playback, 2)
        playback_vars = interp_data(playback_vars, 2)
        playback_ix = 0
        prevMat = playback[0]

    clock = pygame.time.Clock()
    
    starttime = time.time()
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if recording is None:
                        print("Recording...")
                        recording = []
                        rec_vars = []
                    else:
                        np.save('recording.npy', np.array(recording, dtype=np.float32))
                        np.save('rec_vars.npy', np.array(rec_vars, dtype=np.float32))
                        recording = None
                        rec_vars = None
                        print("Finished Recording.")
                elif event.key == pygame.K_p:
                    start_playback()
                elif event.key == pygame.K_c:
                    glPixelStorei(GL_PACK_ALIGNMENT, 1)
                    data = glReadPixels(0, 0, window.get_width(), window.get_height(), GL_RGB, GL_UNSIGNED_BYTE)
                    image = Image.frombytes("RGB", (window.get_width(), window.get_height()), data)
                    image = ImageOps.flip(image)
                    image.save('screenshot.png', 'PNG')
                elif event.key == pygame.K_ESCAPE:
                    sys.exit(0)

        mat[3,:3] += vel * (clock.get_time() / 1000)

        if auto_velocity:
            de = obj_render.DE(mat[3]) * auto_multiplier
            if not np.isfinite(de):
                de = 0.0
        else:
            de = 1e20

        all_keys = pygame.key.get_pressed()

        rate = 0.01
        if all_keys[pygame.K_LSHIFT]:   rate *= 0.1
        elif all_keys[pygame.K_RSHIFT]: rate *= 10.0

        # Automated FunkyStuff happens here
        localtime = time.time() - starttime
        keyvars[0] += math.sin(localtime / 7) * 0.0025
        keyvars[1] += math.sin(localtime / 5) * 0.01
        keyvars[2] += math.sin(localtime / 2) * 0.01
        keyvars[3] += math.sin(localtime / 8) * 0.001
        keyvars[4] += math.sin(localtime / 8) * 0.01
        keyvars[5] += math.sin(localtime / 11) * 0.01
        print(keyvars)
        
        if all_keys[pygame.K_INSERT]:   keyvars[0] += rate; print(keyvars)
        if all_keys[pygame.K_DELETE]:   keyvars[0] -= rate; print(keyvars)
        if all_keys[pygame.K_HOME]:     keyvars[1] += rate; print(keyvars)
        if all_keys[pygame.K_END]:      keyvars[1] -= rate; print(keyvars)
        if all_keys[pygame.K_PAGEUP]:   keyvars[2] += rate; print(keyvars)
        if all_keys[pygame.K_PAGEDOWN]: keyvars[2] -= rate; print(keyvars)
        if all_keys[pygame.K_KP7]:      keyvars[3] += rate; print(keyvars)
        if all_keys[pygame.K_KP4]:      keyvars[3] -= rate; print(keyvars)
        if all_keys[pygame.K_KP8]:      keyvars[4] += rate; print(keyvars)
        if all_keys[pygame.K_KP5]:      keyvars[4] -= rate; print(keyvars)
        if all_keys[pygame.K_KP9]:      keyvars[5] += rate; print(keyvars)
        if all_keys[pygame.K_KP6]:      keyvars[5] -= rate; print(keyvars)

        if playback is None:
            prev_mouse_pos = mouse_pos
            mouse_pos = pygame.mouse.get_pos()
            dx,dy = 0,0
            if prev_mouse_pos is not None:
                center_mouse()
                time_rate = (clock.get_time() / 1000.0) / (1 / max_fps)
                dx = (mouse_pos[0] - screen_center[0]) * time_rate
                dy = (mouse_pos[1] - screen_center[1]) * time_rate

            if pygame.key.get_focused():
                if gimbal_lock:
                    look_x += dx * look_speed
                    look_y += dy * look_speed
                    look_y = min(max(look_y, -math.pi/2), math.pi/2)

                    rx = make_rot(look_x, 1)
                    ry = make_rot(look_y, 0)

                    mat[:3,:3] = np.dot(ry, rx)
                else:
                    rx = make_rot(dx * look_speed, 1)
                    ry = make_rot(dy * look_speed, 0)

                    mat[:3,:3] = np.dot(ry, np.dot(rx, mat[:3,:3]))
                    mat[:3,:3] = reorthogonalize(mat[:3,:3])

            acc = np.zeros((3,), dtype=np.float32)
            if all_keys[pygame.K_a]:
                acc[0] -= speed_accel / max_fps
            if all_keys[pygame.K_d]:
                acc[0] += speed_accel / max_fps
            if all_keys[pygame.K_r]:
                acc[1] += speed_accel / max_fps
            if all_keys[pygame.K_f]:
                acc[1] -= speed_accel / max_fps
            if all_keys[pygame.K_w]:
                acc[2] -= speed_accel / max_fps
            if all_keys[pygame.K_s]:
                acc[2] += speed_accel / max_fps
            
            if np.dot(acc, acc) == 0.0:
                vel *= speed_decel # TODO
            else:
                vel += np.dot(mat[:3,:3].T, acc)
                vel_ratio = min(max_velocity, de) / (np.linalg.norm(vel) + 1e-12)
                if vel_ratio < 1.0:
                    vel *= vel_ratio

            if recording is not None:
                recording.append(np.copy(mat))
                rec_vars.append(np.array(keyvars, dtype=np.float32))
        else:
            if playback_ix >= 0:
                ix_str = '%04d' % playback_ix
                glPixelStorei(GL_PACK_ALIGNMENT, 1)
                data = glReadPixels(0, 0, window.get_width(), window.get_height(), GL_RGB, GL_UNSIGNED_BYTE)
                image = Image.frombytes("RGB", (window.get_width(), window.get_height()), data)
                image = ImageOps.flip(image)
                image.save('playback/frame' + ix_str + '.png', 'PNG')
            if playback_ix >= playback.shape[0]:
                playback = None
                break
            else:
                mat = prevMat * 0.98 + playback[playback_ix] * 0.02
                mat[:3,:3] = reorthogonalize(mat[:3,:3])
                keyvars = playback_vars[playback_ix].tolist()
                playback_ix += 1

        for i in range(3):
            shader.set(str(i), keyvars[i])
        shader.set('v', np.array(keyvars[3:6]))
        shader.set('pos', mat[3,:3])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUniformMatrix4fv(matID, 1, False, mat)
        glUniformMatrix4fv(prevMatID, 1, False, prevMat)
        prevMat = np.copy(mat)

        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        pygame.display.flip()
        clock.tick(max_fps)
        frame_num += 1
        
