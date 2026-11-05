#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: flower_trail_task_v0.1.py
Date: 2025-05-15

This script was developed at Hamburg University at the department of health science.
We graciously thank Prof. Dr. Mirko Wegner for his scientific support in this development.
Martin Krentz built the first prototype of the script. Max Montgomery finished the development.


Implements the Flower Trail Task as a virtual cursor navigation task between concentric flower-shaped
SVG paths. Features include:
- Real-time kinematic constraints enforcement
- 100Hz positional data logging
- Automated trial management
- SVG path processing and geometric transformations
- Dynamic boundary violation detection
"""

AUTHOR_NAME = "Martin Krentz"
'''
Edited and adapted by: Max Archibald Montgomery
'''

from svgpathtools import svg2paths
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import matplotlib
import xml.etree.ElementTree as ET
import re
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import time
from math import dist
import math
import csv
from trigger_module import send_trigger
import argparse
from psychopy import core
import os
import platform
import ctypes 
import subprocess 
from show_example import show_example_animation


matplotlib.use('Agg')
screen_res = (1920,1080)

# define SVG namespace
namespace = {'svg': 'http://www.w3.org/2000/svg'}
flower_inner_path_data = """M304.1,555.6c0-7.1,0-14.3,0-21.6c21.1-3.7,33.5-15.7,35.1-37.7c5.2-2.5,10.7-4.3,15.7-7.1
    c10.3-5.6,19.2-12.9,26.7-21.8c9.5-11.3,15.6-24.3,18.6-38.6c1.6-7.6,1.8-15.3,1.4-23.1c-0.5-11.3-3.7-21.8-8.3-32
    c-2.4-5.3-5.4-10.4-9.2-14.9c-2.4-2.8-3-5.7-2.7-9.2c1.2-12.9,3.1-39.6,3-40.8c-14.4,9-27.8,18.7-42.4,28c-4.8-15.7-8-31.4-13-47.5
    c-9.9,13.6-19.4,26.6-29.2,40.1c-9.6-13.4-18.9-26.1-28.5-39.5c-4.7,15.9-7.9,31.3-13,47c-14.2-9.2-27.2-19.3-41.1-28.5
    c0.1,2.3,3,37,3.4,42.2c0.2,2.1-0.4,3.8-1.8,5.4c-8.1,9.8-13.9,20.7-17,32.4c-1.9,7-2.9,14.4-2.9,22c0,10.2,1.4,19.8,4.4,28.9
    c3.9,11.7,10.2,22.4,19.2,32c8.3,9,18.1,16,29.1,21.2c3.5,1.6,7.1,2.8,10.9,4.2c2.2,21.7,14.6,33.8,36,37.2c0,7.1,0,14.2,0,21.2"""

instructions_mouse = {
"welcome_message": [
"In dieser Aufgabe zeichnen Sie die Blume nach.",
"Die Blume besteht aus zwei Linien, die einen Weg bilden.",
"Ihre Aufgabe ist es, eine Linie ZWISCHEN diesen beiden Linien zu zeichnen, von Anfang bis Ende.",
"Geben Sie Ihr Bestes, die Linien der Blume nicht zu berühren oder zu überkreuzen.",
"Bewegen Sie sich so schnell und genau wie möglich.",
""
],
}


instructions_tablet = {
"welcome_message":[
"In dieser Aufgabe zeichnen Sie die Blume mit dem Stift auf dem Tablet vor Ihnen nach.",
"Die Blume besteht aus zwei Linien, die einen Weg bilden.",
"Ihre Aufgabe ist es, eine Linie ZWISCHEN diesen beiden Linien zu zeichnen, von Anfang bis Ende.",
"Geben Sie Ihr Bestes, die Linien der Blume nicht zu berühren oder zu überkreuzen.",
"Bewegen Sie sich so schnell und genau wie möglich.",
""
],
}


def force_pygame_to_front():
    """
    Uses OS-specific commands to force the Pygame window to the foreground.

    This should be called immediately after pygame.display.set_mode().
    It handles Windows, macOS, and Linux (requires 'wmctrl').
    """
    print("Attempting to force Pygame window to the front...")
    os_name = platform.system()
    try:
        wm_info = pygame.display.get_wm_info()
        
        if os_name == "Windows":
            user32 = ctypes.windll.user32
            hwnd = wm_info['window']
            # Force window to show and set as foreground
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            print("Windows: SetForegroundWindow called.")

        elif os_name == "Darwin":  # macOS
            # Uses AppleScript to bring the application to the front
            script = f'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true'
            subprocess.run(['/usr/bin/osascript', '-e', script], check=True)
            print("macOS: AppleScript for frontmost process executed.")
        
        elif os_name == "Linux":
            # Uses wmctrl to activate the window by its ID
            window_id = wm_info['window']
            subprocess.run(['wmctrl', '-i', '-a', str(window_id)], check=True)
            print(f"Linux: 'wmctrl -i -a {window_id}' executed.")
            
    except (KeyError, subprocess.CalledProcessError, FileNotFoundError, Exception) as e:
        print(f"Warning: Could not force window to front on {os_name}: {e}")
        print("The window may have launched in the background.")

def display_welcome_screen(screen, clock, condition):
    """
    Displays a condition-specific welcome message and waits for user input.
    """
    selected_message_lines = []
    if condition == 1:
        # Use .get() for safer dictionary access, providing a default empty list
        selected_message_lines = instructions_mouse.get("welcome_message", [])
        task_modality = "Mouse"
    elif condition == 2:
        selected_message_lines = instructions_tablet.get("welcome_message", [])
        task_modality = "Tablet"
    else:
        print(f"Unknown condition {condition} for welcome message. Skipping welcome screen.")
        return

    if not selected_message_lines:
        print(f"No welcome message content found for condition {condition} ({task_modality}). Skipping welcome screen.")
        return

    screen_width = screen.get_width()
    screen_height = screen.get_height()

    # Define fonts
    try:
        font_title = pygame.font.Font(None, 52) # For a title like "Welcome"
        font_message = pygame.font.Font(None, 36) # For the message lines
        font_prompt = pygame.font.Font(None, 30)  # For "Press any key"
    except pygame.error as e:
        print(f"Font loading error: {e}. Using default system font.")
        font_title = pygame.font.SysFont(None, 52)
        font_message = pygame.font.SysFont(None, 36)
        font_prompt = pygame.font.SysFont(None, 30)


    # Colors
    white = (255, 255, 255)
    black = (0, 0, 0)
    gray = (200, 200, 200)

    # Render text surfaces
    title_surface = font_title.render(f"Flower Task ({task_modality})", True, black)
    
    rendered_lines = []
    for line_text in selected_message_lines:
        text_surface = font_message.render(line_text, True, black)
        rendered_lines.append(text_surface)

    prompt_surface = font_prompt.render("Drücken Sie die Leertaste zu anfangen...", True, gray)

    # Calculate layout
    line_spacing = 10
    total_text_height = title_surface.get_height() + 20 # Space after title
    for rs in rendered_lines:
        total_text_height += rs.get_height() + line_spacing
    total_text_height += prompt_surface.get_height() + 30 # Space before prompt

    start_y = (screen_height - total_text_height) // 2
    
    current_y = start_y
    title_rect = title_surface.get_rect(center=(screen_width // 2, current_y + title_surface.get_height() // 2))
    current_y += title_surface.get_height() + 20 + line_spacing

    message_rects = []
    for text_surface in rendered_lines:
        rect = text_surface.get_rect(center=(screen_width // 2, current_y + text_surface.get_height() // 2))
        message_rects.append((text_surface, rect))
        current_y += text_surface.get_height() + line_spacing
    
    current_y += 30 # Extra space before prompt
    prompt_rect = prompt_surface.get_rect(center=(screen_width // 2, current_y + prompt_surface.get_height() // 2))

    # Display loop
    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                waiting_for_input = False

        screen.fill(white)
        screen.blit(title_surface, title_rect)
        for surface, rect in message_rects:
            screen.blit(surface, rect)
        screen.blit(prompt_surface, prompt_rect)

        pygame.display.flip()
        clock.tick(30) # Keep CPU usage reasonable during wait

def tablet_stand_setup_screen(screen, clock):
    """
    Tablet-Vorbereitung: Teilnehmer/in richtet den Ständer auf den Klebeband-Markierungen aus
    und bestätigt mit Doppeldruck der Leertaste.

    Blockiert, bis zwei SPACE-Tastendrücke detektiert wurden (innerhalb von 3s Abstand; ansonsten
    wird einfach weitergezählt – d.h. es muss insgesamt zweimal SPACE gedrückt werden).

    Returns
    -------
    bool
        True, wenn Doppeldruck registriert; False nur, wenn Fenster geschlossen wurde.
    """
    # --- Texte (Deutsch) ---
    lines = [
        "Tablet-Bedingung: Vorbereitung",
        "",
        "1. Platzieren Sie den Ständer auf dem Tisch so,",
        "   dass seine Füße GENAU über den Klebeband-Markierungen stehen.",
        "2. Achten Sie darauf, dass der Ständer stabil ist und sich nicht mehr bewegt.",
        "3. Legen Sie das Tablet in den Ständer, falls noch nicht geschehen.",
        "",
        "Wenn alles korrekt positioniert ist:",
        ">> Drücken Sie ZWEIMAL die LEERTASTE, um fortzufahren. <<",
        "",
        "Bei Problemen: Rufen Sie die Versuchsleitung."
    ]

    # --- Farben & Schriftarten ---
    white = (255, 255, 255)
    black = (0, 0, 0)
    gray  = (128, 128, 128)
    green = (0, 180, 0)

    try:
        font_title   = pygame.font.Font(None, 60)
        font_message = pygame.font.Font(None, 36)
        font_prompt  = pygame.font.Font(None, 32)
    except pygame.error:
        font_title   = pygame.font.SysFont(None, 60)
        font_message = pygame.font.SysFont(None, 36)
        font_prompt  = pygame.font.SysFont(None, 32)

    screen_w = screen.get_width()
    screen_h = screen.get_height()

    # Pre-render
    rendered = []
    for i, txt in enumerate(lines):
        if i == 0:
            srf = font_title.render(txt, True, black)
        elif txt.startswith(">>"):
            srf = font_prompt.render(txt, True, green)
        else:
            srf = font_message.render(txt, True, black if txt else gray)
        rendered.append(srf)

    total_h = sum(s.get_height() for s in rendered) + (len(rendered) - 1) * 8
    start_y = (screen_h - total_h) // 2

    space_count = 0
    # track timing between presses (optional; user can still press slowly)
    last_space_time = None
    max_gap = 3.0  # seconds between presses to count as a "double" (soft logic)

    running = True
    while running:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if last_space_time is None or (now - last_space_time) <= max_gap:
                        space_count += 1
                    else:
                        # gap too large -> restart counting
                        space_count = 1
                    last_space_time = now

                    if space_count >= 2:
                        return True

        # draw
        screen.fill(white)
        y = start_y
        for s in rendered:
            r = s.get_rect(center=(screen_w // 2, y + s.get_height() // 2))
            screen.blit(s, r)
            y += s.get_height() + 8

        # progress indicator
        prog_txt = font_prompt.render(f"SPACE: {space_count}/2", True, gray)
        prog_rect = prog_txt.get_rect(center=(screen_w // 2, screen_h - 60))
        screen.blit(prog_txt, prog_rect)

        pygame.display.flip()
        clock.tick(30)

def tablet_stand_teardown_screen(screen, clock):
    """
    Tablet-Abschluss: Teilnehmer/in stellt den Ständer zurück in die Ausgangsposition
    (weg vom Tablet) und bestätigt mit Doppeldruck der Leertaste.

    Returns
    -------
    bool
        True, wenn Doppeldruck registriert; False nur, wenn Fenster geschlossen wurde.
    """
    lines = [
        "Tablet-Bedingung: Abschluss",
        "",
        "Bitte stellen Sie den Ständer jetzt zurück an seine Ausgangsposition",
        "   (weg vom Tablet, auf die markierte Ablagefläche).",
        "Entfernen Sie das Tablet vorsichtig aus dem Ständer, falls nötig.",
        "Vergewissern Sie sich, dass der Ständer NICHT mehr auf den Klebeband-Markierungen steht.",
        "",
        "Wenn der Ständer korrekt zurückgestellt ist:",
        ">> Drücken Sie ZWEIMAL die LEERTASTE, um fortzufahren. <<",
        "",
        "Bei Unsicherheit: Bitte melden Sie sich bei der Versuchsleitung."
    ]

    white = (255, 255, 255)
    black = (0, 0, 0)
    gray  = (128, 128, 128)
    blue  = (0, 112, 192)

    try:
        font_title   = pygame.font.Font(None, 60)
        font_message = pygame.font.Font(None, 36)
        font_prompt  = pygame.font.Font(None, 32)
    except pygame.error:
        font_title   = pygame.font.SysFont(None, 60)
        font_message = pygame.font.SysFont(None, 36)
        font_prompt  = pygame.font.SysFont(None, 32)

    screen_w = screen.get_width()
    screen_h = screen.get_height()

    rendered = []
    for i, txt in enumerate(lines):
        if i == 0:
            srf = font_title.render(txt, True, black)
        elif txt.startswith(">>"):
            srf = font_prompt.render(txt, True, blue)
        else:
            srf = font_message.render(txt, True, black if txt else gray)
        rendered.append(srf)

    total_h = sum(s.get_height() for s in rendered) + (len(rendered) - 1) * 8
    start_y = (screen_h - total_h) // 2

    space_count = 0
    last_space_time = None
    max_gap = 3.0

    while True:
        now = time.perf_counter()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if last_space_time is None or (now - last_space_time) <= max_gap:
                    space_count += 1
                else:
                    space_count = 1
                last_space_time = now
                if space_count >= 2:
                    return True

        screen.fill(white)
        y = start_y
        for s in rendered:
            r = s.get_rect(center=(screen_w // 2, y + s.get_height() // 2))
            screen.blit(s, r)
            y += s.get_height() + 8

        prog_txt = font_prompt.render(f"SPACE: {space_count}/2", True, gray)
        prog_rect = prog_txt.get_rect(center=(screen_w // 2, screen_h - 60))
        screen.blit(prog_txt, prog_rect)

        pygame.display.flip()
        clock.tick(30)




def extract_path_data(path_data):
    commands = re.findall(r'([MmZzLlHhVvCcSsQqTtAa])|(-?[0-9]*\.?[0-9]+)', path_data)
    coordinates = []
    current_pos = (0, 0)

    i = 0
    command_length = len(commands)

    while i < command_length:
        cmd = commands[i][0]
        if cmd in 'Zz':
            i += 1
            continue
        elif cmd in 'Mm':
            current_pos = (float(commands[i + 1][1]), float(commands[i + 2][1]))
            coordinates.append(current_pos)
            i += 3
        elif cmd in 'Ll':
            current_pos = (current_pos[0] + float(commands[i + 1][1]),
                           current_pos[1] + float(commands[i + 2][1]))
            coordinates.append(current_pos)
            i += 3
        elif cmd in 'Cc':
            control1 = (current_pos[0] + float(commands[i + 1][1]), current_pos[1] + float(commands[i + 2][1]))
            control2 = (current_pos[0] + float(commands[i + 3][1]), current_pos[1] + float(commands[i + 4][1]))
            end_point = (current_pos[0] + float(commands[i + 5][1]), current_pos[1] + float(commands[i + 6][1]))

            # sample every points on cubic bezier curve
            steps = 20
            for t in np.linspace(0, 1, steps):
                x = (1 - t)**3 * current_pos[0] + \
                    3 * (1 - t)**2 * t * control1[0] + \
                    3 * (1 - t) * t**2 * control2[0] + \
                    t**3 * end_point[0]
                y = (1 - t)**3 * current_pos[1] + \
                    3 * (1 - t)**2 * t * control1[1] + \
                    3 * (1 - t) * t**2 * control2[1] + \
                    t**3 * end_point[1]
                coordinates.append((x, y))

            current_pos = end_point
            i += 7

    return coordinates

vertices_list = extract_path_data(flower_inner_path_data)
vertices_list.append((304.1, 555.0999999999999)) #manual addition to close this current shape

def reduce_redundant_points(vertices, threshold=0.1):
    reduced = []
    for vertex in vertices:
        if not reduced or (abs(vertex[0] - reduced[-1][0]) > threshold) or (abs(vertex[1] - reduced[-1][1]) > threshold):
            reduced.append(vertex)
    return reduced

vertices_list = reduce_redundant_points(vertices_list)

def scale_vertices(vertices, scale_factor, centroid):
    return [
        ((vertex[0] - centroid[0]) * scale_factor + (screen_res[0] // 2),
         (vertex[1] - centroid[1]) * scale_factor + (screen_res[1]  // 2))
        for vertex in vertices
    ]
scale_factor = 2.5

if vertices_list:
    centroid_x = sum(vertex[0] for vertex in vertices_list) / len(vertices_list)
    centroid_y = sum(vertex[1] for vertex in vertices_list) / len(vertices_list)
    centroid = (centroid_x, centroid_y)
    expansion_distance = 30
    centered_scaled_vertices = scale_vertices(vertices_list, scale_factor, centroid)

def generate_outer_path(inner_vertices, offset_distance=15):
    if not inner_vertices:
        return []
    poly = Polygon(inner_vertices).buffer(offset_distance, join_style=2)
    return list(poly.exterior.coords)

outer_vertices = generate_outer_path(vertices_list, offset_distance=10)
inner_scaled = scale_vertices(vertices_list, scale_factor, centroid)
outer_scaled = scale_vertices(outer_vertices, scale_factor, centroid)

if inner_scaled and outer_scaled:
    stalk_y = max(v[1] for v in inner_scaled)
    outer_scaled = [
        (x, min(y, stalk_y))
        for x, y in outer_scaled
    ]

def point_in_polygon(point, polygon):
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def hover_on_dot(screen, clock, dot_pos, tolerance=12, hover_time_required=2.0, instructions=None, condition = 1):
    pygame.mouse.set_visible(True)
    pygame.event.set_grab(False)
    font = pygame.font.Font(None, 36)
    local_instructions = None # Initialize a variable to hold the chosen instructions
    
    if condition == 1:
        local_instructions = instructions_mouse 
        # Maybe do something specific to A
        print(local_instructions["welcome_message"]) 
    elif condition == 2:
        local_instructions = instructions_tablet
        # Maybe do something specific to B
        print(local_instructions["welcome_message"])
    else:
        print("Unknown condition!")
        return # Or handle error appropriately
            
    hover_start = None
    total_text_height = len(instructions) * 40
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        screen.fill((255, 255, 255))
        pygame.draw.circle(screen, (0, 200, 0), (int(dot_pos[0]), int(dot_pos[1])), 10)
        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.circle(screen, (255, 0, 0), (int(mouse_pos[0]), int(mouse_pos[1])), 5)
        # Center text
        start_y = (screen.get_height() - total_text_height) // 2
        for i, line in enumerate(instructions):
            text = font.render(line, True, (0, 0, 0))
            text_rect = text.get_rect(center=(screen.get_width() // 2, start_y + 20 + 40 * i))
            screen.blit(text, text_rect)
        if dist(mouse_pos, dot_pos) <= tolerance:
            if hover_start is None:
                hover_start = time.perf_counter()
            hover_duration = time.perf_counter() - hover_start
            pygame.draw.arc(
                screen,
                (0, 200, 0),
                (dot_pos[0] - 16, dot_pos[1] - 16, 32, 32),
                -math.pi / 2,
                -math.pi / 2 + 2 * math.pi * min(hover_duration / hover_time_required, 1),
                4
            )
            if hover_duration >= hover_time_required:
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)
                return mouse_pos
        else:
            hover_start = None
        pygame.display.flip()
        clock.tick(60)

def run_trial(screen, clock, trial_num, inner_scaled, outer_scaled, start_pos, condition):
    end_trigger = 2
    
    stalk_base_y = max(p[1] for p in inner_scaled)
    outer_y_values = [p[1] for p in outer_scaled]
    barrier_top_y = (max(outer_y_values) + min(outer_y_values)) / 2
    tolerance = 5
    stalk_points = [p for p in outer_scaled if abs(p[1] - stalk_base_y) <= tolerance]
    left_x = min(p[0] for p in stalk_points)
    right_x = max(p[0] for p in stalk_points)
    mid_x = (left_x + right_x) / 2

    virtual_cursor_pos = list(start_pos)
    path_points = [tuple(virtual_cursor_pos)]

    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    start_time = time.perf_counter()
    next_sample = start_time + 0.01
    log_data = []
    finished = False
    last_pos = virtual_cursor_pos.copy()

    # Send a single, unambiguous start-of-trial trigger
    try:
        start_trigger = 1
        send_trigger(start_trigger)
        print(f"Start of trial trigger {start_trigger} sent (Condition: {condition}).")
    except Exception as e:
        print(f"Error sending start trigger: {e}")
        
    running = True
    while running:
        current_time = time.perf_counter()
        while current_time >= next_sample:
            sample_time = next_sample - start_time
            x, y = virtual_cursor_pos
            point = (x, y)
            dist_to_inner = min(dist(point, p) for p in inner_scaled)
            dist_to_outer = min(dist(point, p) for p in outer_scaled)
            in_outer = point_in_polygon(point, outer_scaled)
            in_inner = point_in_polygon(point, inner_scaled)
            valid_position = in_outer and not in_inner

            if valid_position:
                deviation = 0.0
                violation_type = "none"
            elif in_inner:
                deviation = dist_to_inner
                violation_type = "inner"
            else:
                deviation = dist_to_outer
                violation_type = "outer"

            log_data.append({
                'timestamp': round(sample_time, 4),
                'x': round(x, 2),
                'y': round(y, 2),
                'in_bounds': valid_position,
                'dist_to_inner': round(dist_to_inner, 3),
                'dist_to_outer': round(dist_to_outer, 3),
                'deviation': round(deviation, 3),
                'violation_type': violation_type
            })

            last_pos = [x, y]
            next_sample += 0.01

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEMOTION:
                # Use a ternary operator to set sensitivity: 0.6 for mouse, 1.0 (no change) for others.
                sensitivity = 0.6 if condition == 1 else 1.0
                new_x = virtual_cursor_pos[0] + (event.rel[0] * sensitivity)
                new_y = virtual_cursor_pos[1] + (event.rel[1] * sensitivity)
                new_y = min(new_y, stalk_base_y)
                if new_y > barrier_top_y:
                    if virtual_cursor_pos[0] < mid_x and new_x >= mid_x:
                        new_x = mid_x - 1
                    elif virtual_cursor_pos[0] > mid_x and new_x <= mid_x:
                        new_x = mid_x + 1
                new_x = max(0, min(screen_res[0], new_x))
                new_y = max(0, min(screen_res[0], new_y))
                virtual_cursor_pos = [new_x, new_y]

        elapsed = time.perf_counter() - start_time
        finish_triggered = (
                mid_x <= virtual_cursor_pos[0] <= right_x and
                stalk_base_y - 1 <= virtual_cursor_pos[1] <= stalk_base_y + 5
        )

        if finish_triggered and not finished:
            finish_time = elapsed
            finished = True

        if len(path_points) == 0 or dist(path_points[-1], virtual_cursor_pos) > 1:
            path_points.append(tuple(virtual_cursor_pos))

        screen.fill((255, 255, 255))
        pygame.draw.aalines(screen, (0, 0, 0), True, inner_scaled)
        pygame.draw.aalines(screen, (0, 0, 0), True, outer_scaled)
        pygame.draw.line(screen, (0, 255, 0), (left_x, stalk_base_y), (mid_x, stalk_base_y), 5)
        pygame.draw.line(screen, (255, 0, 0), (mid_x, stalk_base_y), (right_x, stalk_base_y), 5)
        if len(path_points) > 1:
            pygame.draw.lines(screen, (0, 0, 0), False, path_points, 1)
        pygame.draw.circle(screen, (255, 165, 0), (int(virtual_cursor_pos[0]), int(virtual_cursor_pos[1])), 5)
        font = pygame.font.Font(None, 36)
        text = font.render(f"Time: {min(elapsed, 30):.1f}s", True, (0, 0, 0))
        screen.blit(text, (20, 20))

        if elapsed > 30 or finished:
            running = False
        pygame.display.flip()
        clock.tick(200)
    # send end_trigger for trial
    if end_trigger is not None:
        try:
            send_trigger(end_trigger) # Corrected function name
            print(f"End of trial trigger {end_trigger} sent.")
        except Exception as e:
            print(f"Error sending end trigger {end_trigger}: {e}")
    pygame.event.set_grab(False)
    pygame.mouse.set_visible(True)

    return {
        "trial": trial_num,
        "time": finish_time if finished else 60,
        "success": finished,
        "path": path_points,
        "log": log_data
    }




def run_block(condition, participant_id):
    # Initialize results list once at the beginning
    results = []
    
    # Initialize Pygame
    pygame.init()
    try:
        screen = pygame.display.set_mode(screen_res)
        
        #force pygame to front
        force_pygame_to_front()
        # ---------------------

        if condition == 1:
            pygame.display.set_caption("Flower Tracing Task - Mouse Condition")
            pygame.display.set_caption("Flower Tracing Task - Mouse Condition")
        elif condition == 2:
            pygame.display.set_caption("Flower Tracing Task - Tablet Condition")
        else:
            pygame.display.set_caption("Flower Tracing Task")
        clock = pygame.time.Clock()
        
        # --- Display the welcome screen ---
        display_welcome_screen(screen, clock, condition)

        # --- Show the example animation ---
        show_example_animation(screen, clock)
        
        TRIAL_DURATION = 60
        NUM_TRIALS = 10

        # --- Calculate dot position ---
        stalk_base_y = max(p[1] for p in inner_scaled)
        tolerance = 5
        stalk_points = [p for p in outer_scaled if abs(p[1] - stalk_base_y) <= tolerance]
        left_x = min(p[0] for p in stalk_points)
        right_x = max(p[0] for p in stalk_points)
        mid_x = (left_x + right_x) / 2
        start_x = (left_x + mid_x) / 2
        start_y = stalk_base_y
        dot_pos = (int(start_x), int(start_y))

        # --- Initial hover-to-start screen ---
        if condition == 1: # Mouse condition
            start_instructions = [
                "Bewegen Sie den Mauszeiger auf den grünen Punkt.",
                "",
                "Halten Sie den Mauszeiger 2 Sekunden lang ruhig auf dem Punkt, um zu beginnen."
            ]
        else: # Tablet condition
            start_instructions = [
                "Setzen Sie die Stiftspitze auf den grünen Punkt und bewegen Sie sie nicht.",
                "Heben Sie den Stift nicht an, bis der Durchgang beendet ist.",
                "Halten Sie den Stift 2 Sekunden lang ruhig auf dem Punkt, um fortzufahren."
            ]
            
        # Call hover_on_dot only ONCE to start the first trial
        start_pos = hover_on_dot(screen, clock, dot_pos, tolerance=12, hover_time_required=2.0, instructions=start_instructions, condition=condition)

        # --- Main experiment loop ---
        for trial in range(NUM_TRIALS):
            result = run_trial(screen, clock, trial + 1, inner_scaled, outer_scaled, start_pos, condition)
            results.append(result)

            # --- Between-trials feedback screen ---
            # This block is now CORRECTLY INDENTED inside the for loop
            if trial < NUM_TRIALS - 1:
                if condition == 1: # Mouse condition
                    feedback_lines = [
                        f"Durchgang {result['trial']} beendet!",
                        f"Zeit: {result['time']:.2f}s",
                        "Erfolgreich!" if result['success'] else "Zeit abgelaufen!",
                        "",
                        "Für den nächsten Durchgang:",
                        "Halten Sie den Mauszeiger 2 Sekunden lang ruhig auf dem grünen Punkt."
                    ]
                else: # Tablet condition
                    feedback_lines = [
                        f"Durchgang {result['trial']} beendet!",
                        f"Zeit: {result['time']:.2f}s",
                        "Erfolgreich!" if result['success'] else "Zeit abgelaufen!",
                        "",
                        "Zum Fortfahren: Stift 2 Sekunden lang auf den grünen Punkt setzen.",
                        "Heben Sie den Stift nicht an, bis der nächste Durchgang beginnt."
                    ]
                # This call gets the starting position for the NEXT trial
                start_pos = hover_on_dot(screen, clock, dot_pos, tolerance=12, hover_time_required=2.0, instructions=feedback_lines, condition=condition)
        
        # --- Save data after all trials are complete ---
        # This is now correctly placed AFTER the for loop has finished
        save_results(results, condition, participant_id)
        save_block_summary(results, condition, participant_id, save_dir='data/flower_trail')

    finally:
        # This is guaranteed to run, ensuring Pygame doesn't linger
        print("Quitting Pygame.")
        pygame.quit()
    
    return results


def save_results(results, condition, participant_id, save_dir='data/flower_trail'):
    """
    Save all flower-task trial samples from a block into a single CSV.

    Parameters
    ----------
    results : list of dict
        Output from run_block(); each element:
            {
                'trial': int,
                'time': float,
                'success': bool,
                'path': [(x,y), ...],
                'log': [ {timestamp, x, y, in_bounds, dist_to_inner, dist_to_outer,
                          deviation, violation_type}, ... ]
            }
    condition : int
        1 = Mouse, 2 = Tablet (or whatever mapping you use).
    participant_id : str
        Participant identifier string.
    save_dir : str
        Directory where CSV will be written. Default 'data/flower_trail'.

    Returns
    -------
    str
        Full path to the written CSV file (or None if nothing saved).
    """
    import csv  # local import OK; also safe if global
    import os

    if not results:
        print("save_results(): No results to save; skipping file write.")
        return None

    # Ensure output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Target filename pattern: participant_condition_flower.csv
    filename = f"{participant_id}_{condition}_flower.csv"
    filepath = os.path.join(save_dir, filename)

    # Determine header. We add participant_id, condition, trial in front of the per-sample fields.
    # Assume at least one trial with at least one log row exists (we checked results non-empty,
    # but we double-guard below).
    first_log_row = None
    for tr in results:
        if tr.get('log'):
            first_log_row = tr['log'][0]
            break
    if first_log_row is None:
        print("save_results(): Results contained trials but no sample logs; skipping file write.")
        return None

    sample_fieldnames = list(first_log_row.keys())  # e.g., timestamp, x, y, ...
    fieldnames = ['participant_id', 'condition', 'trial'] + sample_fieldnames

    n_rows_written = 0
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial_dict in results:
            trial_num = trial_dict.get('trial', -1)
            logs = trial_dict.get('log', [])
            for row in logs:
                out_row = {
                    'participant_id': participant_id,
                    'condition': condition,
                    'trial': trial_num,
                }
                out_row.update(row)
                writer.writerow(out_row)
                n_rows_written += 1

    print(f"Saved {n_rows_written} samples across {len(results)} trial(s) to {filepath}")
    return filepath


def run_flower_drawing(win, io, participant_id, version, **kwargs):
    """
    Entry point for calling the Pygame-based flower task from a PsychoPy script.

    Tablet-Bedingung (version == 2):
        - Vor dem eigentlichen Block: tablet_stand_setup_screen()
        - Nach dem Block: tablet_stand_teardown_screen()

    Mouse-Bedingung (version == 1):
        - Direkt zum Block (unverändert).

    Parameters
    ----------
    win : psychopy.visual.Window or None
        Main PsychoPy window to minimize/restore.
    io : unused / placeholder for integration consistency.
    participant_id : str
        Participant identifier (e.g., "P001").
    version : int
        1 = Mouse, 2 = Tablet (passed to run_block as 'condition').
    **kwargs :
        Ignored; kept for API compatibility.

    Returns
    -------
    None
    """
    print(f"--- Launching Pygame Flower Task for sub-{participant_id}, version {version} ---")

    # -------------------------------
    # Minimize PsychoPy main window
    # -------------------------------
    if win:
        print("Minimizing PsychoPy window...")
        win.fullscr = False
        try:
            win.winHandle.minimize()
        except Exception as e:
            print(f"Warning: could not minimize PsychoPy window cleanly: {e}")
        win.flip()

    # -------------------------------------------
    # Tablet pre-task stand setup (separate loop)
    # -------------------------------------------
    if version == 2:
        print("[Tablet] Showing stand setup screen...")
        pygame.init()
        try:
            screen = pygame.display.set_mode(screen_res)
            pygame.display.set_caption("Tablet: Ständer ausrichten")
            clock = pygame.time.Clock()
            confirmed = tablet_stand_setup_screen(screen, clock)
            if not confirmed:
                print("Tablet stand setup aborted (window closed or not confirmed). Continuing anyway.")
        finally:
            pygame.quit()
            print("[Tablet] Setup screen closed.")

    # -------------------------
    # Run main Flower block
    # -------------------------
    error = None
    try:
        run_block(condition=version, participant_id=participant_id)
    except Exception as e:
        print(f"CRITICAL ERROR inside Pygame flower task: {e}")
        error = e  # stash; decide below whether to re-raise

    # ------------------------------------------------
    # Tablet post-task stand teardown (separate loop)
    # ------------------------------------------------
    if version == 2:
        print("[Tablet] Showing stand teardown screen...")
        pygame.init()
        try:
            screen = pygame.display.set_mode(screen_res)
            pygame.display.set_caption("Tablet: Ständer zurückstellen")
            clock = pygame.time.Clock()
            confirmed = tablet_stand_teardown_screen(screen, clock)
            if not confirmed:
                print("Tablet stand teardown aborted (window closed or not confirmed).")
        finally:
            pygame.quit()
            print("[Tablet] Teardown screen closed.")

    # -----------------------------------
    # Restore PsychoPy main window
    # -----------------------------------
    if win:
        print("Restoring PsychoPy window...")
        win.fullscr = True
        try:
            win.winHandle.maximize()
            win.winHandle.activate()
        except Exception as e:
            print(f"Warning: could not fully restore PsychoPy window: {e}")
        win.flip()
        core.wait(0.5)  # brief pause to ensure visible

    print("--- Pygame Flower Task complete. Returning control to main script. ---")




def save_block_summary(results, condition, participant_id, save_dir='data/flower_trail'):
    import csv, os
    if not results:
        return None
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{participant_id}_{condition}_flower_summary.csv"
    filepath = os.path.join(save_dir, filename)
    fieldnames = ['participant_id','condition','trial','time','success']
    with open(filepath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for tr in results:
            w.writerow({
                'participant_id': participant_id,
                'condition': condition,
                'trial': tr.get('trial'),
                'time': tr.get('time'),
                'success': tr.get('success')
            })
    print(f"Saved block summary to {filepath}")
    return filepath


# --- Standalone Execution Logic ---
def standalone_main():
    """
    Main function to run the experiment from the command line for testing.
    """
    parser = argparse.ArgumentParser(description="Run the Flower Tracing Task.")
    parser.add_argument("participant_id", type=str, help="Participant ID (e.g., 'P001').")
    parser.add_argument("condition", type=int, choices=[1, 2], help="Condition: 1 for Mouse, 2 for Tablet.")
    
    args = parser.parse_args()
    
    print(f"Starting standalone block for Participant: {args.participant_id}, Condition: {args.condition}")
    run_block(condition=args.condition, participant_id=args.participant_id)
    print("Experiment block finished.")

if __name__ == "__main__":
    standalone_main()

