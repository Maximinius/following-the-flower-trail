#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: show_example.py
Date: 2025-11-24

This script displays an animated example of the flower tracing task.
It shows the correct path being drawn in red between the two flower outlines.
This is intended to be shown to participants before they start the task
to give them a clear visual instruction of what is expected.
"""

import pygame
import sys
import numpy as np
import re
from shapely.geometry import Polygon
import math

# --- Copied from flower_module.py for consistency ---

screen_res = (1920, 1080)

flower_inner_path_data = """M304.1,555.6c0-7.1,0-14.3,0-21.6c21.1-3.7,33.5-15.7,35.1-37.7c5.2-2.5,10.7-4.3,15.7-7.1
    c10.3-5.6,19.2-12.9,26.7-21.8c9.5-11.3,15.6-24.3,18.6-38.6c1.6-7.6,1.8-15.3,1.4-23.1c-0.5-11.3-3.7-21.8-8.3-32
    c-2.4-5.3-5.4-10.4-9.2-14.9c-2.4-2.8-3-5.7-2.7-9.2c1.2-12.9,3.1-39.6,3-40.8c-14.4,9-27.8,18.7-42.4,28c-4.8-15.7-8-31.4-13-47.5
    c-9.9,13.6-19.4,26.6-29.2,40.1c-9.6-13.4-18.9-26.1-28.5-39.5c-4.7,15.9-7.9,31.3-13,47c-14.2-9.2-27.2-19.3-41.1-28.5
    c0.1,2.3,3,37,3.4,42.2c0.2,2.1-0.4,3.8-1.8,5.4c-8.1,9.8-13.9,20.7-17,32.4c-1.9,7-2.9,14.4-2.9,22c0,10.2,1.4,19.8,4.4,28.9
    c3.9,11.7,10.2,22.4,19.2,32c8.3,9,18.1,16,29.1,21.2c3.5,1.6,7.1,2.8,10.9,4.2c2.2,21.7,14.6,33.8,36,37.2c0,7.1,0,14.2,0,21.2"""

def extract_path_data(path_data):
    # This function is complex and seems to work, so I'll copy it as is.
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
            steps = 20
            for t in np.linspace(0, 1, steps):
                x = (1 - t)**3 * current_pos[0] + 3 * (1 - t)**2 * t * control1[0] + 3 * (1 - t) * t**2 * control2[0] + t**3 * end_point[0]
                y = (1 - t)**3 * current_pos[1] + 3 * (1 - t)**2 * t * control1[1] + 3 * (1 - t) * t**2 * control2[1] + t**3 * end_point[1]
                coordinates.append((x, y))
            current_pos = end_point
            i += 7
    return coordinates

vertices_list = extract_path_data(flower_inner_path_data)
vertices_list.append((304.1, 555.0999999999999))

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

def generate_outer_path(inner_vertices, offset_distance=15):
    if not inner_vertices:
        return []
    poly = Polygon(inner_vertices).buffer(offset_distance, join_style=2)
    return list(poly.exterior.coords)

# Generate inner, outer, and middle paths
inner_scaled = scale_vertices(vertices_list, scale_factor, centroid)

outer_vertices = generate_outer_path(vertices_list, offset_distance=10)
outer_scaled = scale_vertices(outer_vertices, scale_factor, centroid)

# Generate the middle path for the example animation
middle_vertices = generate_outer_path(vertices_list, offset_distance=5)
middle_scaled = scale_vertices(middle_vertices, scale_factor, centroid)

# Adjust stalk as in flower_module
if inner_scaled and outer_scaled:
    stalk_y = max(v[1] for v in inner_scaled)
    outer_scaled = [(x, min(y, stalk_y)) for x, y in outer_scaled]
    # Also adjust middle path
    middle_scaled = [(x, min(y, stalk_y)) for x, y in middle_scaled]

# --- Animation Function ---

def show_example_animation(screen, clock):
    """
    Displays an animation of the flower being drawn correctly.
    """
    # --- SETUP PATHS (Same as before) ---
    start_y = max(p[1] for p in middle_scaled)
    candidate_indices = [i for i, p in enumerate(middle_scaled) if p[1] > start_y - 10]
    if not candidate_indices:
        start_index_ccw = 0
    else:
        start_index_ccw = min(candidate_indices, key=lambda i: middle_scaled[i][0])

    end_index_ccw = (start_index_ccw - 1 + len(middle_scaled)) % len(middle_scaled)

    if start_index_ccw > end_index_ccw:
        path_ccw = middle_scaled[start_index_ccw:] + middle_scaled[:end_index_ccw + 1]
    else:
        path_ccw = middle_scaled[start_index_ccw:end_index_ccw + 1]
    
    path_cw = list(reversed(path_ccw))
    path_to_draw = [path_cw[-1]] + path_cw[:-1]
    path_to_draw.pop()

    # --- FONTS & SURFACES ---
    try:
        font_title = pygame.font.SysFont('freesansbold', 52)
        font_prompt = pygame.font.SysFont('freesansbold', 36)
    except pygame.error:
        font_title = pygame.font.Font(None, 52)
        font_prompt = pygame.font.Font(None, 36)
        
    white = (255, 255, 255)
    black = (0, 0, 0)
    red = (255, 0, 0)

    title_surface = font_title.render("Beispiel: So zeichnen Sie die Blume", True, black)
    title_rect = title_surface.get_rect(center=(screen_res[0] // 2, 50))
    
    # FIX 1: Move text UP. 
    # (1080 - 150) = 930. This ensures it is visible even if the window 
    # title bar pushes the bottom of the window off-screen.
    prompt_surface = font_prompt.render("Drücken Sie die Leertaste, um fortzufahren...", True, black)
    prompt_rect = prompt_surface.get_rect(center=(screen_res[0] // 2, screen_res[1] - 150))

    # --- ANIMATION LOOP ---
    drawn_path = [path_to_draw[0]]
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Allow skipping animation
            if event.type == pygame.KEYDOWN or event.type == pygame.MOUSEBUTTONDOWN:
                running = False
        
        # Draw Logic
        if len(drawn_path) < len(path_to_draw):
            drawn_path.append(path_to_draw[len(drawn_path)])
        else:
            running = False 

        screen.fill(white)
        screen.blit(title_surface, title_rect)
        pygame.draw.aalines(screen, black, True, inner_scaled)
        pygame.draw.aalines(screen, black, True, outer_scaled)
        if len(drawn_path) > 1:
            pygame.draw.lines(screen, red, False, drawn_path, 3)
        
        pygame.display.flip()
        clock.tick(120)

    # --- FINAL STATIC DISPLAY (WAIT LOOP) ---
    # We assume the path is fully drawn now
    full_path = path_to_draw

    # Clear events to prevent accidental double-skips
    pygame.event.clear()

    print("Animation finished. Waiting for input...")

    waiting_for_input = True
    while waiting_for_input:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    waiting_for_input = False
        
        # FIX 2: CONTINUOUS DRAWING
        # We must redraw the screen inside this loop to ensure 
        # the prompt stays visible and the OS doesn't blank the window.
        screen.fill(white)
        screen.blit(title_surface, title_rect)
        pygame.draw.aalines(screen, black, True, inner_scaled)
        pygame.draw.aalines(screen, black, True, outer_scaled)
        
        # Draw the full red path
        pygame.draw.lines(screen, red, False, full_path, 3)
        
        # Draw the prompt
        screen.blit(prompt_surface, prompt_rect)
        
        pygame.display.flip()
        clock.tick(30)

    # Optional: Save the example image after they are done viewing it, or before.
    # Putting it here ensures it doesn't lag the transition to the prompt.
    pygame.image.save(screen, "flower_example.png")

# --- Standalone Execution Logic ---
def main():
    """
    Main function to run the example animation standalone for testing.
    """
    pygame.init()
    try:
        screen = pygame.display.set_mode(screen_res)
        pygame.display.set_caption("Flower Tracing Example")
        clock = pygame.time.Clock()
        show_example_animation(screen, clock)
    finally:
        pygame.quit()
        print("Exited successfully.")

if __name__ == "__main__":
    main()
