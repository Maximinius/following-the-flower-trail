
# maze_module.py
import pygame
import os
import logging # Keep logging for potential system messages if needed
import time # Keep for time.sleep
from pygame.math import Vector2
import math
import csv
import random
import threading # Import threading
from psychopy import core # Import PsychoPy core for Clock
from collections import deque # Import deque at the top level
import sys
from trigger_module import send_trigger
import platform
import ctypes # For Windows
import subprocess # For Linux

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
        

# --- Configuration ---
ZOOM_FACTOR = 0.8
POINTER_RADIUS_RATIO = 0.25 # Pointer radius relative to cell size
LOG_INTERVAL_MS = 100 # Target logging interval in milliseconds
DEFAULT_MAZE_PATH = "pre_generated_mazes.npy" # Default path, change if needed
DEFAULT_LOG_DIR = "data/maze_logs" # Directory to save log files

def get_participant_id():
    from psychopy import gui, core # gui needs to be imported here
    dlg = gui.Dlg(title="Participant ID")
    dlg.addField("Subject Number:")
    dlg.show()
    if dlg.OK:
        return dlg.data
        [0]
    else:
        core.quit()
        return None # Should return None or raise an error if not OK

def load_mazes(path: str):
    """
    Load mazes (and, if present, pre-computed solutions) from *path*.

    Returns
    -------
    list
        • Old format  –>  [np.ndarray, …]  
        • New format  –>  [(np.ndarray, [(x,y), …]), …]
          (internally the file stores dicts; we turn them into tuples
          for cleaner unpacking later).

    Notes
    -----
    ``allow_pickle=True`` is required because the *new* file stores
    Python objects (dicts).
    """
    import numpy as np

    raw = np.load(path, allow_pickle=True)
    if len(raw) == 0:
        raise ValueError(f"'{path}' is empty.")

    first = raw[0]

    # New format that has both solution and mazes in the same file
    if isinstance(first, dict) and {"maze", "solution"} <= first.keys():
        return [
            (item["maze"], item["solution"])
            for item in raw
            if isinstance(item, dict)
        ]


def get_maze_and_solution(maze_item):
    """
    Normalise a maze entry coming from ``load_mazes`` to the tuple

        (maze_grid : np.ndarray, solution_coords : list[(x,y)] | None)

    so the rest of the code doesn’t need to care about file format.
    """
    if isinstance(maze_item, tuple) and len(maze_item) == 2:
        return maze_item              # new format – unpacked already
    return maze_item, None            # old format – must solve on-the-fly



def get_random_maze(mazes):
    """
    Selects and removes a random maze from the mazes list.
    Raises an error if no mazes remain.
    """

    index = random.randrange(len(mazes))
    print(f"Selected maze index {index} (from {len(mazes)} remaining).")
    return mazes.pop(index) # pop removes the item


def calculate_layout(screen_width, screen_height, maze_shape):
    """Calculates cell size and offsets for centering the maze."""
    print(f"DEBUG: Calculating layout for maze shape: {maze_shape}") # DEBUG
    if not isinstance(maze_shape, (tuple, list)) or len(maze_shape) != 2:
         raise TypeError(f"Invalid maze_shape: {maze_shape}. Must be a tuple/list of length 2.")
    if maze_shape[0] <= 0 or maze_shape[1] <= 0:
        raise ValueError(f"Maze dimensions must be positive. Got: {maze_shape}")
    max_cell_w = screen_width / maze_shape[1]
    max_cell_h = screen_height / maze_shape[0]
    cell_size = int(min(max_cell_w, max_cell_h) * ZOOM_FACTOR)
    if cell_size <= 0:
        print(f"Warning: Calculated cell_size is non-positive ({cell_size}). Setting to 1.")
        cell_size = 1
    maze_width = cell_size * maze_shape[1]
    maze_height = cell_size * maze_shape[0]
    offset_x = (screen_width - maze_width) // 2
    offset_y = (screen_height - maze_height) // 2
    print(f"Layout calculated: cell_size={cell_size}, offset=({offset_x}, {offset_y})")
    return cell_size, offset_x, offset_y

def create_walls_and_paths(maze_grid, cell_size, offset):
    """Creates lists of pygame.Rect objects for walls and path cells."""
    print("DEBUG: Creating wall and path Rects...") # DEBUG
    walls = []
    path_cells = []
    offset_x, offset_y = offset
    rows, cols = maze_grid.shape
    for y in range(rows):
        for x in range(cols):
            rect = pygame.Rect(
                offset_x + x * cell_size,
                offset_y + y * cell_size,
                cell_size, cell_size
            )
            if rect.width <= 0 or rect.height <= 0:
                 print(f"Warning: Created invalid Rect at grid ({x},{y}) with size ({rect.width},{rect.height})")
                 continue
            if maze_grid[y, x] == 1:
                walls.append(rect)
            else:
                path_cells.append(rect)
    print(f"Created {len(walls)} wall rectangles and {len(path_cells)} path rectangles.")
    return walls, path_cells

def find_start_end_positions(maze_grid):
    """Finds the (x, y) grid coordinates for the start and end points."""
    print("DEBUG: Finding start/end grid positions...") # DEBUG
    start = None
    end = None
    rows, cols = maze_grid.shape
    for x in range(cols):
        if maze_grid[0, x] == 0:
            start = (x, 0)
            print(f"DEBUG: Found start at grid {start}") # DEBUG
            break
    for x in range(cols): # Use original logic: find first path cell from left in last row
        if maze_grid[rows - 1, x] == 0:
            end = (x, rows - 1)
            print(f"DEBUG: Found end at grid {end}") # DEBUG
            break
    if start is None: print("Warning: No start position found.")
    if end is None: print("Warning: No end position found.")
    return start, end

def solve_maze(maze_grid):
    """Solves the maze using Breadth-First Search (BFS).
    NOTE: Depreciated because we solve the mazes and load the solutions directly now. I keep this for legacy purposes
    """
    print("DEBUG: Attempting to solve maze using BFS...") # DEBUG
    start, end = find_start_end_positions(maze_grid)
    if start is None or end is None:
        print("Error: Cannot solve maze, start or end position not found or invalid.")
        return []
    rows, cols = maze_grid.shape
    queue = deque([(start, [start])])
    visited = {start}
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    step_count = 0
    while queue:
        step_count += 1
        if step_count > rows * cols * 2:
             print("Error: BFS exceeded maximum steps.")
             return []
        (current_pos, path) = queue.popleft()
        cx, cy = current_pos
        if current_pos == end:
            print(f"Solution path found with length {len(path)} in {step_count} BFS steps.")
            return path
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            next_pos = (nx, ny)
            if 0 <= nx < cols and 0 <= ny < rows:
                if maze_grid[ny, nx] == 0 and next_pos not in visited:
                    visited.add(next_pos)
                    new_path = list(path)
                    new_path.append(next_pos)
                    queue.append((next_pos, new_path))
    print(f"Warning: No solution path found after {step_count} BFS steps.")
    return []

def get_solution_path_rects(solution_coords, cell_size, offset):
    """Converts maze solver coordinates into pygame.Rects for 'on_path' checks."""
    print("DEBUG: Creating solution path Rects...") # DEBUG
    rects = []
    offset_x, offset_y = offset
    for x, y in solution_coords:
        rect = pygame.Rect(
            offset_x + x * cell_size,
            offset_y + y * cell_size,
            cell_size, cell_size
        )
        rects.append(rect)
    print(f"DEBUG: Created {len(rects)} solution path rectangles.")
    return rects


'''
Experiment tracker class is used for logging purposes. Change input settings at the beginning of the script to customise logging for your own needs. I personally like 100ms as it is granular
'''

class ExperimentTracker:
    def __init__(self, trial_number, subject_id, log_dir=DEFAULT_LOG_DIR):
        self.trial_number = trial_number
        self.subject_id = subject_id
        self.log_dir = log_dir
        print(f"DEBUG: Initializing ExperimentTracker for Trial {trial_number}, Subject {subject_id}") # DEBUG

        # --- PsychoPy Clock for high-resolution timing ---
        self.logging_clock = core.Clock()
        self.trial_start_time = None # Will be set when logging starts

        # --- State variables to be updated by the main thread ---
        self._latest_pos = None
        self._latest_moving = False
        self._latest_collision = False
        self._latest_on_path = False
        self._lock = threading.Lock() # Lock for thread-safe state updates

        # --- Logging thread control ---
        self._logging_active = False
        self._log_thread = None
        self._log_interval_sec = LOG_INTERVAL_MS / 1000.0

        # --- State for speed/acceleration calculation (internal to logger) ---
        self._last_log_time = None
        self._last_log_pos = None
        self._last_log_speed = 0.0

        # --- CSV File Setup ---
        base_filename = f"maze_{subject_id}_{trial_number}_{time.strftime('%Y-%m-%d')}"
        log_extension = ".csv"
        self.filepath = os.path.join(log_dir, base_filename + log_extension)
        
        version = 2
        while os.path.exists(self.filepath):
            self.filepath = os.path.join(log_dir, f"{base_filename}_v{version}{log_extension}")
            version += 1
        
        self.filename = os.path.basename(self.filepath)
        self.file = None
        self.csv_writer = None
        try:
            # Ensure directory exists
            print(f"DEBUG: Ensuring log directory exists: {log_dir}") # DEBUG
            os.makedirs(log_dir, exist_ok=True)
            print(f"DEBUG: Attempting to open log file: {self.filepath}") # DEBUG

            self.file = open(self.filepath, "w", newline="")
            self.csv_writer = csv.writer(self.file)
            self.csv_writer.writerow([
                "timestamp", "x", "y",
                "moving", "collision", "on_path",
                "speed", "acceleration"
            ])
            print(f"Opened log file: {self.filepath}")
        except IOError as e:
            print(f"CRITICAL ERROR: Could not open log file {self.filename} in {self.log_dir}: {e}")
            print("Logging will be disabled for this trial.")
            if self.file: # Close if partially opened
                 try: self.file.close()
                 except: pass
            self.file = None
            self.csv_writer = None
            self.filepath = None # Indicate failure
            # Consider raising an error here if logging is absolutely critical
            # raise IOError(f"Failed to open log file: {e}") from e


    def update_state(self, pos: Vector2, moving: bool, collision: bool, on_path: bool):
        """
        Called by the main game thread to update the latest known state.
        """
        # print(f"DEBUG: update_state called - Pos: {pos}, Moving: {moving}, Collision: {collision}, OnPath: {on_path}") # DEBUG - Can be VERY verbose
        with self._lock:
            # Make a copy of the position vector to avoid race conditions if
            # the main thread modifies its vector after this call
            self._latest_pos = pos.copy() if pos else None
            self._latest_moving = moving
            self._latest_collision = collision
            self._latest_on_path = on_path

    def start_logging(self):
        """
        Starts the background logging thread.
        """
        print("DEBUG: start_logging called.") # DEBUG
        if not self.csv_writer: # Don't start if file failed to open
            print("Logging disabled: CSV file writer not available.")
            return

        if self._log_thread is not None and self._log_thread.is_alive():
            print("Logging thread already running.")
            return

        # Ensure initial state is available BEFORE starting thread
        with self._lock:
            if self._latest_pos is None:
                print("Warning: Starting logger before initial position is set.")
                # Optionally, wait or provide a default starting state?
                # For now, the logger loop handles the None case.

        print(f"Starting logging for trial {self.trial_number}...")
        self._logging_active = True
        self.logging_clock.reset() # Reset clock at the start of logging
        self.trial_start_time = self.logging_clock.getTime() # Record zero point
        print(f"DEBUG: Logger trial_start_time set to: {self.trial_start_time}") # DEBUG
        self._last_log_time = None # Reset history for speed/accel
        self._last_log_pos = None
        self._last_log_speed = 0.0

        self._log_thread = threading.Thread(target=self._logging_loop, daemon=True)
        print("DEBUG: Starting logging thread...") # DEBUG
        self._log_thread.start()

    def stop_logging(self):
        """
        Signals the logging thread to stop and waits for it to finish.
        Also writes completion marker and closes the file.
        NOTE: A bunch of debug statements in this function makes it look scarier than it is. I just had a lot of trouble with it... Might remove these in the future.
        """
        print("DEBUG: stop_logging called.") # DEBUG
        if not self._logging_active and self._log_thread is None:
             # Logging might have been disabled or already stopped
             print("Stop logging called, but logging was not active or thread doesn't exist.")
             # Still try to close the file if it exists
             if self.file:
                 try:
                     self.file.close()
                     print(f"Log file closed (was inactive): {self.filepath}")
                 except Exception as e:
                     print(f"Error closing inactive log file: {e}")
                 finally:
                     self.file = None
             return

        print(f"Stopping logging for trial {self.trial_number}...")
        self._logging_active = False # Signal the loop to stop

        if self._log_thread and self._log_thread.is_alive():
            print("DEBUG: Joining logging thread...") # DEBUG
            self._log_thread.join(timeout=1.5) # Wait a bit longer for the thread
            if self._log_thread.is_alive():
                print("Warning: Logging thread did not terminate gracefully after 1.5s.")
        self._log_thread = None # Mark thread as stopped/joined
        print("Logging thread stopped.")

        # Log completion and close file
        if self.file and self.csv_writer:
            print("DEBUG: Writing completion marker and closing file...") # DEBUG
            try:
                # Use the logger's clock for final duration relative to its start
                duration = self.logging_clock.getTime() # Already relative to reset
                # Write a marker row indicating completion
                self.csv_writer.writerow([f"{duration:.4f}", "COMPLETED", "", "", "", "", "", ""])
                print(f"Completion marker written to {self.filepath}")
            except Exception as e:
                print(f"Error writing completion row: {e}")
            finally:
                # Ensure file is closed even if writing the marker fails
                try:
                    self.file.close()
                    print(f"Log file saved and closed: {self.filepath}")
                except Exception as e:
                    print(f"Error closing log file: {e}")
                finally:
                     self.file = None # Mark as closed
        elif self.file: # Ensure file is closed even if writer failed initially
            print("DEBUG: Closing file (writer was invalid)...") # DEBUG
            try:
                self.file.close()
                print(f"Log file closed (writer might have been invalid): {self.filepath}")
            except Exception as e:
                print(f"Error closing file (no writer): {e}")
            finally:
                 self.file = None


    def _logging_loop(self):
        """
        The main loop for the logging thread. Runs until _logging_active is False.
        """
        # Initialize next_log_time based on the clock's state *after* reset in start_logging
        # Add the first interval duration to the start time
        if self.trial_start_time is None: # Should not happen if start_logging was called
             print("Error: Logging loop started but trial_start_time is None.")
             self.trial_start_time = self.logging_clock.getTime() # Fallback

        # Ensure trial_start_time is a float before adding
        if not isinstance(self.trial_start_time, float):
             print(f"Error: trial_start_time is not a float ({type(self.trial_start_time)}). Resetting clock again.")
             self.logging_clock.reset()
             self.trial_start_time = self.logging_clock.getTime()

        next_log_time = self.trial_start_time + self._log_interval_sec
        missed_deadlines = 0

        print(f"Logging loop started. Interval: {self._log_interval_sec:.4f}s. First log target: {next_log_time:.4f}") # DEBUG

        while self._logging_active:
            current_time = self.logging_clock.getTime()

            # --- Wait until the next scheduled log time ---
            # Calculate how long to sleep
            sleep_duration = next_log_time - current_time
            if sleep_duration > 0.0001: # Only sleep if duration is meaningful
                # Use a precise sleep if possible (core.wait is PsychoPy's high-res wait)
                try:
                    # print(f"DEBUG: Logger sleeping for {sleep_duration:.5f}s") # DEBUG - Very verbose
                    # core.wait(sleep_duration, hogCPUperiod=0) # hogCPUperiod=0 tries not to hog # << OLD
                    time.sleep(sleep_duration) # << NEW: Simpler sleep for background thread
                # except AttributeError: # Fallback if core.wait not available or behaves differently # << NO LONGER NEEDED
                #     time.sleep(sleep_duration)
                except Exception as e: # Catch other potential core.wait errors
                    # print(f"Warning: core.wait failed: {e}. Falling back to time.sleep().") # << Message might change
                    print(f"Warning: time.sleep encountered an issue (unlikely): {e}.")
                    time.sleep(sleep_duration) # Fallback, though time.sleep itself rarely fails this way

            # Re-check current time after waiting/potential delay
            current_time = self.logging_clock.getTime()

            # Check if we missed the deadline significantly
            # Allow for small timing variations (e.g., half the interval)
            # Only check if we are actually supposed to log (i.e., past next_log_time)
            if current_time >= next_log_time:
                if current_time > next_log_time + (self._log_interval_sec / 2.0):
                    missed_deadlines += 1
                    if missed_deadlines % 100 == 0: # Log warning periodically
                        lag = current_time - next_log_time
                        print(f"Warning: Logger thread missed ~{missed_deadlines} deadlines (current lag: {lag*1000:.1f} ms). System load high or interval too short?")
                    # Reset next_log_time to catch up based on the current time plus one interval
                    next_log_time = current_time + self._log_interval_sec
                else:
                    # Schedule the next log time precisely based on the previous target
                    next_log_time += self._log_interval_sec

                # --- Read the latest state safely ---
                pos = None # Ensure pos is defined in this scope
                moving = False
                collision = False
                on_path = False
                with self._lock:
                    # Copy mutable objects like Vector2
                    if self._latest_pos:
                        pos = self._latest_pos.copy()
                    moving = self._latest_moving
                    collision = self._latest_collision
                    on_path = self._latest_on_path

                if pos is None: # Wait until the main thread provides an initial position
                    # print("Logger waiting for initial position...") # Can be noisy
                    continue # Skip logging this cycle and wait for the next one

                # --- Log the captured state ---
                # Use the actual time the sample was processed
                # Timestamp should be relative to the trial start (which getTime() provides after reset)
                timestamp = current_time
                x, y = pos.x, pos.y

                # --- Compute speed & acceleration based on logger's history ---
                speed = 0.0
                acceleration = 0.0
                if self._last_log_time is not None and self._last_log_pos is not None:
                    dt = timestamp - self._last_log_time
                    # Ensure dt is reasonably positive to avoid instability
                    if dt > 1e-6: # Check against a small positive number (0.001 ms)
                        dx = x - self._last_log_pos[0]
                        dy = y - self._last_log_pos[1]
                        dist = math.hypot(dx, dy)
                        speed = dist / dt
                        # Ensure speed is non-negative
                        speed = max(0.0, speed)
                        acceleration = (speed - self._last_log_speed) / dt
                    else: # If dt is too small or negative, reuse last speed, zero acceleration
                        # print(f"DEBUG: dt too small ({dt:.6f}), reusing speed.") # DEBUG
                        speed = self._last_log_speed
                        acceleration = 0.0
                # else: First sample after start or reset, speed/accel remain 0.0


                # --- Write row to CSV ---
                try:
                    if self.csv_writer:
                        # print(f"DEBUG: Logging row: T={timestamp:.4f}, Pos=({x:.1f},{y:.1f})") # DEBUG - Verbose
                        self.csv_writer.writerow([
                            f"{timestamp:.4f}",
                            f"{x:.2f}", f"{y:.2f}", # Position precision
                            int(moving),
                            int(collision),
                            int(on_path),
                            f"{speed:.3f}",      # Speed precision
                            f"{acceleration:.3f}" # Acceleration precision
                        ])
                    else: # Should not happen if loop is active, but safety check
                         self._logging_active = False # Stop loop if writer gone
                         print("Error: CSV writer became unavailable during logging.")
                         break
                except Exception as e:
                    print(f"Error writing log row: {e}")
                    # Option: Stop logging if file errors persist?
                    # self._logging_active = False
                    # break

                # --- Update logger's history for next calculation ---
                self._last_log_time = timestamp
                self._last_log_pos = (x, y)
                self._last_log_speed = speed

            # End of the "if current_time >= next_log_time" block
            # If we weren't scheduled to log yet, the loop repeats after a potential short sleep

        print("Logging loop finished.")

# --- Pointer Class (Handles position, radius, and continuous collision checking) ---
class Pointer:
    def __init__(self, start_pos, radius):
        print(f"DEBUG: Initializing Pointer.") # DEBUG
        # Ensure start_pos is a Vector2
        if isinstance(start_pos, (tuple, list)) and len(start_pos) == 2:
            self.pos = Vector2(start_pos)
        elif isinstance(start_pos, Vector2):
            self.pos = start_pos.copy() # Use copy to avoid external modification
        else:
            raise TypeError(f"start_pos must be a Vector2, tuple, or list of 2 numbers. Got: {start_pos}")

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError(f"Pointer radius must be a positive number. Got: {radius}")
        self.radius = float(radius)
        print(f"Pointer initialized at {self.pos} with radius {self.radius:.2f}")

    @staticmethod
    def _clamp(value, min_val, max_val):
        """Clamps a value within a specified range."""
        return max(min_val, min(value, max_val))

    # --- check_continuous_collision is NOT USED by the simplified update_cursor ---

def check_continuous_collision(self, start_pos: Vector2, movement: Vector2, walls: list[pygame.Rect]):
        """
        [THIS FUNCTION IS CURRENTLY BYPASSED BY THE SIMPLIFIED update_cursor]
        Checks for collision along a movement vector against walls using Minkowski difference
        and ray casting (simplified via clipline).
        """
        # print(f"DEBUG: check_continuous_collision - Start: {start_pos}, Move: {movement}") # DEBUG - Verbose
        if movement.length_squared() < 1e-12: # Ignore negligible movement
            # print("DEBUG: Negligible movement, skipping collision check.") # DEBUG
            return None, None, movement

        earliest_t = 1.0 # Time of collision (0=start, 1=end of movement)
        colliding_wall_rect = None
        collision_normal = None
        final_collision_point = None # Store the point of earliest collision

        end_pos = start_pos + movement
        movement_len_sq = movement.length_squared() # Cache for calculating 't'
        if movement_len_sq < 1e-12: # Avoid division by zero later
            movement_len_sq = 1e-12

        # Iterate through each wall to find the earliest collision time
        for wall_index, wall in enumerate(walls): # DEBUG: Added index
            # Minkowksi difference: Expand wall by pointer radius
            expanded_rect = pygame.Rect(
                wall.left - self.radius,
                wall.top - self.radius,
                wall.width + 2 * self.radius,
                wall.height + 2 * self.radius
            )

            # Ray-casting against the expanded rectangle using clipline
            try:
                # clipline expects tuples or lists, not Vector2 directly
                clipped_line = expanded_rect.clipline(
                    (start_pos.x, start_pos.y),
                    (end_pos.x, end_pos.y)
                )
            except TypeError as e:
                print(f"Warning: Error in clipline with start={start_pos}, end={end_pos}, rect={expanded_rect}: {e}")
                continue # Skip this wall if clipline fails

            if clipped_line:
                # print(f"DEBUG: Clipline intersect with wall {wall_index} ({wall}), result: {clipped_line}") # DEBUG - Verbose
                p1 = Vector2(clipped_line[0])
                diff_vec = p1 - start_pos
                dist_sq = diff_vec.length_squared()
                t = dist_sq / movement_len_sq # Use cached squared length
                t = self._clamp(t, 0.0, 1.0)

                if t < earliest_t - 1e-9:
                    # print(f"DEBUG: New earliest collision at t={t:.4f} with wall {wall_index}") # DEBUG - Verbose
                    earliest_t = t
                    colliding_wall_rect = wall # Store the original wall rect
                    intersect_center_pos = start_pos + movement * t
                    final_collision_point = intersect_center_pos # Store this point

                    # --- Determine Collision Normal ---
                    closest_x = self._clamp(intersect_center_pos.x, wall.left, wall.right)
                    closest_y = self._clamp(intersect_center_pos.y, wall.top, wall.bottom)
                    closest_point_on_wall = Vector2(closest_x, closest_y)
                    normal_vec = intersect_center_pos - closest_point_on_wall
                    if normal_vec.length_squared() > 1e-12:
                        collision_normal = normal_vec.normalize()
                    else:
                        # print(f"DEBUG: Collision exactly on edge/corner of wall {wall_index}. Using movement dir for normal.") # DEBUG
                        move_dir = movement.normalize() if movement.length_squared() > 1e-12 else Vector2(0,0)
                        wall_center = Vector2(wall.center)
                        diff_to_center = intersect_center_pos - wall_center
                        if abs(diff_to_center.x / (wall.width + 1e-6)) > abs(diff_to_center.y / (wall.height + 1e-6)):
                             collision_normal = Vector2(math.copysign(1, diff_to_center.x), 0)
                        else:
                             collision_normal = Vector2(0, math.copysign(1, diff_to_center.y))
                        if collision_normal.length_squared() > 1e-12:
                            collision_normal.normalize_ip()
                        else:
                             if move_dir.length_squared() > 0: collision_normal = -move_dir
                             else: collision_normal = Vector2(1,0)


        # --- Process Collision Result ---
        if earliest_t < 1.0 - 1e-10 and collision_normal is not None: # Use tolerance for comparison
            # print(f"DEBUG: Collision detected! Earliest t={earliest_t:.4f}, Normal={collision_normal}") # DEBUG
            epsilon = 1e-6 # Small distance to prevent sticking
            movement_len = movement.length()
            safe_t = max(0.0, earliest_t - epsilon / movement_len if movement_len > epsilon else 0.0)
            allowed_movement = movement * safe_t
            remaining_t = 1.0 - earliest_t
            remaining_movement = movement * remaining_t
            dot_product = remaining_movement.dot(collision_normal)
            if dot_product < 0:
                slide_movement = remaining_movement - collision_normal * dot_product
                # print(f"DEBUG: Sliding applied. Remaining: {remaining_movement}, Slide: {slide_movement}") # DEBUG
            else:
                slide_movement = remaining_movement
                # print(f"DEBUG: No slide needed (dot={dot_product:.3f}). Remaining: {remaining_movement}") # DEBUG
            resolved_movement = allowed_movement + slide_movement
            # print(f"DEBUG: Resolved movement: {resolved_movement}") # DEBUG
            return final_collision_point, collision_normal, resolved_movement
        else:
            # print("DEBUG: No collision detected.") # DEBUG
            return None, None, movement


# --- update_cursor Function. This works now
def update_cursor(pointer: 'Pointer', walls: list[pygame.Rect], tracker: 'ExperimentTracker', solution_rects: list[pygame.Rect]):
    dx, dy = pygame.mouse.get_rel()
    sensitivity = 0.2
    total_mv = Vector2(dx * sensitivity, dy * sensitivity)

    initial_pos = pointer.pos.copy()
    frame_collision_occurred = False

    if total_mv.length_squared() < 1e-9:
        moving_flag = False
        collision_flag = False
        for w in walls:
            expanded = pygame.Rect(w.left - pointer.radius, w.top - pointer.radius,
                                     w.width + 2 * pointer.radius, w.height + 2 * pointer.radius)
            if expanded.collidepoint(pointer.pos.x, pointer.pos.y):
                collision_flag = True
                break
        frame_collision_occurred = collision_flag
        on_path_flag = any(r.collidepoint(pointer.pos.x, pointer.pos.y) for r in solution_rects)
        tracker.update_state(pointer.pos, moving_flag, frame_collision_occurred, on_path_flag)
        return

    max_step_dist = pointer.radius / 2.0
    steps = 1
    if total_mv.length_squared() > 1e-9 and total_mv.length() > max_step_dist:
        steps = int(math.ceil(total_mv.length() / max_step_dist))
    
    step_mv = total_mv / steps if steps > 0 else Vector2(0,0)
    current_pos = pointer.pos.copy()

    for _ in range(steps):
        desired_pos = current_pos + step_mv
        
        for iter_count in range(5): # Iterative resolution
            best_wall_to_resolve = None
            max_numeric_overlap = -float('inf') # Can be negative if no overlap
            
            # For resolution data
            resolution_data = {
                "normal": Vector2(0,0), 
                "is_deep_penetration": False,
                "closest_point_on_wall_for_deep": Vector2(0,0) # Only for deep
            }

            # 1. Identify the best candidate wall for collision resolution
            for wall in walls:
                # Broad phase check (optional, but expanded_rect is already this)
                expanded_check_rect = pygame.Rect(
                    wall.left - pointer.radius, wall.top - pointer.radius,
                    wall.width + 2 * pointer.radius, wall.height + 2 * pointer.radius
                )
                if not expanded_check_rect.collidepoint(desired_pos.x, desired_pos.y):
                    continue # Pointer center not even in the broad collision zone for this wall

                # Precise check for this wall
                closest_x = max(wall.left, min(desired_pos.x, wall.right))
                closest_y = max(wall.top, min(desired_pos.y, wall.bottom))
                closest_point_on_wall_vec = Vector2(closest_x, closest_y)
                
                penetration_vector_to_wall = desired_pos - closest_point_on_wall_vec
                dist_sq_to_closest_point = penetration_vector_to_wall.length_squared()

                current_numeric_overlap = 0
                is_deep = False

                if dist_sq_to_closest_point < 1e-12: # Center is ON/INSIDE wall AABB
                    is_deep = True
                    # Estimate overlap as radius, or distance to push out to nearest edge.
                    # For simplicity in selection, prioritize deep penetrations.
                    # Actual push logic will handle it.
                    current_numeric_overlap = pointer.radius # Treat as significant overlap
                else:
                    dist_to_closest_point = math.sqrt(dist_sq_to_closest_point)
                    current_numeric_overlap = pointer.radius - dist_to_closest_point
                
                if current_numeric_overlap > max_numeric_overlap:
                    max_numeric_overlap = current_numeric_overlap
                    best_wall_to_resolve = wall
                    resolution_data["is_deep_penetration"] = is_deep
                    if is_deep:
                        resolution_data["closest_point_on_wall_for_deep"] = closest_point_on_wall_vec
                        # Normal for deep penetration will be determined during push
                    else:
                        resolution_data["normal"] = penetration_vector_to_wall.normalize()

            # 2. If a collision needs resolution, resolve it
            if best_wall_to_resolve and max_numeric_overlap > 1e-9: # Use small tolerance for "meaningful" overlap
                frame_collision_occurred = True
                wall_to_push_from = best_wall_to_resolve

                if resolution_data["is_deep_penetration"]:
                    # Deep penetration: desired_pos is ON/INSIDE wall_to_push_from AABB
                    wall_center = Vector2(wall_to_push_from.center)
                    norm_delta_x = abs(desired_pos.x - wall_center.x) / (wall_to_push_from.width / 2 + 1e-7)
                    norm_delta_y = abs(desired_pos.y - wall_center.y) / (wall_to_push_from.height / 2 + 1e-7)
                    push_correction = Vector2(0,0)
                    epsilon_clearance = 1e-7 # Minimal clearance

                    if norm_delta_x > norm_delta_y:
                        if desired_pos.x < wall_center.x:
                            target_x = wall_to_push_from.left - pointer.radius - epsilon_clearance
                            push_correction.x = target_x - desired_pos.x
                        else:
                            target_x = wall_to_push_from.right + pointer.radius + epsilon_clearance
                            push_correction.x = target_x - desired_pos.x
                    else:
                        if desired_pos.y < wall_center.y:
                            target_y = wall_to_push_from.top - pointer.radius - epsilon_clearance
                            push_correction.y = target_y - desired_pos.y
                        else:
                            target_y = wall_to_push_from.bottom + pointer.radius + epsilon_clearance
                            push_correction.y = target_y - desired_pos.y
                    
                    if push_correction.length_squared() < 1e-12: # Fallback if no correction calculated
                        diff_to_center = desired_pos - wall_center
                        if diff_to_center.length_squared() < 1e-12:
                            push_correction = Vector2(pointer.radius + epsilon_clearance, 0)
                        else:
                            push_correction = diff_to_center.normalize() * (pointer.radius + epsilon_clearance)
                    desired_pos += push_correction
                else:
                    # Shallow penetration
                    # max_numeric_overlap is the 'overlap' distance
                    # resolution_data["normal"] is the correct normal
                    # Use a very small epsilon to prevent sticking but minimize jitter
                    desired_pos += resolution_data["normal"] * (max_numeric_overlap + 1e-7) 
            else:
                # No meaningful overlap with any wall, or no wall found
                break # Exit iterative resolution for this sub-step
        
        current_pos = desired_pos

    pointer.pos = current_pos
    final_pos = pointer.pos # Use the final updated pointer.pos

    moved_flag = (final_pos - initial_pos).length_squared() > (0.01)**2
    on_path_flag = any(r.collidepoint(final_pos.x, final_pos.y) for r in solution_rects)
    tracker.update_state(final_pos, moved_flag, frame_collision_occurred, on_path_flag)


# --- UI Functions (run_countdown, instructions_screen, ending_screen) ---
def run_countdown(screen, countdown_time, start_pos_pixels):
    """
    Displays a countdown on the screen and moves the mouse to start_pos_pixels.
    """
    print("DEBUG: Starting countdown...") # DEBUG
    font_large = pygame.font.Font(None, 120) # Larger font for countdown
    font_small = pygame.font.Font(None, 80) # Smaller font for "Go!"
    screen_center_x = screen.get_width() // 2
    screen_center_y = screen.get_height() // 2
    blue_color = (0, 0, 200)
    white_color = (255, 255, 255) # Background color

    # Initial short pause
    screen.fill(white_color)
    pygame.display.flip() # Show blank screen first
    pygame.time.delay(500)

    # Countdown loop
    for count in range(countdown_time, 0, -1):
        print(f"DEBUG: Countdown: {count}") # DEBUG
        screen.fill(white_color)
        countdown_text = font_large.render(str(count), True, blue_color)
        text_rect = countdown_text.get_rect(center=(screen_center_x, screen_center_y))
        screen.blit(countdown_text, text_rect)
        pygame.display.flip()
        pygame.time.delay(1000) # Wait 1 second

    # Display "Go!"
    print("DEBUG: Countdown: Go!") # DEBUG
    screen.fill(white_color)
    go_text = font_small.render("Go!", True, blue_color)
    text_rect = go_text.get_rect(center=(screen_center_x, screen_center_y))
    screen.blit(go_text, text_rect)
    pygame.display.flip()
    pygame.time.delay(750) # Show "Go!" for a bit

    # Move mouse to start coordinates and reset relative movement buffer
    try:
        # Ensure start_pos_pixels is integer tuple/list for set_pos
        start_pos_int = (int(start_pos_pixels[0]), int(start_pos_pixels[1]))
        print(f"DEBUG: Setting mouse position to {start_pos_int}") # DEBUG
        pygame.mouse.set_pos(start_pos_int)
        pygame.mouse.get_rel() # IMPORTANT: Clear relative buffer AFTER setting position
        print(f"Mouse positioned at: {start_pos_int}")
        print("DEBUG: Grabbing mouse input and hiding cursor.") # DEBUG
        pygame.event.set_grab(True) # Confine mouse to window
        pygame.mouse.set_visible(False) # Hide system cursor
    except Exception as e:
        print(f"Error setting mouse position or grabbing input: {e}")
        # Decide how to handle this - maybe exit?
        pygame.quit()
        exit(f"Failed to initialize mouse state: {e}")
    print("DEBUG: Countdown finished.") # DEBUG


def instructions_screen(screen):
    """Displays instructions and waits for SPACE key press."""
    print("DEBUG: Displaying instructions screen.") # DEBUG
    screen.fill((255, 255, 255)) # White background
    font_title = pygame.font.Font(None, 60)
    font_text = pygame.font.Font(None, 36)
    blue_color = (0, 0, 180)
    black_color = (0, 0, 0)
    screen_width = screen.get_width()
    screen_height = screen.get_height()

    instructions = [("Willkommen zur Labyrinth-Aufgabe!", font_title, blue_color, -150), 
    ("", font_text, black_color, -100), 
    # Spacer,
    ("Führen Sie mit Ihrer Maus den roten Kreis", font_text, black_color, -50),
    ("von oben zum grünen Zielbereich am unteren Rand.", font_text, black_color, -20),
    ("", font_text, black_color, 80), # Spacer
    ("Drücken Sie die LEERTASTE, um den ersten Durchgang zu starten.", font_text, blue_color, 150)]

    # Render each line
    y_pos = screen_height // 2 - 150 # Start position for first line
    line_height = 30 # Adjust spacing as needed

    for line, font, color, y_offset in instructions: # Using original offsets for simplicity
         if line: # Skip empty lines for rendering
             text_surface = font.render(line, True, color)
             text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2 + y_offset))
             screen.blit(text_surface, text_rect)

    pygame.display.flip()
    print("DEBUG: Instructions displayed, waiting for SPACE.") # DEBUG

    # Wait for SPACE press or QUIT/ESCAPE
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                print("Exited during instructions.")
                exit() # Exit the script cleanly
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("DEBUG: SPACE pressed, starting experiment.") # DEBUG
                    waiting = False
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    print("Exited during instructions.")
                    exit() # Exit the script cleanly
        pygame.time.Clock().tick(30) # Prevent busy-waiting


def ending_screen(screen):
    """Displays the final 'thank you' screen and waits for key press/quit."""
    print("DEBUG: Displaying ending screen.") # DEBUG
    # Release mouse grab and make cursor visible
    try: # Add try-except in case pygame is already quit
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
    except pygame.error as e:
        print(f"DEBUG: Pygame error during ending screen cleanup (might be expected if quit early): {e}")


    screen.fill((255, 255, 255)) # White background
    font = pygame.font.Font(None, 48)
    blue_color = (0, 0, 180)
    screen_center_x = screen.get_width() // 2
    screen_center_y = screen.get_height() // 2

    lines = [
        "Task Completed!",
        "",
        "Thank you for your participation.",
        "",
        "Press any key or close the window to exit."
    ]
    y_offset = -len(lines) * 25 # Adjust starting y based on number of lines

    for line in lines:
            if line:
                text = font.render(line, True, blue_color)
                text_rect = text.get_rect(center=(screen_center_x, screen_center_y + y_offset))
                screen.blit(text, text_rect)
            y_offset += 50 # Spacing between lines

    pygame.display.flip()
    print("DEBUG: Ending screen displayed, waiting for exit.") # DEBUG

    # Wait for any key press or quit event
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            if event.type == pygame.KEYDOWN:
                waiting = False
        pygame.time.Clock().tick(30) # Prevent busy-waiting
    print("DEBUG: Exiting after ending screen.") # DEBUG


def initialize_pygame():
    """Initialize Pygame and return screen surface."""
    try:
        pygame.init()
        info = pygame.display.Info()
        print(f"Attempting Pygame fullscreen mode: {info.current_w}x{info.current_h}")
        screen = pygame.display.set_mode((info.current_w, info.current_h), 
        pygame.FULLSCREEN | pygame.DOUBLEBUF)
        
        # force pygajme to the front :)
        force_pygame_to_front()

        pygame.display.set_caption("Maze Task")
        screen.fill((255, 255, 255))  # Immediate blank screen
        pygame.display.flip()
        return screen
    except pygame.error as e:
        logging.critical(f"Pygame initialization failed: {e}") # Use logging
        raise RuntimeError(f"Pygame initialization failed: {e}") from e # Re-raise

def load_mazes_safely(maze_path):
    """Load maze data with error handling."""
    try:
        return load_mazes(maze_path)
    except Exception as e:
        raise RuntimeError(f"Maze loading failed: {e}")

def setup_trial(maze_item, screen, subject_id, trial_num, log_dir):
    """Configure all components for a single trial."""
    # Get maze data
    maze_grid, solution_coords = get_maze_and_solution(maze_item)
    if solution_coords is None:
        solution_coords = solve_maze(maze_grid)
    
    # Calculate layout and collisions
    screen_width, screen_height = screen.get_size()
    cell_size, offset_x, offset_y = calculate_layout(screen_width, screen_height, maze_grid.shape)
    walls, _ = create_walls_and_paths(maze_grid, cell_size, (offset_x, offset_y))
    
    # Start/end positions
    start_grid, end_grid = find_start_end_positions(maze_grid)
    start_pos = Vector2(
        offset_x + start_grid[0]*cell_size + cell_size/2,
        offset_y + start_grid[1]*cell_size + cell_size/2
    )
    end_rect = pygame.Rect(
        offset_x + end_grid[0]*cell_size,
        offset_y + end_grid[1]*cell_size,
        cell_size, cell_size
    )
    
    # Create game objects
    pointer = Pointer(start_pos, cell_size*POINTER_RADIUS_RATIO)
    tracker = ExperimentTracker(trial_num, subject_id, log_dir)
    solution_rects = get_solution_path_rects(solution_coords, cell_size, (offset_x, offset_y))
    
    return {
        "maze_grid": maze_grid,
        "walls": walls,
        "pointer": pointer,
        "tracker": tracker,
        "end_rect": end_rect,
        "solution_rects": solution_rects,
        "start_pos": start_pos
    }

def trial_game_loop(screen, pointer, walls, tracker, solution_rects, end_rect):
    """Execute main gameplay phase for a trial."""
    clock = pygame.time.Clock()
    completed = False
    tracker.start_logging()
    
    try:
        while True:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return False  # Trial aborted
            
            # Game state updates
            update_cursor(pointer, walls, tracker, solution_rects)
            
            # Check completion
            if end_rect.collidepoint(pointer.pos):
                if not completed:
                    tracker.stop_logging()
                    return True  # Successful completion
            
            # Rendering
            screen.fill((255, 255, 255))
            for w in walls: pygame.draw.rect(screen, (0,0,0), w)
            pygame.draw.rect(screen, (0,200,0), end_rect)
            pygame.draw.circle(screen, (200,0,0), pointer.pos, int(pointer.radius))
            pygame.display.flip()
            clock.tick(120)
            
    finally:
        if tracker._logging_active:
            tracker.stop_logging()

# In maze_module_experimental.py

def wait_for_mouse_click(screen, message="Click to begin next trial"):
    """Display a message and wait for mouse click to continue."""
    font = pygame.font.Font(None, 48)
    # Text should be visible on a black background
    text_color = (200, 200, 200) # Light grey, not pure white
    bg_color = (0, 0, 0)

    text = font.render(message, True, text_color)
    text_rect = text.get_rect(center=screen.get_rect().center)

    screen.fill(bg_color)
    screen.blit(text, text_rect)
    pygame.display.flip()

    pygame.mouse.set_visible(True)  # Ensure mouse is visible for clicking
    pygame.event.set_grab(False)   # Ensure mouse is not grabbed
    pygame.event.clear()           # Clear any stale events

    waiting = True
    while waiting:
        for event_item in pygame.event.get(): # Renamed to avoid conflict with 'event' module
            if event_item.type == pygame.QUIT:
                pygame.quit()
                sys.exit() # 'sys' must be imported
            if event_item.type == pygame.MOUSEBUTTONDOWN:
                if event_item.button == 1: # Left mouse button
                    waiting = False
            if event_item.type == pygame.KEYDOWN:
                if event_item.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit() # 'sys' must be imported
                elif event_item.key == pygame.K_SPACE: # Allow space to continue
                    waiting = False
        pygame.time.Clock().tick(30)

    # DO NOT set mouse grab/invisible here. Let run_countdown handle it.
    pygame.event.clear() # Clear the click/keypress event
    return True # Indicate success


def run_single_trial(trial_num, maze_item, screen, subject_id, log_dir):
    # ... (setup_trial as before) ...
    try:
        config = setup_trial(maze_item, screen, subject_id, trial_num, log_dir)
        if not config["tracker"].filepath:
            print("Log file for tracker not available. Skipping trial.") # Or log
            return {"success": False, "log_file": None}

        # Participant setup phase (wait screen)
        if not wait_for_mouse_click(screen, f"Trial {trial_num}: Click to begin"):
            # This path isn't really hit if wait_for_mouse_click uses sys.exit on quit
            print(f"Trial {trial_num} wait cancelled.")
            return {"success": False, "log_file": config["tracker"].filepath} # Log file might exist

        # Countdown and actual trial start
        # run_countdown moves the mouse, sets grab, and hides cursor
        run_countdown(screen, 3, config["start_pos"]) 
        send_trigger(1)

        # pygame.event.clear() # Already in run_countdown and wait_for_mouse_click end
        # pygame.mouse.set_pos(config["start_pos"]) # Done by run_countdown
        # pygame.event.set_grab(True) # Done by run_countdown
        # pygame.mouse.set_visible(False) # Done by run_countdown

        # Core gameplay
        success = trial_game_loop(
            screen, config["pointer"], config["walls"],
            config["tracker"], config["solution_rects"], config["end_rect"]
        )
        send_trigger(2)


        return {"success": success, "log_file": config["tracker"].filepath}
        win.flip()

    except Exception as e:
        # Log the full traceback for better debugging
        logging.error(f"Trial {trial_num} error: {str(e)}", exc_info=True)
        # Attempt to ensure tracker is stopped if it was started and an error occurred
        if 'config' in locals() and config and config.get("tracker") and config["tracker"]._logging_active:
            config["tracker"].stop_logging()
        return {"success": False, "log_file": config.get("tracker").filepath if 'config' in locals() and config else None}
    finally:
        # These are crucial for ensuring the system is usable after a Pygame window,
        # especially if it was fullscreen or grabbed input.
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
        # gc.collect() # Already here
        # pygame.event.clear() # Already here
        # print("THREADS:", threading.enumerate()) # Already here


def run_maze_task(
    n_trials: int = 1,
    subject_id: str | None = None,
    maze_path: str = DEFAULT_MAZE_PATH,
    log_dir: str = DEFAULT_LOG_DIR
):
    """
    Run the maze experiment, including an optional 5-minute break at the end.

    Parameters
    ----------
    n_trials : int
        Number of trials to run.
    subject_id : str | None
        External subject-ID override (falls back to get_participant_id()).
    maze_path : str
        Path to the maze-definition file.
    log_dir : str
        Directory for log files.
    """
    try:
        screen = initialize_pygame()
        mazes = load_mazes_safely(maze_path)
        instructions_screen(screen)
        participant_id = subject_id or get_participant_id()

        # ----- trials -----
        trial_results = []
        for trial_num in range(1, n_trials + 1):
            if not mazes:  # ran out of mazes
                break

            result = run_single_trial(
                trial_num,
                mazes.pop(),  # take next maze
                screen,
                participant_id,
                log_dir
            )
            trial_results.append(result)
            time.sleep(1)  # short pause between trials

        # ----- end-of-task screens -----
        ending_screen(screen)
        # ----- clean-up & return -----
        pygame.quit()
        return trial_results  # nothing after this line executes

    except Exception as e:
        logging.critical(f"Experiment aborted: {str(e)}")
        if 'screen' in locals():  # ensure screen exists before quitting
            pygame.quit()
        return []



if __name__ == "__main__":
    import argparse
    import logging
    import os

    parser = argparse.ArgumentParser(description="Run the maze task.")
    parser.add_argument('--subject_id', type=str, required=True, help='Participant ID')
    parser.add_argument('--n_trials', type=int, default=10, help='Number of trials')
    args = parser.parse_args()

    # --- SETTINGS ---
    subject_id = args.subject_id
    n_trials = args.n_trials
    maze_path = "pre_generated_mazes.npy"
    log_dir = "data/maze_logs"

    # --- SETUP ---
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print(f"DEBUG: Starting maze task for Subject: {subject_id} with {n_trials} trials.")

    # --- RUN THE EXPERIMENT ---
    run_maze_task(
        n_trials=n_trials,
        subject_id=subject_id,
        maze_path=maze_path,
        log_dir=log_dir
    )

    print("Maze task script finished execution.")