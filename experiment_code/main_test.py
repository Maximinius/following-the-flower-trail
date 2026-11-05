#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 18:21:02 2025

@author: max
"""

# Imports
import os
import sys
import csv
import random
from psychopy import visual, core, event, gui
from psychopy.iohub import launchHubServer # NEW: Import for centralized iohub
import subprocess 
import flanker_module
import gradCPT_module
import flower_module
import Wisconsin_module as wcst
import paper_flower_module as paper_flower
from break_module import run_break_routine
from trigger_module import send_trigger




# =============================================================================
# Helper functions (show_wait_screen, get_participant_id, etc.)
# These functions remain largely the same, but we will ensure they
# are always called with the main 'win' object.
# =============================================================================

def show_wait_screen(win, text="The experiment is ready.\n\nPlease wait for the researcher to begin.", key_to_wait_for='9'):
    """Displays a waiting screen and waits for a specific key press to continue."""
    wait_message = visual.TextStim(win, text=text, height=0.05, color='white', wrapWidth=1.5)
    
    keep_waiting = True
    while keep_waiting:
        wait_message.draw()
        win.flip()
        
        keys = event.getKeys(keyList=[key_to_wait_for, 'escape'])
        if key_to_wait_for in keys:
            print(f"Key '{key_to_wait_for}' pressed. Continuing to experiment.")
            keep_waiting = False
        elif 'escape' in keys:
            print("Escape key pressed. Exiting experiment.")
            core.quit()

def get_participant_id():
    """Gets the participant ID via a simple dialog box."""
    dlg = gui.Dlg(title="Participant ID")
    dlg.addField("Subject Number:")
    dlg.addField("Questionnaire ID:") 
    dlg.show()
    if dlg.OK:
        return dlg.data[0]
    else:
        core.quit()
        return None

def instruction_screens(win, screens, key_to_wait_for='space'):
    """Displays a series of instruction screens."""
    message = visual.TextStim(win, text='', height=0.05, color='white', wrapWidth=1.8)
    prompt_text = f"\n\nDrücken Sie die LEERTASTE, um fortzufahren."

    for screen_data in screens:
        title = screen_data.get("title", "")
        content_list = screen_data.get("content", [])
        content_text = "\n".join(content_list)
        full_text = f"{title}\n\n{content_text}{prompt_text}"
        message.setText(full_text)
        
        event.clearEvents(eventType='keyboard')
        
        keep_waiting = True
        while keep_waiting:
            message.draw()
            win.flip()
            
            keys = event.getKeys(keyList=[key_to_wait_for, 'escape'])
            if key_to_wait_for in keys:
                keep_waiting = False
            elif 'escape' in keys:
                print("Escape key pressed. Exiting experiment.")
                core.quit()
    
    win.flip()

# Instruction screen dictionaries 

# Bildschirm 1 – Überblick
screen1 = {
    "title": "",
    "content": [
        "Sie absolvieren 6 Runden.",
        "Jede Runde besteht aus einer kurzen Aufgabe und anschließend 3 Minuten Ruhe (Augen geschlossen) zur EEG‑Aufzeichnung.",
        "Bitte folgen Sie den Anweisungen auf dem Bildschirm."
    ]
}


# Bildschirm 2 – Sitzhaltung
screen2 = {
    "title": "Sitzposition",
    "content": [
        "Nehmen Sie eine stabile, bequeme Sitzposition ein:",
        "- Rücken anlehnen und aufrecht sitzen",
        "- Beide Füße flach auf den Boden stellen",
        "- Hände locker auf den Tisch legen"
    ]
}

# Bildschirm 3 – Geräte am Platz
screen3 = {
    "title": "Geräte am Platz",
    "content": [
        "Folgende Geräte liegen bereit:",
        "- Stift und Papier",
        "- Maus",
        "- Grafiktablett",
        "- Tastatur",
        "- Ständer",
        "",
        "Bitte machen Sie sich mit diesen Geräten vertraut – Sie benötigen sie während der Aufgaben."
    ]
}

# Bildschirm 4 – Ablauf eines Durchgangs
screen4 = {
    "title": "Ablauf eines Durchgangs",
    "content": [
        "Jeder der 6 Durchgänge besteht aus:",
        "1. Aufgabe",
        "2. Kurze Phase zum Anpassen Ihrer Sitzposition oder Geräte",
        "3. 3‑minütige Ruhephase (Augen geschlossen)"
    ]
}

# Bildschirm 5 – Bewegung & Ruhe
screen5 = {
    "title": " ",
    "content": [
        "Für saubere EEG‑Daten bitte so ruhig wie möglich bleiben:",
        "- Nicht wackeln, drehen oder den Stuhl rollen",
        "- Rücken angelehnt, Füße flach",
        "- Große Blinzler, Kieferbewegungen und Schlucken minimieren",
        "",
        "Bewegen Sie sich nur, wenn Sie in den Aufgaben ausdrücklich dazu aufgefordert werden."
    ]
}

# Bildschirm 6 – Zusammenfassung & Start
screen6 = {
    "content": [
        "Kurz zusammengefasst:",
        "- 6 Durchgänge (Aufgabe + 5‑min‑Ruhe)",
        "- Bildschirm‑Anweisungen Schritt für Schritt befolgen",
        "- Während jeder Ruhephase: Augen schließen & still bleiben",
        "",
        "Bereit? Wir beginnen mit der ersten Aufgabe."
    ]
}


def display_end_screen(win):
    """Shows a message that the task block has ended."""
    title = visual.TextStim(win, text="Block Beendet", pos=(0.0, 0.1), height=0.10, color="white", bold=True)
    prompt = visual.TextStim(win, text="Drücken Sie die Leertaste, um die Pause zu beginnen.", pos=(0.0, -0.05), height=0.06, wrapWidth=1.6, color="white")

    while True:
        title.draw()
        prompt.draw()
        win.flip()
        keys = event.getKeys(keyList=["space", "escape"])
        if "space" in keys:
            break
        if "escape" in keys:
            core.quit()
    win.flip()

def display_final_screen(win):
    """Shows a final thank you message and waits for the experimenter to press '9'."""
    from psychopy import visual, event, core

    message = visual.TextStim(
        win, 
        text="Das Experiment ist nun beendet.\n\nWir sind gleich bei Ihnen. Bitte bleiben Sie auf Ihrem Platz sitzen.",
        height=0.07, 
        color='pink'
    )
    message.draw()
    win.flip()

    # Wait until the '9' key is pressed
    while True:
        keys = event.getKeys()
        if '9' in keys:
            break
        core.wait(0.1)


def configure_experiment_environment():
    """Sets the script's directory as the current working directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    os.chdir(script_dir)
    print(f"Working directory and module path set to: {script_dir}")


def run_flower_task(win, io, participant_id, version=0, **kwargs):
    """
    Wrapper around ``flower_module.run_flower_drawing`` that temporarily
    minimises the main PsychoPy window so the pygame window (mouse or tablet
    version) jumps to the foreground, and then restores the PsychoPy window
    when the task is finished.

    Parameters
    ----------
    win : psychopy.visual.Window
        The experiment’s main window.
    io : psychopy.iohub.client.ioHubConnection
        Shared ioHub connection (unused here, kept for signature symmetry).
    participant_id : str
        The subject identifier.
    version : int, optional
        1 = mouse version, 2 = tablet version, anything else = default.
    **kwargs
        Any extra arguments forwarded to ``flower_module.run_flower_drawing``.
    """
    import platform
    from psychopy import core

    print(f"\n--- Preparing to launch Flower Drawing (version {version}) "
          f"for participant {participant_id} ---")

    # --------------------------------------------------------------------- #
    # 1) Minimise the main PsychoPy window so the pygame window won’t start
    #    hidden behind it.
    # --------------------------------------------------------------------- #
    win.fullscr = False
    win.winHandle.minimize()
    win.flip()
    core.wait(0.5)  # give the OS a moment to register the change

    try:
        # ----------------------------------------------------------------- #
        # 2) Run the actual flower‑drawing task.
        # ----------------------------------------------------------------- #
        flower_module.run_flower_drawing(
            win=win, io=io, participant_id=participant_id,
            version=version, **kwargs
        )
        print("--- Flower Drawing completed successfully ---")

    except Exception as e:
        print("--- 🐛 ERROR! Flower Drawing failed and was skipped. ---")
        print(f"--- Error details: {e} ---")
        raise  # let the outer try/except handle logging & recovery

    finally:
        # ----------------------------------------------------------------- #
        # 3) Restore the PsychoPy window.
        # ----------------------------------------------------------------- #
        print("Restoring PsychoPy window…")
        win.fullscr = True
        win.winHandle.maximize()
        win.winHandle.activate()
        win.flip()
        core.wait(0.5)

        # 4) Extra safeguard to bring it to the front (OS‑specific).
        try:
            if platform.system() == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = win.winHandle.handle
                user32.ShowWindow(hwnd, 5)          # SW_SHOW
                user32.SetForegroundWindow(hwnd)
            elif platform.system() == "Darwin":      # macOS
                import os
                os.system(
                    f"""/usr/bin/osascript -e 'tell app "System Events" to """
                    f"""set frontmost of the first process whose unix id is {os.getpid()} to true'"""
                )
        except Exception as e2:
            print(f"Could not force window to front: {e2}")


def run_maze_subprocess(win, participant_id, io, n_trials=10, **kwargs):
    """
    Runs the maze task in a separate process while keeping its window in the foreground.
    """
    import platform
    import time
    import subprocess
    import sys
    from psychopy import core

    print(f"\n--- Preparing to launch Maze Task Subprocess for participant {participant_id} ---")
    
    # 1. Minimize PsychoPy window
    print("Minimizing PsychoPy window...")
    win.fullscr = False
    win.winHandle.minimize()
    win.flip()
    core.wait(0.5)  # allow OS to register minimize

    # 2. Prepare the command
    command = [
        sys.executable,
        'maze_module_experimental.py',
        '--subject_id', str(participant_id),
        '--n_trials', str(n_trials)
    ]
    print(f"Executing command: {' '.join(command)}")

    try:
        # 3. Run the subprocess
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        print("--- Maze Task Subprocess Completed Successfully ---")
        print("Maze Output:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"--- CRITICAL ERROR in Maze Task Subprocess ---")
        print(f"Return Code: {e.returncode}")
        print("--- Maze STDOUT ---\n", e.stdout)
        print("--- Maze STDERR ---\n", e.stderr)
        core.quit()

    finally:
        # 4. Restore the PsychoPy window
        print("Restoring PsychoPy window...")
        win.fullscr = True
        win.winHandle.maximize()
        win.winHandle.activate()
        win.flip()
        core.wait(0.5)  # small pause to allow restore

        # 5. Force PsychoPy window to foreground (optional safeguard)
        try:
            if platform.system() == "Windows":
                import ctypes
                user32 = ctypes.windll.user32
                hwnd = win.winHandle.handle
                user32.ShowWindow(hwnd, 5)  # SW_SHOW
                user32.SetForegroundWindow(hwnd)
            elif platform.system() == "Darwin":  # macOS
                import os
                os.system(
                    '''/usr/bin/osascript -e 'tell app "System Events" to set frontmost of the first process whose unix id is {} to true' '''.format(os.getpid())
                )
        except Exception as e:
            print(f"Could not force window to front: {e}")




def main():
    """
    Run the full experiment within a single PsychoPy instance.

    4-bit trigger policy (Brain Products wireless):
        Block starts (unique per task):
            9  = flanker
            10 = flower_mouse
            11 = flower_tablet
            12 = flower_paper
            13 = maze
            14 = gradcpt
            15 = wcst
        Breaks (global, between blocks only):
            7  = BREAK START
            8  = BREAK END
        In-block event codes available to tasks: 1..6
            (Accuracy/outcome to be logged in PsychoPy data instead of 7/8.)

    The function:
      - randomizes task order,
      - sends the task’s unique block-start code before the task runs,
      - brackets the 5-minute resting-state with BREAK START/END (7/8),
      - logs both task order and a breaks log that explicitly links each break to the preceding task.
    """
    # ---- constants (wire level) ----
    BLOCK_START = {
        'flanker': 9,
        'mouse_flower': 10,
        'tablet_flower': 11,
        'paper_flower': 12,
        'maze': 13,
        'gradcpt': 14,
        'wcst': 15,
    }
    BREAK_START = 7
    BREAK_END = 8

    configure_experiment_environment()

    io = None
    win = None

    try:
        # 1) Participant ID
        participant_id = get_participant_id()
        if not participant_id:
            print("No participant ID entered. Exiting.")
            return

        # 2) ioHub + window
        print("Initializing ioHub and main experiment window...")
        io = launchHubServer()
        win = visual.Window(fullscr=True, units='height', color='grey', useRetina=True, waitBlanking=True)

        # 3) Initial wait + instructions
        show_wait_screen(win, text="Experiment ist bereit.\n\nWarten auf Versuchsleiter.", key_to_wait_for='9')
        all_instructions = [screen1, screen2, screen3, screen4, screen5, screen6]
        instruction_screens(win, all_instructions)
        # Optional small marker to flag "end of instructions" (must be 1..15). Avoid 7/8, 9..15.
        send_trigger(6)  # optional

        # 4) Define & randomize tasks (keep unique flower modalities!)
        tasks = [
            {'name': 'paper_flower',  'function': paper_flower.run_task,                'params': {}},
            {'name': 'mouse_flower',  'function': run_flower_task,                      'params': {'version': 1}},
            {'name': 'tablet_flower', 'function': run_flower_task,                      'params': {'version': 2}},
            {'name': 'maze',          'function': run_maze_subprocess,                  'params': {'n_trials': 10}},
            {'name': 'flanker',       'function': flanker_module.run_flanker_task,      'params': {'n_congruent': 200, 'n_incongruent': 100}},
            {'name': 'gradcpt',       'function': gradCPT_module.run_gradCPT_task,      'params': {}},
            {'name': 'wcst',          'function': wcst.run_wcst,                        'params': {}},
        ]
        random.shuffle(tasks)

        # 5) Log randomized order
        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        csv_path = os.path.join(data_dir, f"sub-{participant_id}_task-order_log.csv")
        header = ["participant_id"] + [f"order_{i+1}" for i in range(len(tasks))]
        order_row = [participant_id] + [task['name'] for task in tasks]
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            writer.writerow(order_row)
        print(f"Task order logged to {csv_path}")

        # Breaks log (explicit linkage for analysis convenience)
        breaks_csv_path = os.path.join(data_dir, f"sub-{participant_id}_breaks_log.csv")
        with open(breaks_csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["participant_id", "break_index", "after_task", "break_start_time_s", "break_end_time_s"])

        exp_clock = core.Clock()
        break_index = 0

        # 6) Run tasks → always follow with 3-min break
        for i, task in enumerate(tasks):
            task_name = task['name']
            block_code = BLOCK_START[task_name]

            print(f"\n--- Starting task {i+1}/{len(tasks)}: {task_name} for participant {participant_id} ---")

            # Unique block-start marker for this task (4-bit)
            send_trigger(block_code)

            try:
                task_func = task['function']
                task_params = task.get('params', {})
                task_func(win=win, io=io, participant_id=participant_id, **task_params)
                print(f"--- Completed: {task_name} ---")
            except Exception as e:
                print(f"--- 🐛 ERROR! Task '{task_name}' failed and was skipped. ---")
                print(f"--- Continuing with the next task. ---")
                print(f"--- Error details: {e} ---")

            # End-of-task screen
            display_end_screen(win)

            # BREAK: explicit markers (7/8), do NOT use 0 as an event
            print(f"--- Starting 5-minute break after '{task_name}' (Triggers: {BREAK_START} -> {BREAK_END}) ---")
            send_trigger(BREAK_START)
            break_start_t = exp_clock.getTime()

            # Show 5-min rest; ensure this routine does not send its own triggers if we're handling them here
            run_break_routine(
                win=win,
                duration=3,       # minutes
                start_trigger=None,
                end_trigger=None
            )

            send_trigger(BREAK_END)
            break_end_t = exp_clock.getTime()

            # Persist explicit linkage (redundant with event-order logic, but convenient for MNE)
            break_index += 1
            with open(breaks_csv_path, "a", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow([participant_id, break_index, f"after_{task_name}", f"{break_start_t:.3f}", f"{break_end_t:.3f}"])

        print("\nAll tasks and resting-state periods completed.")
        # 7) Final end screen
        display_final_screen(win)

    except Exception as e:
        print(f"CRITICAL ERROR during experiment: {e}")
    finally:
        print("--- Shutting down experiment ---")
        if win:
            win.close()
        if io:
            io.quit()
        core.quit()


if __name__ == "__main__":
    main()