"""Standalone Arrow Flanker Task (PsychoPy) - CORRECTED

This module is a self-contained, runnable script for an Arrow Flanker Task.

Key changes from the original:
- Fixed the response collection loop by removing `core.wait()`.
- Rebuilt the `main()` function to properly initialize PsychoPy's window and
  ioHub server, allowing the script to be run standalone.
- Added robust cleanup to ensure the window and server close correctly.
- Removed the unused `get_response` helper function.
"""
from __future__ import annotations
import os
import sys
import random
import argparse
from typing import Optional, List, Dict, Any
import pandas as pd
from psychopy import visual, core, event, gui, monitors
from psychopy import iohub
# ==============================================================================
# Constants
# ==============================================================================

# --- Stimulus Definitions ---
# congruency: 0 = congruent, 1 = incongruent
# target_direction: 1 = left, 0 = right
ARROW_STIMULI = [
    {"text": "< < < < <", "congruency": 0, "target_direction": 1},
    {"text": "> > > > >", "congruency": 0, "target_direction": 0},
    {"text": "> > < > >", "congruency": 1, "target_direction": 1},
    {"text": "< < > < <", "congruency": 1, "target_direction": 0},
]

# --- Trigger Codes (Example values) ---
STIM_CONGRUENT_LEFT = 1
STIM_CONGRUENT_RIGHT = 2
STIM_INCONGRUENT_LEFT = 3
STIM_INCONGRUENT_RIGHT = 4
RESP_CORRECT_LEFT = 13
RESP_CORRECT_RIGHT = 14
RESP_INCORRECT_LEFT = 15
RESP_INCORRECT_RIGHT = 16

# ==============================================================================
# Robust Trigger Handling
# ==============================================================================
try:
    import trigger_module  # User-supplied; must have send_trigger(int)
except ImportError:
    class _TriggerFallback:
        """A fallback class that prints trigger codes instead of sending them."""
        def send_trigger(self, code: int) -> None:
            print(f"[Trigger Fallback] send_trigger({code})")
    trigger_module = _TriggerFallback()

# ==============================================================================
# Custom Exception
# ==============================================================================
class UserQuitException(Exception):
    """Raised when participant presses Escape to terminate early."""
    pass

# ==============================================================================
# Helper Functions
# ==============================================================================

def _create_trial_list(base_stimuli: List[Dict], n_congruent: int, n_incongruent: int) -> List[Dict]:
    """Generates and shuffles a balanced list of trials."""
    congruent_stim = [s for s in base_stimuli if s["congruency"] == 0]
    incongruent_stim = [s for s in base_stimuli if s["congruency"] == 1]

    trial_list = []
    if congruent_stim and n_congruent > 0:
        trial_list.extend(random.choices(congruent_stim, k=n_congruent))

    if incongruent_stim and n_incongruent > 0:
        trial_list.extend(random.choices(incongruent_stim, k=n_incongruent))

    random.shuffle(trial_list)
    return trial_list


def get_participant_id_gui(initial_text: str = "") -> Optional[str]:
    """Prompts user for a participant ID via GUI. Returns None if cancelled."""
    dlg = gui.Dlg(title="Participant Information")
    dlg.addField("Participant ID:", initial=initial_text)
    
    dlg_data = dlg.show()
    if not dlg.OK:
        return None  # User pressed Cancel
    
    pid = dlg_data[0]
    return pid.strip() if pid and pid.strip() else None


def show_instructions(win: visual.Window):
    """Displays instructions and waits for the spacebar."""
    instruction_text = (
        "In dieser Aufgabe wird Ihnen eine Reihe von Pfeilen gezeigt.\n"
        "Sie sollen auf den Pfeil in der Mitte reagieren und die äußeren Pfeile ignorieren.\n\n"
        "Zeigt der mittlere Pfeil nach links (<), drücken Sie die 'Y'-Taste (oder 'Z').\n"
        "Zeigt der mittlere Pfeil nach rechts (>), drücken Sie die 'M'-Taste.\n\n"
        "Bitte reagieren Sie so schnell und genau wie möglich.\n\n"
        "Drücken Sie die Leertaste, um die Aufgabe zu starten."
    )
    stim = visual.TextStim(win, text=instruction_text, height=0.05, wrapWidth=1.5, color='white')
    stim.draw()
    win.flip()
    
    keys = event.waitKeys(keyList=['space', 'escape'])
    if 'escape' in keys:
        raise UserQuitException()


# ==============================================================================
# Main Task Runner
# ==============================================================================

def run_flanker_task(
    win,
    io,  # kept for API compatibility; not used for key collection
    participant_id: str,
    n_congruent: int,
    n_incongruent: int,
    max_resp_time: float = 3.0,
    iti: float = 0.2,
    fixation_dur: float = 0.5,
    end_text_height: float = 0.05,
    end_wrap: float = 1.5,
):
    """
    Arrow Flanker Task (event-based key polling version).

    Updated to the 4-bit scheme:
      - In-block trigger codes only (block start is sent by main()):
          1 = STIM_CONGRUENT_LEFT
          2 = STIM_CONGRUENT_RIGHT
          3 = STIM_INCONGRUENT_LEFT
          4 = STIM_INCONGRUENT_RIGHT
          5 = RESP_LEFT
          6 = RESP_RIGHT
      - No outcome (correct/incorrect) triggers; accuracy is logged in the data.

    Parameters
    ----------
    win : psychopy.visual.Window
        Active experiment window (already opened in the main script).
    io : iohub connection or None
        Ignored for key collection (kept for API compatibility).
    participant_id : str
        Participant identifier.
    n_congruent : int
        Number of congruent trials to sample.
    n_incongruent : int
        Number of incongruent trials to sample.
    max_resp_time : float
        Response window in seconds. Default 3.0.
    iti : float
        Inter-trial interval in seconds. Default 0.2.
    fixation_dur : float
        Fixation duration (s). Default 0.5.
    end_text_height : float
        Height for the end message text. Default 0.05 (~5% of screen height).
    end_wrap : float
        Wrap width for end message. Default 1.5.
    """
    # ---- In-block 4-bit codes (1..6) ----
    STIM_CONGRUENT_LEFT = 1
    STIM_CONGRUENT_RIGHT = 2
    STIM_INCONGRUENT_LEFT = 3
    STIM_INCONGRUENT_RIGHT = 4
    RESP_LEFT = 5
    RESP_RIGHT = 6

    pid_str = participant_id.strip().replace(" ", "_")

    # ---------------- Data paths ----------------
    data_dir = os.path.join(os.getcwd(), "data", "flanker_data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, f"flanker_sub-{pid_str}_task-arrowflanker.csv")

    # ---------------- Stimuli ----------------
    fixation = visual.TextStim(win, text="+", height=0.2, color='black')
    flanker_stim = visual.TextStim(win, text="", height=0.15, font='monospace', color='black')

    # ---------------- Trial list ----------------
    all_trials = _create_trial_list(ARROW_STIMULI, n_congruent, n_incongruent)
    if not all_trials:
        print("Warning (Flanker): No trials generated; returning without running task.")
        return

    # ---------------- Instructions ----------------
    win.mouseVisible = False
    show_instructions(win)  # raises UserQuitException on 'escape'

    # ---------------- Response mapping ----------------
    left_keys = ('y', 'z')  # German layout tolerant
    right_keys = ('m',)
    quit_keys = ('escape',)

    results = []
    resp_clock = core.Clock()  # reused; reset on each flip

    try:
        for i, trial in enumerate(all_trials, start=1):
            is_congruent = (trial["congruency"] == 0)
            is_left = (trial["target_direction"] == 1)

            # Determine stimulus trigger (1..4)
            if is_congruent and is_left:
                stim_trigger = STIM_CONGRUENT_LEFT
            elif is_congruent and not is_left:
                stim_trigger = STIM_CONGRUENT_RIGHT
            elif not is_congruent and is_left:
                stim_trigger = STIM_INCONGRUENT_LEFT
            else:
                stim_trigger = STIM_INCONGRUENT_RIGHT

            # ----- Fixation -----
            fixation.draw()
            win.flip()
            core.wait(fixation_dur)

            # ----- Stimulus -----
            flanker_stim.setText(trial['text'])
            flanker_stim.draw()

            # Reset RT clock and send stimulus trigger exactly on flip
            win.callOnFlip(resp_clock.reset)
            win.callOnFlip(trigger_module.send_trigger, stim_trigger)
            stim_onset = win.flip()

            # ----- Response loop -----
            response = None
            rt_ms = None
            accuracy = -1  # -1 = timeout
            response_trigger = 0  # 5/6 if a valid response was made

            event.clearEvents(eventType='keyboard')

            while resp_clock.getTime() < max_resp_time:
                keys = event.getKeys(timeStamped=resp_clock)
                if not keys:
                    continue
                for key, t in keys:
                    key = key.lower()
                    if key in quit_keys:
                        raise UserQuitException()
                    if key in left_keys:
                        response = 'y'
                        rt_ms = t * 1000.0
                        response_trigger = RESP_LEFT
                        trigger_module.send_trigger(response_trigger)
                        break
                    if key in right_keys:
                        response = 'm'
                        rt_ms = t * 1000.0
                        response_trigger = RESP_RIGHT
                        trigger_module.send_trigger(response_trigger)
                        break
                if response is not None:
                    break

            # ----- Score (no outcome triggers) -----
            if response is not None:
                is_correct = (response == 'y' and is_left) or (response == 'm' and not is_left)
                accuracy = 1 if is_correct else 0
            else:
                accuracy = -1  # timeout

            # ----- ITI -----
            win.flip(clearBuffer=True)
            core.wait(iti)

            # ----- Log trial -----
            results.append({
                "participant_id": pid_str,
                "trial_number": i,
                "congruency": "congruent" if is_congruent else "incongruent",
                "target_direction": "left" if is_left else "right",
                "stimulus": trial["text"],
                "response": response,       # 'y' | 'm' | None
                "accuracy": accuracy,       # 1=correct, 0=incorrect, -1=timeout
                "rt_ms": rt_ms,             # None on timeout
                "stimulus_trigger": stim_trigger,     # 1..4
                "response_trigger": response_trigger, # 0 (none) or 5/6
                "stim_onset_ts": stim_onset,          # PsychoPy time, if you want it
            })

        print(f"Flanker task complete. {len(results)} trials collected.")

    except UserQuitException:
        print("Participant terminated Flanker early (escape). Partial data will be saved.")

    finally:
        # ---------------- Save data ----------------
        if results:
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"Flanker data saved: {output_path}")
        else:
            print("Flanker: No data to save.")

        # ---------------- End message ----------------
        win.mouseVisible = True
        thank_you_stim = visual.TextStim(
            win,
            text="Task complete.\nDrücken Sie LEERTASTE, um fortzufahren.",
            height=end_text_height,
            wrapWidth=end_wrap,
            color='black'
        )
        thank_you_stim.draw()
        win.flip()
        event.waitKeys(keyList=['space', 'escape'])





# ==============================================================================
# Command-Line Interface (Corrected for Standalone Execution)
# ==============================================================================
def main():
    """Parses command-line arguments and runs the task."""
    parser = argparse.ArgumentParser(description="Run the Arrow Flanker Task.")
    parser.add_argument(
        "participant_id", nargs="?", default=None,
        help="Participant ID. If not provided, a GUI dialog will appear."
    )
    parser.add_argument("--n-congruent", type=int, default=40, help="Number of congruent trials.")
    parser.add_argument("--n-incongruent", type=int, default=40, help="Number of incongruent trials.")
    parser.add_argument("--fullscreen", action="store_true", help="Run in fullscreen mode.")
    parser.add_argument("--no-gui", dest='use_gui', action="store_false", help="Do not show GUI for ID input.")
    
    args = parser.parse_args()

    pid = args.participant_id
    if not pid and args.use_gui:
        pid = get_participant_id_gui()
    
    if not pid:
        print("Error: A participant ID is required.")
        sys.exit(1)

    # --- Setup PsychoPy Window and ioHub Server ---
    win = None
    io = None
    try:
        io = iohub.launchHubServer()
        win = visual.Window(
            size=[800, 600],
            fullscr=args.fullscreen,
            units="height",
            color="grey"
        )
        
        # Run the main task
        run_flanker_task(
            win=win,
            io=io,
            participant_id=pid,
            n_congruent=args.n_congruent,
            n_incongruent=args.n_incongruent,
        )

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # --- Cleanup ---
        if io:
            io.quit()
        if win:
            win.close()
        core.quit()

if __name__ == "__main__":
    main()