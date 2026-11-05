from psychopy import visual, core
from psychopy.iohub import launchHubServer
import os
import csv
import sys
from trigger_module import send_trigger
import pygame
from show_example import show_example_animation, screen_res


# ------------------------------------------------------------------
# 1. tablet_stand_setup_screen  –  fixed key normalisation
# ------------------------------------------------------------------
from psychopy import visual, core

# ------------------------------------------------------------------
# 1. tablet_stand_setup_screen  –  counts only real key‑downs now
# ------------------------------------------------------------------
from psychopy import visual, core

def tablet_stand_setup_screen(
    win,
    io,
    *,
    proceed_keys=('space',),
    escape_keys=('escape',),
    max_gap=3.0,
    poll_interval=0.01,
    text_color='black',
    highlight_color='green',
    debug=False,
):
    """
    Preparation: show stand‑setup instructions and wait until SPACE is pressed
    twice with ≤ `max_gap` s between key‑**down** events.

    Returns
    -------
    bool
        True  – double‑press registered  
        False – aborted (ESC pressed or window closed)
    """
    kb = io.devices.keyboard

    # ── key normalisation ──────────────────────────────────────────
    SPACE_ALIASES = {' ', 'space', 'spacebar'}

    def _norm(k):
        ks = str(k)
        if ks == ' ':
            return 'space'
        ks = ks.strip().lower()
        return 'space' if ks in SPACE_ALIASES else ks

    proceed_norm = {_norm(k) for k in proceed_keys}
    escape_norm  = {_norm(k) for k in escape_keys}

    # ── text stimuli ──────────────────────────────────────────────
    lines = [
        "1. Platzieren Sie den Ständer auf dem Tisch so,",
        "   dass seine Füße GENAU über den Klebeband‑Markierungen stehen.",
        "2. Achten Sie darauf, dass der Ständer stabil ist und sich nicht mehr bewegt.",
        "3. Legen Sie das papierstabel in den Ständer, falls noch nicht geschehen.",
        "",
        "Wenn alles korrekt positioniert ist:",
        ">> Drücken Sie ZWEIMAL die LEERTASTE, um fortzufahren. <<",
        "",
        "Bei Problemen: Rufen Sie die Versuchsleitung.",
    ]

    stim_list, y = [], 0.35
    for i, txt in enumerate(lines):
        stim_list.append(
            visual.TextStim(
                win,
                text=txt,
                pos=(0, y),
                height=0.08 if i == 0 else 0.05,
                color=highlight_color if txt.startswith('>>') else text_color,
                wrapWidth=1.8,
                anchorVert='top',
            )
        )
        y -= (0.08 if i == 0 else 0.05) * 1.4

    prog = visual.TextStim(
        win, text="SPACE: 0/2", pos=(0, -0.85), height=0.05, color=text_color
    )

    # ── main loop ─────────────────────────────────────────────────
    kb.clearEvents()
    space_cnt, last_press_time = 0, None
    win.color = 'white'

    while True:
        # draw frame
        for s in stim_list:
            s.draw()
        prog.text = f"SPACE: {space_cnt}/2"
        prog.draw()
        win.flip()

        # handle input
        for ev in kb.getEvents():
            if ev.duration:      # ignore key‑up (duration > 0)
                continue
            key = _norm(ev.key)
            if debug:
                print(f"[tablet_setup] {ev.event_type} key={ev.key!r}->{key!r} t={ev.time:.3f}")
            if key in escape_norm:
                return False
            if key in proceed_norm:
                now = ev.time
                if last_press_time is None or (now - last_press_time) <= max_gap:
                    space_cnt += 1
                else:
                    space_cnt = 1
                last_press_time = now
                if space_cnt >= 2:
                    return True
        core.wait(poll_interval)


# ------------------------------------------------------------------
# 2. tablet_stand_teardown_screen  –  same single‑tap safeguard
# ------------------------------------------------------------------
def tablet_stand_teardown_screen(
    win,
    io,
    *,
    proceed_keys=('space',),
    escape_keys=('escape',),
    max_gap=3.0,
    poll_interval=0.01,
    text_color='black',
    highlight_color='green',
    debug=False,
):
    """
    Preparation: show stand‑setup instructions and wait until SPACE is pressed
    twice with ≤ `max_gap` s between key‑**down** events.

    Returns
    -------
    bool
        True  – double‑press registered  
        False – aborted (ESC pressed or window closed)
    """
    kb = io.devices.keyboard

    # ── key normalisation ──────────────────────────────────────────
    SPACE_ALIASES = {' ', 'space', 'spacebar'}

    def _norm(k):
        ks = str(k)
        if ks == ' ':
            return 'space'
        ks = ks.strip().lower()
        return 'space' if ks in SPACE_ALIASES else ks

    proceed_norm = {_norm(k) for k in proceed_keys}
    escape_norm  = {_norm(k) for k in escape_keys}

    # ── text stimuli ──────────────────────────────────────────────
    lines = [
        "Bitte stellen Sie den Ständer jetzt zurück",
        "an seine Ausgangsposition",
        "(weg vom Tablet, auf die markierte Ablagefläche).",
        "",  # This will act as a blank line for spacing
        "Vergewissern Sie sich, dass der Ständer",
        "NICHT mehr auf den Klebeband-Markierungen steht.",
        "",
        "Wenn der Ständer korrekt zurückgestellt ist:",
        ">> Drücken Sie ZWEIMAL die LEERTASTE, um fortzufahren. <<",
    ]

    stim_list, y = [], 0.35
    for i, txt in enumerate(lines):
        stim_list.append(
            visual.TextStim(
                win,
                text=txt,
                pos=(0, y),
                height=0.08 if i == 0 else 0.05,
                color=highlight_color if txt.startswith('>>') else text_color,
                wrapWidth=1.8,
                anchorVert='top',
            )
        )
        y -= (0.08 if i == 0 else 0.05) * 1.4

    prog = visual.TextStim(
        win, text="SPACE: 0/2", pos=(0, -0.85), height=0.05, color=text_color
    )

    # ── main loop ─────────────────────────────────────────────────
    kb.clearEvents()
    space_cnt, last_press_time = 0, None
    win.color = 'white'

    while True:
        # draw frame
        for s in stim_list:
            s.draw()
        prog.text = f"SPACE: {space_cnt}/2"
        prog.draw()
        win.flip()

        # handle input
        for ev in kb.getEvents():
            if ev.duration:      # ignore key‑up (duration > 0)
                continue
            key = _norm(ev.key)
            if debug:
                print(f"[tablet_setup] {ev.event_type} key={ev.key!r}->{key!r} t={ev.time:.3f}")
            if key in escape_norm:
                return False
            if key in proceed_norm:
                now = ev.time
                if last_press_time is None or (now - last_press_time) <= max_gap:
                    space_cnt += 1
                else:
                    space_cnt = 1
                last_press_time = now
                if space_cnt >= 2:
                    return True
        core.wait(poll_interval)







def draw_page_turn_screen(win, trial_num, press_prompt, text_color, last_rt=None):
    """
    Show a screen instructing participants to turn to the sheet for `trial_num`.
    Optionally displays the time from the previous trial.

    Parameters
    ----------
    win : psychopy.visual.Window
    trial_num : int
    press_prompt : str
        Localised text for the key the participant must press.
    text_color : str
    last_rt : float or None
    """
    feedback_text = ""
    if last_rt is not None:
        feedback_text = (
            f"Durchgang {trial_num - 1} beendet!\n"
            f"Zeit: {last_rt:.2f}s\n\n"
        )

    page_text = (
        f"{feedback_text}"
        f"Bitte drehen Sie nun den Papierstapel so,\n"
        f"dass Blatt {trial_num} oben liegt.\n\n"
        f"Drücken Sie {press_prompt}, um fortzufahren."
    )
    stim = visual.TextStim(
        win, text=page_text, wrapWidth=1.8, color=text_color, height=0.07
    )
    stim.draw()


def run_task(
    win,
    participant_id,
    io,
    num_trials=10,
    text_color='black',
    proceed_keys=('space',),
    escape_keys=('escape',),
    poll_interval=0.01,
    debug=False,
    **kwargs
):
    """
    Paper Flower Task (production, with page‑turn prompt).

    … (docstring unchanged except now includes page‑turn step) …
    """
    print(f"--- Running Paper Flower Task for participant {participant_id} ---")

    kb = io.devices.keyboard

    # ------------------------------------------------------------
    # Key normalization
    # ------------------------------------------------------------
    SPACE_ALIASES = {' ', 'space', 'spacebar'}
    def _norm_key(k):
        """Lowercase, strip, and collapse spacebar aliases to 'space'."""
        if k is None:
            return None
        ks = str(k)
        if ks == ' ':
            return 'space'
        ks = ks.strip().lower()
        if ks in SPACE_ALIASES:
            return 'space'
        return ks

    proceed_norm = None if proceed_keys is None else {_norm_key(k) for k in proceed_keys}
    escape_norm = {_norm_key(k) for k in escape_keys}
    allow_any = proceed_norm is None

    # ------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------
    
    def _save_data(data, filename):
        """Save a list of dictionaries to a CSV file."""
        if not data:
            print("[_save_data] No data to save.")
            return

        # Use the keys from the first trial's dictionary as headers
        headers = data[0].keys()
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(data)
            print(f"--- Data successfully saved to {filename} ---")
        except Exception as e:
            print(f"*** CRITICAL: FAILED TO SAVE DATA to {filename} ***")
            print(f"*** Reason: {e} ***")

    def _drain_all(tag):
        total = 0
        while True:
            evs = kb.getEvents()
            if not evs:
                break
            total += len(evs)
        if debug:
            print(f"[{tag}] drained {total} event(s).")
        return total

    def _wait_for_key(tag):
        """Wait for a key-**down** event (ignore key-ups) matching proceed or escape."""
        while True:
            evs = kb.getEvents()
            if evs:
                for ev in evs:
                    # Ignore key-up: iohub sets duration>0 for releases
                    if getattr(ev, "duration", None):
                        if debug:
                            print(f"[{tag}] IGNORE key-up key={ev.key!r} dur={ev.duration:.3f} t={ev.time:.6f}")
                        continue

                    k_raw = ev.key
                    k = _norm_key(k_raw)
                    if debug:
                        print(f"[{tag}] key-down key={k_raw!r} -> norm={k!r} t={ev.time:.6f}")
                    if k in escape_norm:
                        print(f"*** Escape detected ({tag}). Aborting. ***")
                        return None, None
                    if allow_any or (proceed_norm and k in proceed_norm):
                        return k, ev
            core.wait(poll_interval)

    # ------------------------------------------------------------
    # Data output path
    # ------------------------------------------------------------
    data_dir = os.path.join("data", "paper_flower_data")
    os.makedirs(data_dir, exist_ok=True)
    outfile = os.path.join(
        data_dir, f"sub-{participant_id}_task-paperflower_events.csv"
    )
    results = []
    
    # Instruct participants to put the tablet on the table
    if not tablet_stand_setup_screen(win, io,
                                      proceed_keys=proceed_keys,
                                      escape_keys=escape_keys,
                                      debug=debug):
        print("=== Aborted during tablet‑stand setup ===")
        return None

    # *** IMPORTANT: drop any leftover events from the double-SPACE ***
    kb.clearEvents()
    if debug:
        print("[post-setup] keyboard events cleared.")

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

    # -------------------------
    # Run Pygame example animation
    # -------------------------
    pygame.init()
    try:
        pygame_screen = pygame.display.set_mode(screen_res)
        pygame.display.set_caption("Flower Tracing Example (Paper Task)") # A distinct caption
        pygame_clock = pygame.time.Clock()
        show_example_animation(pygame_screen, pygame_clock)
    finally:
        pygame.quit()

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

    # ------------------------------------------------------------
    # Instructions
    # ------------------------------------------------------------
    if allow_any:
        press_txt = "eine Taste"
    else:
        press_txt = "die Leertaste" if proceed_norm == {'space'} else " / ".join(sorted(proceed_norm))

    instr_text = (
        "In dieser Aufgabe zeichnen Sie die Blume auf Papier nach.\n\n"
        "Die Blume besteht aus zwei Linien, die einen Weg bilden.\n"
        "Ihre Aufgabe ist es, eine Linie ZWISCHEN diesen beiden Linien zu zeichnen, von Anfang bis Ende.\n"
        "Geben Sie Ihr Bestes, die Linien nicht zu berühren oder zu überkreuzen.\n\n"
        f"Sie werden insgesamt {num_trials} Durchgänge durchführen.\n"
        f"Drücken Sie nach jeder Zeichnung so schnell wie möglich {press_txt}.\n\n"
        f"Wenn Sie bereit sind, drücken Sie {press_txt}, um zu beginnen."
    )
    instr_stim = visual.TextStim(
        win, text=instr_text, wrapWidth=1.8, color=text_color, height=0.06
    )
    instr_stim.draw()
    win.flip()

    # Drain once more after the flip, in case a release got timestamped late.
    _drain_all("pre-instructions")
    print("[instructions] waiting for start key...")
    start_key, start_ev = _wait_for_key("instructions")
    if start_key is None:
        print("=== Aborted at instructions ===")
        return None
    print(f"[instructions] got key={start_key!r} t={start_ev.time:.6f}")

    # ------------------------------------------------------------
    # Main trial loop
    # ------------------------------------------------------------
    last_rt = None
    for trial in range(1, num_trials + 1):
        print(f"\n--- Trial {trial}/{num_trials} ---")

        # --------------------------------------------------------
        # Page‑turn prompt
        # --------------------------------------------------------
        if allow_any:
            press_prompt = "eine Taste"
        else:
            press_prompt = (
                "die Leertaste" if proceed_norm == {'space'} else " / ".join(sorted(proceed_norm))
            )

        draw_page_turn_screen(win, trial, press_prompt, text_color, last_rt=last_rt)
        win.flip()

        _drain_all(f"trial{trial}-pre-page-turn")
        page_key, page_ev = _wait_for_key(f"trial{trial}-page-turn")
        if page_key is None:
            print(f"=== Aborted during trial {trial} page‑turn ===")
            _save_data(results, outfile)
            return results if results else None
        print(f"[trial {trial}] page‑turn key={page_key!r} t={page_ev.time:.6f}")
        
        # --------------------------------------------------------
        # [NEW] SIT STILL prompt
        # --------------------------------------------------------
        sit_still_text = (
            "Hände an die Startposition.\n\n"
            "WICHTIG:\nBitte bewegen Sie sich während des Countdowns (5..1)\nNICHT mehr.\n\n"
            "Sitzen Sie ganz still, bis die Zeichen-Aufforderung erscheint.\n\n"
            f"Drücken Sie {press_prompt}, wenn Sie bereit sind."
        )
        sit_still_stim = visual.TextStim(
            win, text=sit_still_text, wrapWidth=1.8, color=text_color, height=0.07
        )
        sit_still_stim.draw()
        win.flip()

        _drain_all(f"trial{trial}-pre-sit-still")
        still_key, still_ev = _wait_for_key(f"trial{trial}-sit-still")
        if still_key is None:
            print(f"=== Aborted during trial {trial} sit-still prompt ===")
            _save_data(results, outfile)
            return results if results else None
        print(f"[trial {trial}] sit-still key={still_key!r} t={still_ev.time:.6f}")
        # --------------------------------------------------------
        # [END NEW]
        # --------------------------------------------------------


        # --------------------------------------------------------
        # COUNTDOWN 5..1
        # --------------------------------------------------------
        for count in range(5, 0, -1):
            print(f"[trial {trial}] countdown {count}")
            countdown_stim = visual.TextStim(
                win, text=str(count), color=text_color, height=0.3
            )
            countdown_stim.draw()
            win.flip()

            timer = core.CountdownTimer(1.0)
            while timer.getTime() > 0:
                evs = kb.getEvents()
                if evs:
                    for ev in evs:
                        # During countdown, still ignore key-up; allow ESC to abort.
                        if getattr(ev, "duration", None):
                            if debug:
                                print(f"[trial {trial}] countdown IGNORE key-up {ev.key!r}")
                            continue
                        k = _norm_key(ev.key)
                        if debug:
                            print(f"[trial {trial}] countdown saw key-down {ev.key!r} norm={k!r}")
                        if k in escape_norm:
                            print("*** Escape detected during countdown. Aborting task. ***")
                            _save_data(results, outfile)
                            return results if results else None
                core.wait(0.01)

        # --------------------------------------------------------
        # PROMPT
        # --------------------------------------------------------
        if allow_any:
            press_prompt = "eine Taste"
        else:
            press_prompt = (
                "die Leertaste" if proceed_norm == {'space'} else " / ".join(sorted(proceed_norm))
            )
        prompt_text = (
            "Bitte zeichnen Sie nun die Blume (ZWISCHEN den Linien).\n"
            f"Drücken Sie danach so schnell wie möglich {press_prompt}."
        )
        prompt_stim = visual.TextStim(
            win, text=prompt_text, wrapWidth=1.8, color=text_color, height=0.07
        )
        prompt_stim.draw()
        win.callOnFlip(send_trigger, 1)  # start‑of‑trial trigger
        win.flip()
        start_time = io.getTime()
        print(f"[trial {trial}] prompt shown at iohub_t={start_time:.6f}")

        _drain_all(f"trial{trial}-pre-response")

        print(f"[trial {trial}] waiting for response...")
        resp_key, resp_ev = _wait_for_key(f"trial{trial}")
        if resp_key is None:
            print(f"=== Aborted during trial {trial} response ===")
            _save_data(results, outfile)      
            return results if results else None

        rt = resp_ev.time - start_time
        send_trigger(2)  # end‑of‑trial trigger

        results.append({
            "participant_id": participant_id,
            "trial": trial,
            "rt": rt,
            "key": resp_key,
        })
        print(f"[trial {trial}] key={resp_key!r} rt={rt:.4f}s")

        last_rt = rt

        # ITI
        win.flip()
        core.wait(0.5)

    # ------------------------------------------------------------
    # Save & finish
    # ------------------------------------------------------------
    _save_data(results, outfile)
    print("--- Paper Flower Task Finished ---")
    
    # Teardown instructions
    tablet_stand_teardown_screen(win, io,
                                proceed_keys=proceed_keys,
                                escape_keys=escape_keys,
                                debug=debug)
        
    # After the teardown screen is dismissed, we must show one final
    # screen and pause BEFORE returning. This prevents the parent
    # script from closing the window instantly.
    
    final_stim = visual.TextStim(
        win, text="Aufgabe beendet.\n\nVielen Dank!",
        color=text_color, height=0.07
    )
    final_stim.draw()
    win.flip()
    core.wait(5.0)  

    return results if results else None






# =====================================================================
# Standalone Execution Block for Testing (No changes needed here)
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("--- Running in TEST mode ---")
        participant_id = "test_003"
    else:
        participant_id = sys.argv[1]

    print(f"--- Running Paper Flower Task (Standalone Mode) ---")
    print(f"Participant ID: {participant_id}")

    win = None
    io = None
    try:
        io = launchHubServer()
        win = visual.Window(fullscr=True, color="grey", units="height")
        run_task(
            win=win,
            io=io,
            participant_id=participant_id,
            num_trials = 10,
            text_color='black'
        )
    except Exception as e:
        print(f"An error occurred during the standalone test: {e}")
    finally:
        print("--- Shutting down standalone test ---")
        if win:
            win.close()
        if io:
            io.quit()
        core.quit()