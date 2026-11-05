'''
trigger module
This script is a single function that sends a parallel port trigger to a brain products liveAMP 
serial port. It overcomes the limitation of only sending up to 16 unique numbers because 
"why would anyone need more?" Besides, a serial port should be plenty...

I need more, dear brain products, and hence this script...

simply import and type send_trigger(numerical value you want to send). That's it.

@Author: Max Archibald Montgomery
Support: Max.montgomery@uni-hamburg.de
The development of this script is funded by the department of heealth science at Hamburg University.
Written for internal use at Labor Funke. Please take and adapt freely!

'''


from psychopy import parallel, core, logging
import time

def send_trigger(trigger_value,
                 port_address='0x3FB8',
                 duration_sec=0.05,
                 digit_duration=0.100,
                 *,
                 use_separators=True,
                 inter_digit_idle=0.010,
                 repeats=1,
                 repeat_gap=0.050,
                 busy_hog_last=0.003):
    """
    Send robust trigger bursts to a Brain Products (LiveAmp) parallel/serial adapter.

    SPECIAL CASE:
        Values 1..16 (inclusive) are ALWAYS sent as a single code (no digit-splitting)
        to accommodate hardware/recording setups that reliably pass only 1..16.

    GENERAL CASE:
        - 0..9 -> single pulse (code is that value).
        - Any other integer -> up to 3 digits are sent sequentially, with optional 0 separators.

    Args:
        trigger_value (int): The code to send. 1..16 sends as one pulse; else up to 3 digits.
        port_address (str|int): Parallel port address (e.g., '0x3FB8' or 0x3FB8).
        duration_sec (float): Duration for single-code pulses (incl. the 1..16 special case).
        digit_duration (float): Duration per digit in a multi-digit burst.
        use_separators (bool): If True, insert 0 between digits for inter_digit_idle seconds.
        inter_digit_idle (float): Idle-at-0 between digits (and tail) when separators are used.
        repeats (int): Repeat the entire burst N times (>=1).
        repeat_gap (float): Idle-at-0 gap between repeated bursts.
        busy_hog_last (float): Busy-spin tail on waits to sharpen timing (0.002–0.005 typical).

    Returns:
        bool: True if sequence(s) were sent without exception; False otherwise.
    """
    # ---- helpers -------------------------------------------------------------
    def _to_addr(addr):
        if isinstance(addr, str):
            try:
                return int(addr, 16) if addr.lower().startswith('0x') else int(addr)
            except Exception:
                pass
        return addr

    def _sleep_exact(seconds: float, hog_last: float):
        """Sleep ~seconds with a short final busy-spin for precision."""
        if seconds <= 0:
            return
        start = time.perf_counter()
        bulk = max(0.0, seconds - hog_last)
        if bulk > 0:
            core.wait(bulk, hogCPUperiod=min(hog_last, bulk))
        deadline = start + seconds
        while time.perf_counter() < deadline:
            pass

    # ---- interpret requested code -------------------------------------------
    try:
        val = int(trigger_value)
    except Exception:
        logging.error(f"send_trigger: invalid trigger_value {trigger_value!r}")
        return False

    # NEW: hard guarantee that 1..16 are single unsplit pulses
    if 1 <= val <= 16:
        digits = [val]
        pulse_dur = duration_sec
    elif 0 <= val <= 9:
        digits = [val]  # classic single-digit pulse
        pulse_dur = duration_sec
    else:
        # Multi-digit path (up to 3 digits). Zeros are kept explicitly.
        s = str(abs(val))[:3]
        digits = [int(ch) for ch in s]
        pulse_dur = digit_duration

    # ---- expand into (value, duration) sequence -----------------------------
    sequence = []
    durations = []
    if len(digits) == 1:
        sequence.append(digits[0]); durations.append(pulse_dur)
        # Always finish at 0 with a small explicit idle for edge clarity.
        sequence.append(0); durations.append(inter_digit_idle if use_separators else 0.002)
    else:
        for d in digits:
            sequence.append(d); durations.append(pulse_dur)
            if use_separators:
                sequence.append(0); durations.append(inter_digit_idle)
        if not use_separators or sequence[-1] != 0:
            sequence.append(0); durations.append(inter_digit_idle if use_separators else 0.002)

    # ---- init port -----------------------------------------------------------
    try:
        p_port = parallel.ParallelPort(address=_to_addr(port_address))
    except Exception as e:
        logging.error(f"Parallel Port init failed at {port_address}: {e}")
        print(f"CRITICAL: Could not open parallel port {port_address}. Error: {e}")
        return False

    # ---- send ---------------------------------------------------------------
    ok = True
    try:
        n_repeats = max(1, int(repeats))
        for r in range(n_repeats):
            for code, hold in zip(sequence, durations):
                try:
                    p_port.setData(int(code))
                except Exception as e:
                    logging.error(f"setData({code}) failed: {e}")
                    ok = False
                    break
                _sleep_exact(hold, busy_hog_last)
            if not ok:
                break
            if r < n_repeats - 1:
                try:
                    p_port.setData(0)
                except Exception:
                    pass
                _sleep_exact(max(0.0, float(repeat_gap)), busy_hog_last)
    finally:
        # Always leave line low.
        try:
            p_port.setData(0)
        except Exception:
            pass

    return ok

