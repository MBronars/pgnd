"""File identity against a pwm release index (gpwm data/indexed_episodes.same_mtime / same_file, branch mb/psc-port)."""


def same_mtime(actual_ns, recorded_ns):
    """mtime identity against an index record, to the whole second. PSC Bridges-2 Ocean/Jet (Lustre) store only
    whole-second mtimes, so a byte-identical copy of the release loses the nanoseconds the index recorded."""
    return actual_ns // 1_000_000_000 == recorded_ns // 1_000_000_000


def same_file(stat, size, mtime_ns):
    return stat.st_size == size and same_mtime(stat.st_mtime_ns, mtime_ns)
