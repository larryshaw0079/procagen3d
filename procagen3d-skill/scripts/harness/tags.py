"""Grep-able ProcAgen3D status tags for the stdlib driver."""

OK = "[PROCAGEN3D:OK]"


def warn(tag, msg):
    print(f"[PROCAGEN3D:WARN:{tag}] {msg}")


def fail(tag, msg):
    print(f"[PROCAGEN3D:FAIL:{tag}] {msg}")
