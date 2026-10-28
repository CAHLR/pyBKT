import sys


def print_dot(num: int, tot: int, newline_every: int | None = None) -> None:
    if newline_every == None: newline_every = 25
    sys.stdout.write('.')

    if(num+1 % newline_every == 0 or num == tot-1):
        sys.stdout.write(f" [{int(num + 1)}/{int(tot)}]\n")
    sys.stdout.flush()