import sys

def print_dot(num, tot, newline_every=None):
    if newline_every == None: newline_every = 25
    sys.stdout.write('.')

    if(num+1 % newline_every == 0 or num == tot-1):
        sys.stdout.write(' [%d/%d]\n' % (num+1, tot))
    sys.stdout.flush()