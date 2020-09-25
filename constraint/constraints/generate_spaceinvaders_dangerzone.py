from itertools import product


def token2int(t):
    act = t[0]
    if act == 'r':
        act = 2
    elif act == 'l':
        act = 1
    elif act == 'a':
        act = 0
    l = t[2]
    r = t[4]
    a = t[6]
    return 64 * act * 16 * int(r) + 4 * int(l) + int(a)


def token2state(t):
    act = t[0]
    l = t[2]
    r = t[4]
    a = t[6]
    if (act == 'a' and a == '0') or (act == 'a'
                                     and l == 'l') or (act == 'r'
                                                       and r == '0'):
        return -1
    return 16 * int(r) + 4 * int(l) + int(a)


a = [''.join(s) for s in product(['l'], ['0', '1', '2', '3'])]
l = [''.join(s) for s in product(['r'], ['0', '1', '2', '3'])]
r = [''.join(s) for s in product(['a'], ['0', '1', '2', '3'])]
raw_state_portion = [''.join(s) for s in product(a, l, r)]
raw_tokens = [''.join(s) for s in product(['l', 'r', 'a'], raw_state_portion)]
raw_num_tokens = [str(token2int(s)) for s in raw_tokens]
tokens = '(' + ' '.join(raw_num_tokens) + ')'

states = list(set([token2state(s) for s in raw_tokens]))
states.sort()
states = '(' + ' '.join([str(s) for s in states]) + ')'
start_state = str(token2state('al3r3a3'))
accept_states = "(-1)"
transitions = [
    ' '.join([
        '({} {} {})'.format(s, token2int(t), token2state(t))
        for s in range(-1, 64)
    ]) for t in raw_tokens
]
transitions = '\n'.join(transitions)

prelude = "defdfa dangerzone"
dfa_str = '(' + prelude + '\n(' + tokens + ' ' + states + ' ' + start_state + ' ' + accept_states + ')\n(' + transitions + ')\n)'
with open('spaceinvaders_dangerzone.lisp', 'w') as dfa_file:
    dfa_file.write(dfa_str)
