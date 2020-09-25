import numpy as np
import itertools
from constraint.constraint import Constraint, CountingPotentialConstraint
from constraint.constraints.reacher_dynamic_constraint import reacher_dynamic_constraint


def quadrant(cos, sin):
    if cos >= 0:
        if sin >= 0:
            return 0
        else:
            return 3
    else:
        if sin >= 0:
            return 1
        else:
            return 2


def reacher_state_quadrants(obs):
    angles = obs[:4]
    q1 = quadrant(angles[0], angles[2])
    q2 = quadrant(angles[1], angles[3])
    val = q1 * 4 + q2
    return val


def discretize_act(act, coeff=5):
    return np.abs(int(coeff * act[0]))

def discretize_act(act):
    return np.linalg


def action_threshold(act):
    return np.linalg.norm(act) > 1.


def idx_sign(act, idx):
    s = np.sign(act[idx])
    if s > 0:
        return '2'
    else:
        return '3'


def compute_actuation_constraint(threshold, window=3):
    candidates_list = []
    for i in range(window + 1):
        candidates = itertools.product(range(16), repeat=i)
        candidates = filter(lambda x: sum(x) >= threshold, candidates)
        candidates = map(lambda x: ''.join([str(hex(a)[2:]) for a in x]),
                         candidates)
        candidates = '|'.join(
            list(map(lambda s: '({})'.format(s), candidates)))
        candidates_list.append(candidates)
    return '|'.join(candidates_list)

def compute_dynamic_actuation_constraint():
    fragments_list = []
    return '|'.join(fragments_list)

# REGEX
## ATARI
ACTUATION1D_REGEX_k = lambda k: '2{k}|3{k}'.format(k=k)
DITHERING1D_REGEX_k = lambda k: '(23){k}|(32){k}'.format(k=k, k2=2 * k)
## MUJOCO
REACHER_ACTUATION = compute_actuation_constraint(4)
REACHER_DYNAMIC_ACTUATION = compute_dynamic_actuation_constraint()
REACHER_REVISIT = '|'.join([
    '({0}{1}{0})'.format(a, b) for a, b in itertools.permutations(
        map(lambda x: hex(x)[2:], range(10)), 2)
])
HALF_CHEETAH_ACTUATION = compute_actuation_constraint(5)
HALF_CHEETAH_DITHERING_k = lambda k: '(23){k}|(32){k}'.format(k=k)

# CONSTRAINT DICTS
ATARI_CONSTRAINT_DICT = {
                 '1d_dithering': lambda r: Constraint('1d_dithering', DITHERING1D_REGEX_k(2), r, s_active=False),
                 '1d_actuation': lambda r: Constraint('1d_actuation', ACTUATION1D_REGEX_k(4), r, s_active=False),
                 # '2d_dithering': lambda r: Constraint('2d_dithering', DITHERING2D_REGEX_4, r, s_active=False),
}

MUJOCO_CONSTRAINT_DICT = {
                 # reacher
                 'reacher_revisit': lambda r: Constraint('reacher_revisit', REACHER_REVISIT, r, s_tl=reacher_state_quadrants, a_active=False),
                 'reacher_revisit_counting': lambda r: CountingPotentialConstraint('reacher_revisit_counting', REACHER_REVISIT, r, 0.99, s_tl=reacher_state_quadrants, a_active=False),
                 'reacher_actuation_counting': lambda r: CountingPotentialConstraint('reacher_actuation_counting', REACHER_ACTUATION, r, 0.99, a_tl=lambda x: discretize_act(x, 5), s_active=False),
                 'reacher_dynamic': reacher_dynamic_constraint,
                 # half-cheetah
                 'half_cheetah_dithering_0': lambda r: CountingPotentialConstraint('half_cheetah_dithering_0', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 0), s_active=False),
                 'half_cheetah_dithering_1': lambda r: CountingPotentialConstraint('half_cheetah_dithering_1', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 1), s_active=False),
                 'half_cheetah_dithering_2': lambda r: CountingPotentialConstraint('half_cheetah_dithering_2', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 2), s_active=False),
                 'half_cheetah_dithering_3': lambda r: CountingPotentialConstraint('half_cheetah_dithering_3', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 3), s_active=False),
                 'half_cheetah_dithering_4': lambda r: CountingPotentialConstraint('half_cheetah_dithering_4', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 4), s_active=False),
                 'half_cheetah_dithering_5': lambda r: CountingPotentialConstraint('half_cheetah_dithering_5', HALF_CHEETAH_DITHERING_k(3), r, 0.99, s_tl=reacher_state_quadrants, a_tl=lambda a: idx_sign(a, 5), s_active=False),
                 'half_cheetah_overactuation_0': lambda r: CountingPotentialConstraint('half_cheetah_acuation_0', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 'half_cheetah_overactuation_1': lambda r: CountingPotentialConstraint('half_cheetah_acuation_1', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 'half_cheetah_overactuation_2': lambda r: CountingPotentialConstraint('half_cheetah_acuation_2', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 'half_cheetah_overactuation_3': lambda r: CountingPotentialConstraint('half_cheetah_acuation_3', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 'half_cheetah_overactuation_4': lambda r: CountingPotentialConstraint('half_cheetah_acuation_4', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 'half_cheetah_overactuation_5': lambda r: CountingPotentialConstraint('half_cheetah_acuation_5', HALF_CHEETAH_ACTUATION, r, 0.99, a_tl=lambda a: discretize_act(a, 5), s_active=False),
                 }

CONSTRAINT_DICT = {**ATARI_CONSTRAINT_DICT, **MUJOCO_CONSTRAINT_DICT}
