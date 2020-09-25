import functools
import itertools

import numpy as np
#from skimage.feature import match_template
#import skimage
import math

#from common.atari_wrappers import LazyFrames
from constraint.constraint import Constraint, SoftDenseConstraint

mapping = {}


def register(name):
    def _thunk(func):
        mapping[name] = func
        return func

    return _thunk


def seaquest_sub_depth(frame, last_pos=None):
    # first number vertical, lower is upper
    # second number horizontal, lower is lefter
    leftface_sub_template = np.array(
        [[38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38],
         [38, 38, 38, 38, 73, 77, 38, 38, 38, 38, 38, 38],
         [38, 38, 38, 43, 210, 210, 144, 38, 38, 38, 38, 38],
         [38, 38, 81, 112, 209, 205, 173, 110, 88, 56, 95, 38],
         [38, 60, 210, 210, 208, 205, 205, 205, 209, 209, 142, 38],
         [38, 51, 149, 214, 210, 210, 210, 210, 210, 211, 142, 38],
         [38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38]])
    rightface_sub_template = np.array([
        [38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38, 38],
        [38, 38, 38, 38, 38, 39, 72, 142, 46, 38, 38, 38],
        [38, 38, 38, 38, 38, 117, 206, 210, 51, 38, 38, 38],
        [38, 81, 113, 81, 143, 179, 205, 209, 147, 143, 56, 38],
        [38, 133, 190, 206, 205, 205, 205, 208, 211, 210, 69, 38],
        [38, 133, 189, 177, 177, 177, 177, 177, 147, 83, 38, 38],
        [38, 64, 48, 38, 38, 38, 38, 38, 38, 38, 38, 38],
    ])
    left_conv = match_template(frame, leftface_sub_template, pad_input=True)
    right_conv = match_template(frame, rightface_sub_template, pad_input=True)
    left_pos = np.array(np.argwhere(left_conv == np.max(left_conv))[0])
    right_pos = np.array(np.argwhere(right_conv == np.max(right_conv))[0])
    if np.linalg.norm(left_pos - right_pos) < 4:
        return (left_pos + right_pos) / 2
    elif last_pos is not None and np.linalg.norm(left_pos - last_pos) < 3:
        return left_pos
    elif last_pos is not None and np.linalg.norm(right_pos - last_pos) < 3:
        return right_pos
    else:
        return None


@register('oxygen_Seaquest')
def oxygen_seaquest(is_hard, is_dense, reward_shaping):
    pass


@register('diver_Seaquest')
def diver_Seaquest(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_diver.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    last_pos = None
    last_known_pos = None
    move_dict = {
        0: 'X',
        1: 'S',
        2: 'U',
        3: 'R',
        4: 'L',
        5: 'D',
        6: 'UR',
        7: 'UL',
        8: 'DR',
        9: 'DL',
        10: 'US',
        11: 'RS',
        12: 'LS',
        13: 'DS',
        14: 'URS',
        15: 'ULS',
        16: 'DRS',
        17: 'DLS'
    }
    inv_move_dict = {
        'X': [0],
        'S': [1, 10, 11, 12, 13, 14, 15, 16, 17],
        'U': [2, 6, 7, 10, 14, 15],
        'R': [3, 6, 8, 11, 14, 16],
        'D': [5, 8, 9, 13, 16, 17],
        'L': [4, 7, 9, 12, 15, 17]
    }

    def translation_fn(obs, act, done):
        threshold = 142.0  # The base value in the section of screen where the divers are.
        #4x4 kernels that are "center-of-mass" on each diver.
        #For each frame in the LazyFrame stack (4 total), count the number of divers.
        if not isinstance(obs, LazyFrames):
            obs = obs[0]
        last_frame = np.array(obs)[:, :, -1]
        nonlocal last_pos, last_known_pos
        diver_1 = last_frame[72:74, 32:34]
        diver_2 = last_frame[72:74, 36:38]
        diver_3 = last_frame[72:74, 40:42]
        diver_4 = last_frame[72:74, 44:46]
        diver_5 = last_frame[72:74, 48:50]
        diver_6 = last_frame[72:74, 52:54]
        divers = [diver_1, diver_2, diver_3, diver_4, diver_5, diver_6]
        num_divers = 0
        for diver in divers:
            if np.mean(diver) < threshold:
                num_divers += 1
        # print("Number of divers: {}".format(num_divers))
        action_letters = move_dict[act]
        last_pos = seaquest_sub_depth(last_frame, last_pos)
        if last_pos is not None:
            last_known_pos = last_pos
        return_val = 0
        if last_pos is None:
            temp_last_pos = last_known_pos
        else:
            temp_last_pos = last_pos

        if temp_last_pos is None or temp_last_pos[0] < 30:
            return_val += 0
        elif temp_last_pos[0] > 30 and temp_last_pos[0] < 45:
            return_val += 6
        else:
            return_val += 12

        if num_divers == 0:
            return_val += 0
        elif num_divers < 0 and num_divers < 6:
            return_val += 2
        else:
            return_val += 4

        if 'U' not in action_letters:
            return_val += 0
        else:
            return_val += 1

        if done:
            last_pos = None
            last_known_pos = None

        return return_val

    def inv_translation_fn(token):
        # don't go up
        if token % 2 == 1:
            return inv_move_dict['U']
        else:
            return list(set(range(18)) - set(inv_move_dict['U']))

    if is_dense:
        return SoftDenseConstraint('diver_dense_SpaceInvaders',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn,
                                   gamma=.99)
    return Constraint('diver_SpaceInvaders',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register('dangerzone_SpaceInvaders')
def dangerzone_spaceinvaders(is_hard, is_dense, reward_shaping):
    with open(
            "./baselines/constraint/constraints/spaceinvaders_dangerzone.lisp"
    ) as dfa_file:
        dfa_string = dfa_file.read()

    bullet_detection_baseline = None
    player_detection_baseline = None
    player_template = np.array([[0, 0, 6, 79, 28, 0, 0],
                                [0, 0, 30, 98, 67, 0, 0],
                                [0, 6, 55, 98, 77, 21, 0],
                                [0, 12, 91, 98, 98, 41, 0]])
    bullet_template = np.array([[0, 40, 0], [0, 99, 0], [0, 99, 0], [0, 40,
                                                                     0]])
    translation_dict = dict([(0, 'a'), (1, 'a'), (2, 'r'), (3, 'l'), (4, 'r'),
                             (5, 'l')])
    inv_translation_dict = {'a': [0, 1], 'r': [2, 4], 'l': [3, 5]}

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

    def translation_fn(obs, action, done):
        if not isinstance(obs, LazyFrames):
            obs = obs[0]
        frames = np.array(obs)
        nonlocal bullet_detection_baseline
        nonlocal player_detection_baseline
        old_bullet_player_area = frames[45:-5, :, -2]
        bullet_player_area = frames[45:-5, :, -1]

        old_bullet_conv = match_template(old_bullet_player_area,
                                         bullet_template,
                                         pad_input=True)
        bullet_conv = match_template(bullet_player_area,
                                     bullet_template,
                                     pad_input=True)
        player_conv = match_template(bullet_player_area,
                                     player_template,
                                     pad_input=True)
        if bullet_detection_baseline is None:
            bullet_detection_baseline = np.max(bullet_conv) + 0.02
            #print("bullet threshold", bullet_detection_baseline)
        if player_detection_baseline is None:
            player_detection_baseline = np.minimum(
                np.max(player_conv) + 0.02, 0.95)
            #print("player threshold", player_detection_baseline)

        old_bullet_locs = np.argwhere(
            old_bullet_conv > bullet_detection_baseline)
        bullet_locs = np.argwhere(bullet_conv > bullet_detection_baseline)
        player_locs = np.argwhere(player_conv > player_detection_baseline)

        if done:
            bullet_detection_baseline = None
            player_detection_baseline = None

        if len(player_locs) != 1 or len(bullet_locs) == 0:
            # player is missing or we don't where they are
            # OR no bullets detected
            # then the constraint is inactive
            return 0

        # merge individual points which belong together
        def merge_bullet_areas(bullet_locs):
            bullet_areas = [(a, a, b) for a, b in bullet_locs]
            i = 0
            while i < len(bullet_areas) - 1:
                j = i + 1
                while j < len(bullet_areas):
                    a, b, c = bullet_areas[i]
                    d, e, f = bullet_areas[j]
                    if c == f and (b == d - 1):
                        bullet_areas[i] = (a, e, c)
                        del bullet_areas[j]
                    else:
                        j += 1
                i += 1
            return bullet_areas

        bullet_areas = merge_bullet_areas(bullet_locs)
        old_bullet_areas = merge_bullet_areas(old_bullet_locs)

        # filter out bullets travelling upwards
        down_bullets = []
        for bullet in bullet_areas:
            a, b, c = bullet
            for d, e, f in old_bullet_areas:
                if c == f and d < a and e < b:
                    down_bullets.append([b, c])
                    break
        if len(down_bullets) == 0:
            return 0

        player_loc = player_locs[0]
        player_x_line = (player_loc[1] - 3, player_loc[1] + 3)
        # sort down bullets into lra
        left = []
        right = []
        above = []
        for bullet in down_bullets:
            if bullet[1] < player_x_line[0]:
                left.append(bullet)
            elif bullet[1] > player_x_line[1]:
                right.append(bullet)
            else:
                above.append(bullet)

        distance = lambda p, b: (b[0] - p[0])**2 + (b[1] - p[1])**2
        token_str = translation_dict[action]

        if len(left) == 0:
            token_str += 'l3'
        else:
            min_lbullet_distance = min([distance(player_loc, x) for x in left])
            if min_lbullet_distance < 12:
                token_str += 'l0'
            elif min_lbullet_distance < 24:
                token_str += 'l1'
            else:
                token_str += 'l2'
        if len(right) == 0:
            token_str += 'r3'
        else:
            min_rbullet_distance = min(
                [distance(player_loc, x) for x in right])
            if min_rbullet_distance < 12:
                token_str += 'r0'
            elif min_rbullet_distance < 24:
                token_str += 'r1'
            else:
                token_str += 'r2'
        if len(above) == 0:
            token_str += 'a3'
        else:
            min_abullet_distance = min(
                [distance(player_loc, x) for x in above])
            if min_abullet_distance < 12:
                token_str += 'a0'
            elif min_abullet_distance < 24:
                token_str += 'a1'
            else:
                token_str += 'a2'
            return token2int(token_str)

    def inv_translation_fn(token):
        i = token // 64
        if i == 2:
            c = 'r'
        elif i == 1:
            c = 'l'
        elif i == 0:
            c = 'a'
        return inv_translation_dict[c]

    if is_dense:
        return SoftDenseConstraint('dangerzone_dense_SpaceInvaders',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn,
                                   gamma=.99)
    return Constraint('dangerzone_SpaceInvaders',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register('paddle_ball_distance_Breakout')
def paddle_direction_breakout(is_hard, is_dense, reward_shaping):
    with open(
            "./baselines/constraint/constraints/int_counter_with_null_reset.lisp"
    ) as dfa_file:
        dfa_string = dfa_file.read()

    limit = 10
    trigger_pulled = False

    def translation_fn(obs, action, done):
        # action 0 is noop
        # action 1 is the fire button
        # action 2 goes to the right
        # action 3 goes to the left
        if not isinstance(obs, LazyFrames):
            obs = obs[0]
        frames = np.array(obs)
        paddle_line = frames[-7, 5:-5, -1]
        ball_box = frames[38:-9, 5:-5, -1]
        ball_pixels = np.nonzero(ball_box)
        paddle_pixels = np.nonzero(paddle_line)
        nonlocal trigger_pulled
        if action == 1:
            trigger_pulled = True
        if done:
            trigger_pulled = False
        if not trigger_pulled:
            return 'N'
        try:
            # we get dim 1 of ball pixels and 0 of paddle pixels, which are both horizontal axis
            ball_x_center = (np.min(ball_pixels[1]) +
                             np.max(ball_pixels[1])) / 2.
            paddle_x_center = (np.min(paddle_pixels[0]) +
                               np.max(paddle_pixels[0])) / 2.
            # case where paddle is too far to the right
            if ball_x_center - paddle_x_center < -limit and (action != 3):
                return 1
            # too far to the left
            elif ball_x_center - paddle_x_center > limit and (action != 2):
                return -1
            else:
                return 0
        except ValueError:
            return 'N'

    def inv_translation_fn(token):
        if token == 1:
            return [0, 1, 2]
        elif token == -1:
            return [0, 1, 3]
        else:
            print(token)
            exit()

    if is_dense:
        return SoftDenseConstraint('paddle_direction_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn,
                                   gamma=0.99)
    return Constraint('paddle_direction_Breakout',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register('1d_dithering2_Breakout')
def one_d_dithering_breakout(is_hard, is_dense, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint('1d_dithering2_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   lambda obs, action, done: action,
                                   gamma=0.99)
    return Constraint('1d_dithering2_Breakout',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register('1d_dithering2_SpaceInvaders')
def one_d_dithering_spaceinvaders(is_hard, is_dense, reward_shaping, k=2):
    with open("./baselines/constraint/constraints/1d_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    translation_dict = dict([(0, 1), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    inv_translation_dict = {1: [0, 1], 2: [2, 4], 3: [3, 5]}
    translation_fn = lambda obs, action, done: translation_dict[action]
    inv_translation_fn = lambda token: inv_translation_dict[token]
    if is_dense:
        return SoftDenseConstraint('1d_dithering2_dense_Breakout',
                                   dfa_string,
                                   reward_shaping,
                                   translation_dict,
                                   gamma=.99)
    return Constraint('1d_dithering2_SpaceInvaders',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn,
                      inv_translation_fn=inv_translation_fn)


def build_one_d_actuation(num_actions, k):
    dfa_string_template = '(defdfa {name} (({input_symbols}) ({states}) {start_state} ({accepting_states})) ({transitions}))'
    transition_template = '({initial_state} {target_state} {symbol})'

    name = '1d_{k}_actuation'.format(k=k)
    input_symbols = ' '.join(list(map(str, range(num_actions))))
    states = ' '.join(list(map(str, range(num_actions * k +
                                          1))))  # add one for the start state
    start_state = 0
    accepting_states = ' '.join(
        [str(a * k) for a in range(1, num_actions + 1)])

    transitions = []
    for a in range(num_actions):
        transitions.append(
            transition_template.format(initial_state=0,
                                       target_state=a * k + 1,
                                       symbol=a))
        for r in range(k - 1):
            transitions.append(
                transition_template.format(initial_state=a * k + r + 1,
                                           target_state=a * k + r + 2,
                                           symbol=a))
    transitions = ' '.join(transitions)

    dfa_string = dfa_string_template.format(name=name,
                                            input_symbols=input_symbols,
                                            states=states,
                                            start_state=start_state,
                                            accepting_states=accepting_states,
                                            transitions=transitions)
    return dfa_string


@register('1d_actuation4_Breakout')
def oned_actuation_breakout4(is_hard, is_dense, reward_shaping):
    if is_dense:
        return SoftDenseConstraint(
            '1d_actuation_dense_breakout4',
            build_one_d_actuation(4, k=4),
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('1d_actuation_breakout4',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register('1d_actuation4_SpaceInvaders')
def oned_actuation_spaceinvaders4(is_hard, is_dense, reward_shaping):
    translation_dict = dict([(0, 0), (1, 1), (2, 2), (3, 3), (4, 2), (5, 3)])
    inv_translation_dict = {0: [0], 1: [1], 2: [2, 4], 3: [3, 5]}
    translation_fn = lambda obs, action, done: translation_dict[action]
    inv_translation_fn = lambda token: inv_translation_dict[token]
    if is_dense:
        return SoftDenseConstraint('1d_actuation_dense_SpaceInvaders',
                                   build_one_d_actuation(4, k=4),
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('1d_actuation_SpaceInvaders',
                      build_one_d_actuation(4, k=4),
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=inv_translation_fn)


@register("2d_actuation4_Seaquest")
def twod_actuation4_seaquest(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_actuation.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint(
            '2d_actuation4_dense_Seaquest',
            dfa_string,
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('2d_actuation4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register("2d_dithering4_Seaquest")
def twod_dithering4_seaquest(is_hard, is_dense, reward_shaping):
    with open("./baselines/constraint/constraints/seaquest_dithering.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()
    if is_dense:
        return SoftDenseConstraint(
            '2d_dithering4_dense_Seaquest',
            dfa_string,
            reward_shaping,
            translation_fn=lambda obs, action, done: action,
            gamma=0.99)
    return Constraint('2d_dithering4_Seaquest',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=lambda obs, action, done: action,
                      inv_translation_fn=lambda token: [token])


@register("proximity_pointgoal1")
def proximity_point_goal_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        token = max(obs['hazards_lidar'] - 0.6) // 0.04
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_PointGoal1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_PointGoal1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_cargoal1")
def proximity_car_goal_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        token = max(obs['hazards_lidar'] - 0.6) // 0.04
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_CarGoal1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_CarGoal1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_doggogoal1")
def proximity_doggo_goal_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        token = max(obs['hazards_lidar'] - 0.6) // 0.04
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_DoggoGoal1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_DoggoGoal1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_pointgoal2")
def proximity_point_goal_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        vases_token = max(obs['vases_lidar'] - 0.6) // 0.04
        token = max(hazards_token, vases_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_PointGoal2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_PointGoal2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_cargoal2")
def proximity_car_goal_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        vases_token = max(obs['vases_lidar'] - 0.6) // 0.04
        token = max(hazards_token, vases_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_CarGoal2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_CarGoal2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_doggogoal2")
def proximity_doggo_goal_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        vases_token = max(obs['vases_lidar'] - 0.6) // 0.04
        token = max(hazards_token, vases_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_DoggoGoal2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_DoggoGoal2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_pointbutton1")
def proximity_point_button_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_PointButton1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_PointButton1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_carbutton1")
def proximity_car_button_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_CarButton1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_CarButton1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_doggobutton1")
def proximity_doggo_button_one(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_DoggoButton1',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_DoggoButton1',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_pointbutton2")
def proximity_point_button_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_PointButton2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_PointButton2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_carbutton2")
def proximity_car_button_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token, 0)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_CarButton2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_CarButton2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


@register("proximity_doggobutton2")
def proximity_doggo_button_two(is_hard, is_dense, reward_shaping):
    with open("../constraint/constraints/proximity_highres.lisp"
              ) as dfa_file:
        dfa_string = dfa_file.read()

    def translation_fn(obs, action, done):
        hazards_token = max(obs['hazards_lidar'] - 0.6) // 0.04
        gremlins_token = max(obs['gremlins_lidar'] - 0.6) // 0.04
        wrong_button_token = max(obs['buttons_lidar'] - obs['goal_lidar'] -
                                 0.6) // 0.04
        token = max(hazards_token, gremlins_token, wrong_button_token)
        return token

    if is_dense:
        return SoftDenseConstraint('proximity_DoggoButton2',
                                   dfa_string,
                                   reward_shaping,
                                   translation_fn=translation_fn,
                                   gamma=0.99)
    return Constraint('proximity_DoggoButton2',
                      dfa_string,
                      is_hard,
                      reward_shaping,
                      translation_fn=translation_fn,
                      inv_translation_fn=lambda token: [token])


def get_constraint(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        print("Available constraints", mapping.keys())
        raise ValueError("Unknown constraint type:", name)
