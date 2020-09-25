import sexpdata
from sexpdata import Symbol


class DFA(object):
    def __init__(self, name, alphabet, states, start, accepts,
                 transition_list):
        self._name = name
        self._alphabet = alphabet
        self._states = states
        self._start = start
        self._accepts = accepts
        self._current_state = start
        self.__make_transition_structure(transition_list)
        self.__make_violations_structure()

    def __make_transition_structure(self, transition_list):
        self._transitions = {}
        for state in self._states:
            self._transitions[state] = {}
        for transition in transition_list:
            start_state = transition[0]
            next_state = transition[1]
            transition_symbols = transition[2:]
            for transition_symbol in transition_symbols:
                self._transitions[start_state][transition_symbol] = next_state

    def __make_violations_structure(self):
        self._violators = {}
        for state in self._states:
            self._violators[state] = []
            for symbol in self._alphabet:
                if symbol in self._transitions[state] and self._transitions[
                        state][symbol] in self._accepts:
                    self._violators[state].append(symbol)

    @staticmethod
    def from_string(dfa_string):
        """Constructs an instance of DFA from a string"""
        data = sexpdata.loads(dfa_string)
        if data[0] != sexpdata.Symbol('defdfa'):
            raise ValueError("dfa_string %s is improperly formatted." %
                             (dfa_string))
        name = data[1].value()
        clean_symbols = lambda l: [
            d._val if isinstance(d, Symbol) else d for d in l
        ]
        alphabet = clean_symbols(data[2][0])
        state_list = clean_symbols(data[2][1])
        start_state = data[2][2]
        accept_states = clean_symbols(data[2][3])
        transition_list = [clean_symbols(d) for d in data[3]]
        return DFA(name, alphabet, state_list, start_state, accept_states,
                   transition_list)

    def step(self, token, hypothetical=False):
        """Steps the internal state with an input token"""
        # Handle the case where the env passes the token in an iterable
        if hypothetical:
            save_state = self._current_state
        if hasattr(token, '__iter__'):
            if len(token) > 1:
                raise Exception()
            token = token[0]
        if token in self._transitions[self._current_state]:
            self._current_state = self._transitions[self._current_state][token]
        else:
            self._current_state = self._start
        is_accept = self.is_accepting
        if hypothetical:
            self._current_state = save_state
        return is_accept

    def reset(self):
        """Reset the internal state to the start state"""
        self._current_state = self._start

    @property
    def current_state(self) -> int:
        """Returns the internal state"""
        return self._current_state

    @property
    def states(self):  # -> set(int):
        """Returns the set of state ids"""
        return self._states

    @property
    def accepting_states(self):
        return self._accepts

    @property
    def alphabet(self):  # -> set(int):
        """Returns the set of valid input tokens"""
        return self._alphabet

    @property
    def violating_inputs(self):  # -> set(int)
        """Returns the set of input tokens which would move the state to an accepting one"""
        return self._violators[self._current_state]

    @property
    def is_accepting(self):
        """Returns whether or not the DFA is currently in an accept state"""
        return self._current_state in self._accepts
