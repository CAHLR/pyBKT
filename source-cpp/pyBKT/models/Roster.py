from pyBKT.models import Model
from enum import Enum
import numpy as np

class Roster:
    def __init__(self, students, skill, mastery_state = 0.95, track_progress = False, model = None):
        self.model = model if model is not None else Model()
        self.students = {}
        if isinstance(students, int):
            self.add_students(list(range(1, students + 1)))
        elif isinstance(students, list):
            self.add_students(students)
        self.mastery_state = mastery_state
        self.track_progress = track_progress
        self.skill = skill

    # STATE BASED METHODS

    def reset_state(self, student_name):
        self.students[student_name] = State.DEFAULT_STATE

    def reset_states(self):
        for s in self.students:
            self.reset_state(s)

    def get_state(self, student_name):
        return self.students[student_name]

    def get_states(self):
        return self.students
        
    def update_state(self, student_name, correct, **kwargs):
        self.students[student_name].update(correct, kwargs) 

    def update_states(self, corrects, **kwargs):
        for s in corrects:
            self.update_state(s, corrects[s], **kwargs)

    # STUDENT BASED METHODS

    def add_student(self, student_name, initial_state = StateType.DEFAULT_STATE):
        self.students[student_name] = State(initial_state, self)

    def add_students(self, student_names, initial_states = StateType.DEFAULT_STATE):
        if not isinstance(initial_states, list):
            initial_states = [initial_states] * len(student_names)
        for i, s in enumerate(student_names):
            self.add_student(s, initial_states[i])

    def remove_student(self, student_name):
        del self.students[student_name]

    def remove_students(self, student_names):
        for s in student_names:
            self.remove_student(s)

    # MISCELLANEOUS FUNCTIONS

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model

    def get_mastery_state(self):
        return self.mastery_state

    def set_mastery_state(self, mastery_state):
        self.mastery_state = mastery_state
        for s in self.students:
            self.students[student_name].refresh()

class State:
    def __init__(self, initial_state, roster):
        self.state_type = initial_state
        self.roster = roster
        self.current_state = {'correct_prediction': -1, 'state_prediction': -1}
        self.tracked_states = []

    def update(self, correct, kwargs):
        if isinstance(correct, int):
            data = self.process_data([correct], kwargs)
        elif isinstance(correct, list):
            data = self.process_data(correct, kwargs)
        correct_predictions, state_predictions = self.predict(model, skill, data, self.current_state)
        self.current_state['correct_prediction'] = correct_predictions[-1]
        self.current_state['state_predictions'] = state_predictions[-1]
        
        if self.roster.track_progress:
            self.tracked_states.append(dict(self.current_state))
        self.refresh()

    def process_data(self, corrects, kwargs):
        model_type = [kwargs.get(t, default = False) for t in ('multilearn', 'multiprior', 'multipair', 'multigs')]
        pass # process into native model format

    def predict(self, model, data, state):
        model.fit_model[roster.skill]['pi_0'] = np.array([[1 - state['state_prediction']], [state['state_prediction']]])
        model.fit_model[roster.skill]['prior'] = truemodel['pi_0'][1][0]
        correct_predictions, state_predictions = model._predict(model.fit_model[roster.skill], data)
        return correct_predictions, state_predictions[1]

    def refresh(self):
        if self.current_state['state_prediction'] == -1:
            self.state_type = StateType.DEFAULT_STATE
        elif self.current_state['state_prediction'] >= roster.mastery_state:
            self.state_type = StateType.MASTERED
        else:
            self.state_type = StateType.UNMASTERED

class StateType(Enum):
    DEFAULT_STATE = 1
    UNMASTERED = 2
    MASTERED = 3
