from pyBKT.models import Model
from enum import Enum
import numpy as np

class StateType(Enum):
    DEFAULT_STATE = 1
    UNMASTERED = 2
    MASTERED = 3

class Roster:
    def __init__(self, students, skills, mastery_state = 0.95, track_progress = False, model = None):
        self.skill_rosters = {}
        if isinstance(skills, str):
            skills = [skills]
        for s in skills:
            self.skill_rosters[s] = SkillRoster(students, s, mastery_state = mastery_state, 
                                                track_progress = track_progress, model = model)
        self.model = model
        self.mastery_state = mastery_state
        self.track_progress = track_progress

    # STATE BASED METHODS

    def reset_state(self, skill_name, student_name):
        self.skill_rosters[skill_name].reset_state(student_name)

    def reset_states(self, skill_name):
        self.skill_rosters[skill_name].reset_states()

    def get_mastery_prob(self, skill_name, student_name):
        return self.skill_rosters[skill_name].get_mastery_prob(student_name)

    def get_mastery_probs(self, skill_name):
        return self.skill_rosters[skill_name].get_mastery_probs()

    def get_correct_prob(self, skill_name, student_name):
        return self.skill_rosters[skill_name].get_correct_prob(student_name)

    def get_correct_probs(self, skill_name):
        return self.skill_rosters[skill_name].get_correct_probs()

    def get_state(self, skill_name, student_name):
        return self.skill_rosters[skill_name].get_state(student_name)

    def get_states(self, skill_name):
        return self.skill_rosters[skill_name].get_states()

    def get_state_type(self, skill_name, student_name):
        return self.skill_rosters[skill_name].get_state_type(student_name)

    def get_state_types(self, skill_name):
        return self.skill_rosters[skill_name].get_state_types()
        
    def update_state(self, skill_name, student_name, correct, **kwargs):
        return self.skill_rosters[skill_name].update_state(student_name, correct, **kwargs)

    def update_states(self, skill_name, corrects, **kwargs):
        return self.skill_rosters[skill_name].update_states(correct, **kwargs)

    # STUDENT BASED METHODS

    def add_student(self, skill_name, student_name, initial_state = StateType.DEFAULT_STATE):
        self.skill_rosters[skill_name].add_student(student_name, initial_state)

    def add_students(self, skill_name, student_names, initial_states = StateType.DEFAULT_STATE):
        self.skill_rosters[skill_name].add_students(student_names, initial_states)

    def remove_student(self, skill_name, student_name):
        self.skill_rosters[skill_name].remove_student(student_name)

    def remove_students(self, skill_name, student_names):
        self.skill_rosters[skill_name].remove_students(student_names)

    # MISCELLANEOUS FUNCTIONS

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model
        for s in self.skill_rosters:
            self.skill_rosters[s].set_model(model)

    def get_mastery_state(self):
        return self.mastery_state

    def set_mastery_state(self, mastery_state):
        self.mastery_state = mastery_state
        for s in self.skill_rosters:
            self.skill_rosters[s].set_mastery_state(model)


class SkillRoster:
    def __init__(self, students, skill, mastery_state = 0.95, track_progress = False, model = None):
        self.model = model if model is not None else Model()
        self.students = {}
        self.mastery_state = mastery_state
        self.track_progress = track_progress
        self.skill = skill
        if isinstance(students, int):
            self.add_students(list(range(1, students + 1)))
        elif isinstance(students, list):
            self.add_students(students)

    # STATE BASED METHODS

    def reset_state(self, student_name):
        self.students[student_name] = State(initial_state, roster = self)

    def reset_states(self):
        for s in self.students:
            self.reset_state(s)

    def get_mastery_prob(self, student_name):
        return self.students[student_name].get_mastery_prob()

    def get_mastery_probs(self):
        return {s: self.students[s].get_mastery_prob() for s in self.students}

    def get_correct_prob(self, student_name):
        return self.students[student_name].get_correct_prob()

    def get_correct_probs(self):
        return {s: self.students[s].get_correct_prob() for s in self.students}

    def get_state(self, student_name):
        return self.students[student_name]

    def get_states(self):
        return self.students

    def get_state_type(self, student_name):
        return self.students[student_name].state_type

    def get_state_types(self):
        return {s: self.get_state(s) for s in self.students}
        
    def update_state(self, student_name, correct, **kwargs):
        self.students[student_name].update(correct, kwargs) 
        return self.get_state(student_name)

    def update_states(self, corrects, **kwargs):
        for s in corrects:
            self.update_state(s, corrects[s], **kwargs)
        return self.get_states()

    # STUDENT BASED METHODS

    def add_student(self, student_name, initial_state = StateType.DEFAULT_STATE):
        self.students[student_name] = State(initial_state, roster = self)

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
        for s in self.students:
            self.students[s].update(-1, {'multigs': self.model.model_type[-1], 'multilearn': self.model.model_type[0]}, append = False)

    def get_mastery_state(self):
        return self.mastery_state

    def set_mastery_state(self, mastery_state):
        self.mastery_state = mastery_state
        for s in self.students:
            self.students[student_name].refresh()

    # NATIVE PYTHON FUNCTIONS
    def __repr__(self):
        return 'Roster(%s, %s, %s, %s, %s)' % (repr(len(self.students)), repr(self.skill), 
                                               repr(self.mastery_state), repr(self.track_progress), 
                                               repr(self.model))

class State:
    def __init__(self, state_type, state = None, roster = None):
        self.state_type = state_type
        self.roster = roster
        if state is not None:
            self.current_state = state
        elif self.roster.model.fit_model and self.roster.skill in self.roster.model.fit_model:
            self.current_state = {'correct_prediction': -1, 'state_prediction': self.roster.model.fit_model[self.roster.skill]['prior']}
            self.update(-1, {'multigs': self.roster.model.model_type[-1], 'multilearn': self.roster.model.model_type[0]}, append = False)
        else:
            self.current_state = {'correct_prediction': -1, 'state_prediction': -1}
        self.tracked_states = []

    def get_mastery_prob(self):
        return self.current_state['state_prediction']

    def get_correct_prob(self):
        return self.current_state['correct_prediction']

    def update(self, correct, kwargs, append = True):
        if isinstance(correct, int):
            data = self.process_data(np.array([correct]), kwargs, append = append)
        elif isinstance(correct, np.ndarray):
            data = self.process_data(correct, kwargs, append = append)
        else:
            raise ValueError("need to pass int or np.ndarray")
        correct_predictions, state_predictions = self.predict(self.roster.model, self.roster.skill, data, self.current_state)
        self.current_state['correct_prediction'] = correct_predictions[-1]
        self.current_state['state_prediction'] = state_predictions[-1]
        
        if self.roster.track_progress:
            self.tracked_states.append(dict(self.current_state))
        self.refresh()

    def process_data(self, corrects, kwargs, append = True):
        multilearn, multigs = [kwargs.get(t, False) for t in ('multilearn', 'multigs')]
        gs_ref = self.roster.model.fit_model[self.roster.skill]['gs_names']
        resource_ref = self.roster.model.fit_model[self.roster.skill]['resource_names']
        if append:
            corrects = np.append(corrects, [-1])
        data = corrects + 1
        lengths = np.array([len(corrects)], dtype=np.int64)
        starts = np.array([1], dtype=np.int64)
        
        if multilearn:
            if isinstance(kwargs['multilearn'], list):
                resargs = kwargs['multilearn']
            else:
                resargs = [kwargs['multilearn']]
            resources = np.array(list(map(lambda x: resource_ref[x], resargs)))
        else:
            resources = np.ones(len(data), dtype=np.int64)

        if multigs:
            if isinstance(kwargs['multigs'], list):
                gsargs = kwargs['multigs']
            else:
                gsargs = [kwargs['multigs']]
            data_ref = np.array(list(map(lambda x: gs_ref[x], gsargs)))
            data_temp = np.zeros((len(gs_ref), len(corrects)))
            for i in range(len(data_temp[0]) - 1):
                data_temp[data_ref[i]][i] = data[i]
            data = np.asarray(data_temp, dtype='int32')
        else:
            data = np.asarray([data], dtype='int32')

        Data = {'starts': starts, 'lengths': lengths, 'resources': resources, 'data': data}
        return Data

    def predict(self, model, skill, data, state):
        if state['state_prediction'] > 0:
            old_prior = model.fit_model[self.roster.skill]['pi_0']
            model.fit_model[self.roster.skill]['pi_0'] = np.array([[1 - state['state_prediction']], [state['state_prediction']]])
            model.fit_model[self.roster.skill]['prior'] = model.fit_model[self.roster.skill]['pi_0'][1][0]
        correct_predictions, state_predictions = model._predict(model.fit_model[skill], data)
        model.fit_model[self.roster.skill]['pi_0'] = old_prior if state['state_prediction'] > 0 else model.fit_model[self.roster.skill]['pi_0']
        model.fit_model[self.roster.skill]['prior'] = model.fit_model[self.roster.skill]['pi_0'][1][0]
        return correct_predictions, state_predictions[1]

    def refresh(self):
        if self.current_state['state_prediction'] == -1:
            self.state_type = StateType.DEFAULT_STATE
        elif self.current_state['state_prediction'] >= self.roster.mastery_state:
            self.state_type = StateType.MASTERED
        else:
            self.state_type = StateType.UNMASTERED

    def __repr__(self):
        stype = repr(self.state_type)
        stype = stype[stype.index('<') + 1: stype.index(':')]
        return 'Roster(%s, %s, %s)' % (stype, repr(self.current_state), 'Roster(...)')
