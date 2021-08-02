from pyBKT.models import Model
from enum import Enum
import numpy as np

class StateType(Enum):
    DEFAULT_STATE = "default"
    UNMASTERED = "unmastered"
    MASTERED = "mastered"

class Roster:
    def __init__(self, students, skills, mastery_state = 0.95, track_progress = False, model = None):
        """
        Initializes a Roster with a set of students and skills. Students can be specified as the number of
        students in total or the names/identifiers of all the students. The mastery state threshold can be
        adjusted, but it defaults to 95% mastery probability or more as representing having attained mastery.

        The model can be provided using set_model, but the constructor accepts a pyBKT Model as well.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))

        """
        self.skill_rosters = {}
        if isinstance(skills, str):
            skills = [skills]
        elif not isinstance(skills, list):
            raise ValueError("skills must be a list or string")
        for s in skills:
            self.skill_rosters[s] = SkillRoster(students, s, mastery_state = mastery_state, 
                                                track_progress = track_progress, model = model)
        if model is not None and not isinstance(model, Model):
            raise ValueError("invalid model, must be of type pyBKT.models.Model")
        self.model = model
        if not isinstance(mastery_state, float) or not (0 <= mastery_state <= 1):
            raise ValueError("invalid mastery state, must be between 0 and 1")
        self.mastery_state = mastery_state
        if not isinstance(track_progress, bool):
            raise ValueError("track progress must be a boolean")
        self.track_progress = track_progress

    # STATE BASED METHODS

    def reset_state(self, skill_name, student_name):
        """
        Resets the state for a particular student for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.reset_state('Calculate unit rate', 'Morgan')

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].reset_state(student_name)

    def reset_states(self, skill_name):
        """
        Resets the state for all students for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.reset_states('Calculate unit rate')

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].reset_states()

    def get_mastery_prob(self, skill_name, student_name):
        """
        Fetches mastery probability for a particular student for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_mastery_prob('Calculate unit rate', 'Morgan')
        0.08836967920678879

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_mastery_prob(student_name)

    def get_mastery_probs(self, skill_name):
        """
        Fetches mastery probability for all students for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_mastery_probs('Calculate unit rate')
        {'Morgan': 0.08836967920678879, 'Bob': 0.0076413143121398285}

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_mastery_probs()

    def get_correct_prob(self, skill_name, student_name):
        """
        Fetches correctness probability for a particular student for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_correct_prob('Calculate unit rate', 'Morgan')
        0.5242746611208322

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_correct_prob(student_name)

    def get_correct_probs(self, skill_name):
        """
        Fetches correctness probability for all students for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_correct_probs('Calculate unit rate')
        {'Morgan': 0.5242746611208322, 'Bob': 0.4942939431522598}

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_correct_probs()

    def get_state(self, skill_name, student_name):
        """
        Fetches state for a particular student for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.get_state('Calculate unit rate', 'Bob')
        State(StateType.UNMASTERED, {'correct_prediction': 0.4942939431522598, 'state_prediction': 0.0076413143121398285}, Roster(...))

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_state(student_name)

    def get_states(self, skill_name):
        """
        Fetches states for all students for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_states('Calculate unit rate')
        {'Morgan': State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...)), 'Bob': State(StateType.UNMASTERED, {'correct_prediction': 0.4942939431522598, 'state_prediction': 0.0076413143121398285}, Roster(...))}

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_states()

    def get_state_type(self, skill_name, student_name):
        """
        Fetches the state type (mastered, unmastered) of a particular student for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_state_type('Calculate unit rate', 'Morgan')
        <StateType.UNMASTERED: 2>

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_state_type(student_name)

    def get_state_types(self, skill_name):
        """
        Fetches the state type (mastered, unmastered) of all students for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))
        >>> roster.get_state_types('Calculate unit rate')
        {'Morgan': <StateType.UNMASTERED: 2>, 'Bob': <StateType.UNMASTERED: 2>}

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].get_state_types()
        
    def update_state(self, skill_name, student_name, correct, **kwargs):
        """
        Updates state of a particular student for a skill given one response.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_state('Calculate unit rate', 'Morgan', correct = 1)
        State(StateType.UNMASTERED, {'correct_prediction': 0.5242746611208322, 'state_prediction': 0.08836967920678879}, Roster(...))

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].update_state(student_name, correct, **kwargs)

    def update_states(self, skill_name, corrects, **kwargs):
        """
        Updates state of all students for a skill given one response each.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.update_states('Calculate unit rate', {'Morgan': 1, 'Bob': 1})
        {'Morgan': State(StateType.UNMASTERED, {'correct_prediction': 0.5275795972712107, 'state_prediction': 0.10003241541105552}, Roster(...)), 'Bob': State(StateType.UNMASTERED, {'correct_prediction': 0.5275795972712107, 'state_prediction': 0.10003241541105552}, Roster(...))}

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        return self.skill_rosters[skill_name].update_states(corrects, **kwargs)

    # STUDENT BASED METHODS

    def add_student(self, skill_name, student_name, initial_state = StateType.DEFAULT_STATE):
        """
        Adds student with given name for a skill with an optional initial state.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.add_student('Calculate unit rate', 'Anthony')
        >>> roster.skill_rosters['Calculate unit rate'].students.keys()
        dict_keys(['Morgan', 'Bob', 'Anthony'])

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].add_student(student_name, initial_state)

    def add_students(self, skill_name, student_names, initial_states = StateType.DEFAULT_STATE):
        """
        Adds students with given names for a skill with optional initial states.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.add_students('Calculate unit rate', ['Anthony', 'Jessie'])
        >>> roster.skill_rosters['Calculate unit rate'].students.keys()
        dict_keys(['Morgan', 'Bob', 'Anthony', 'Jessie'])

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].add_students(student_names, initial_states)

    def remove_student(self, skill_name, student_name):
        """
        Removes student with given name for a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.remove_student('Calculate unit rate', 'Morgan')
        >>> roster.skill_rosters['Calculate unit rate'].students.keys()
        dict_keys(['Bob'])

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].remove_student(student_name)

    def remove_students(self, skill_name, student_names):
        """
        Removes students with given names for a skill.

        >>> roster = Roster(['Morgan', 'Bob', 'Jess'], 'Calculate unit rate')
        >>> roster.remove_students('Calculate unit rate', ['Morgan', 'Jess'])
        >>> roster.skill_rosters['Calculate unit rate'].students.keys()
        dict_keys(['Bob'])

        """
        if skill_name not in self.skill_rosters:
            raise ValueError("skill not found in roster")
        self.skill_rosters[skill_name].remove_students(student_names)

    # MISCELLANEOUS FUNCTIONS

    def get_model(self):
        """
        Gets BKT model that the roster uses.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)
        >>> roster.get_model()
        Model(parallel=True, num_fits=5, seed=76759360, defaults=None)

        """
        return self.model

    def set_model(self, model):
        """
        Sets BKT model that the roster uses.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)

        """
        self.model = model
        if not isinstance(model, Model):
            raise ValueError("invalid model, must be of type pyBKT.models.Model")
        for s in self.skill_rosters:
            self.skill_rosters[s].set_model(model)

    def get_mastery_state(self):
        """
        Gets the mastery probability threshold that the roster uses to denote mastery of a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.get_mastery_state()
        0.95

        """
        return self.mastery_state

    def set_mastery_state(self, mastery_state):
        """
        Sets the mastery probability threshold that the roster uses to denote mastery of a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.set_mastery_state(0.8)
        >>> roster.get_mastery_state()
        0.8

        """
        self.mastery_state = mastery_state
        for s in self.skill_rosters:
            self.skill_rosters[s].set_mastery_state(mastery_state)

    # NATIVE PYTHON FUNCTIONS
    def __repr__(self):
        """
        Returns a Python representation of the Roster object.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)

        """
        return 'Roster(%s, %s, %s, %s, %s)' % (repr(len(self.students)), repr(self.skills), 
                                               repr(self.mastery_state), repr(self.track_progress), 
                                               repr(self.model))

class SkillRoster:
    def __init__(self, students, skill, mastery_state = 0.95, track_progress = False, model = None):
        self.model = model
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
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        self.students[student_name] = State(StateType.DEFAULT_STATE, roster = self)

    def reset_states(self):
        for s in self.students:
            self.reset_state(s)

    def get_mastery_prob(self, student_name):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].get_mastery_prob()

    def get_mastery_probs(self):
        return {s: self.students[s].get_mastery_prob() for s in self.students}

    def get_correct_prob(self, student_name):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].get_correct_prob()

    def get_correct_probs(self):
        return {s: self.students[s].get_correct_prob() for s in self.students}

    def get_state(self, student_name):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name]

    def get_states(self):
        return self.students

    def get_state_type(self, student_name):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].state_type

    def get_state_types(self):
        return {s: self.get_state_type(s) for s in self.students}
        
    def update_state(self, student_name, correct, **kwargs):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        self.students[student_name].update(correct, kwargs) 
        return self.get_state(student_name)

    def update_states(self, corrects, **kwargs):
        if any([i not in self.students for i in corrects]):
            raise ValueError("student name not found in roster for this skill")
        for s in corrects:
            self.update_state(s, corrects[s], **kwargs)
        return self.get_states()

    # STUDENT BASED METHODS

    def add_student(self, student_name, initial_state = StateType.DEFAULT_STATE):
        self.students[student_name] = State(initial_state, roster = self)

    def add_students(self, student_names, initial_states = StateType.DEFAULT_STATE):
        if not isinstance(student_names, list):
            raise ValueError("student names must be a list")
        if not isinstance(initial_states, list):
            initial_states = [initial_states] * len(student_names)
        for i, s in enumerate(student_names):
            self.add_student(s, initial_states[i])

    def remove_student(self, student_name):
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
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
            self.students[s].refresh()

    # NATIVE PYTHON FUNCTIONS
    def __repr__(self):
        return 'SkillRoster(%s, %s, %s, %s, %s)' % (repr(len(self.students)), repr(self.skill), 
                                                    repr(self.mastery_state), repr(self.track_progress), 
                                                    repr(self.model))

class State:
    def __init__(self, state_type, state = None, roster = None):
        self.state_type = state_type
        self.roster = roster
        if state is not None:
            self.current_state = state
        elif self.roster.model is not None and self.roster.model.fit_model and self.roster.skill in self.roster.model.fit_model:
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
        if self.roster.model is None:
            raise ValueError("model not specified")
        multilearn, multigs = [kwargs.get(t, False) for t in ('multilearn', 'multigs')]
        gs_ref = self.roster.model.fit_model[self.roster.skill]['gs_names']
        resource_ref = self.roster.model.fit_model[self.roster.skill]['resource_names']
        if append:
            corrects = np.append(corrects, [-1])
        if set(corrects) - set([-1, 0, 1]) != set():
            raise ValueError("data must be binary")
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
        return 'State(%s, %s, %s)' % (stype, repr(self.current_state), 'Roster(...)')
