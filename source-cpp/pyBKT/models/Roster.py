from collections.abc import Hashable, Iterable, Mapping, Sequence
from enum import Enum
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar

import numpy as np

from pyBKT.models.Model import Model
from pyBKT.util.structures import FloatArray, IntArray, SequenceData

if TYPE_CHECKING:
    from typing_extensions import Unpack

StudentT = TypeVar("StudentT", bound=Hashable)


class UpdateOptions(TypedDict, total=False):
    """The learn class and guess/slip class of new answers, for multilearn and multigs models."""

    multilearn: Hashable | list[Any]
    multigs: Hashable | list[Any]


class StatePredictions(TypedDict):
    """A student's latest predictions; -1 means there is none yet."""

    correct_prediction: float
    state_prediction: float


class StateType(Enum):
    DEFAULT_STATE = "default"
    UNMASTERED = "unmastered"
    MASTERED = "mastered"

class Roster:
    def __init__(self, students: int | list[StudentT], skills: str | list[str], mastery_state: float = 0.95,
                 track_progress: bool = False, model: Model | None = None) -> None:
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
        if model is not None and not isinstance(model, Model):
            raise TypeError("invalid model, must be of type pyBKT.models.Model")
        if not isinstance(mastery_state, float) or not (0 <= mastery_state <= 1):
            raise ValueError("invalid mastery state, must be between 0 and 1")
        if not isinstance(track_progress, bool):
            raise TypeError("track progress must be a boolean")
        self.skill_rosters: dict[str, SkillRoster] = {}
        self.students: int | Sequence[Hashable] = students
        if isinstance(skills, str):
            skills = [skills]
        elif not isinstance(skills, list):
            raise TypeError("skills must be a list or string")
        for s in skills:
            self.skill_rosters[s] = SkillRoster(students, s, mastery_state = mastery_state, 
                                                track_progress = track_progress, model = model)
        self.model = model
        self.mastery_state = mastery_state
        self.track_progress = track_progress

    # STATE BASED METHODS

    def reset_state(self, skill_name: str, student_name: Hashable) -> None:
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

    def reset_states(self, skill_name: str) -> None:
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

    def get_mastery_prob(self, skill_name: str, student_name: Hashable) -> float:
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

    def get_mastery_probs(self, skill_name: str) -> dict[Hashable, float]:
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

    def get_correct_prob(self, skill_name: str, student_name: Hashable) -> float:
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

    def get_correct_probs(self, skill_name: str) -> dict[Hashable, float]:
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

    def get_state(self, skill_name: str, student_name: Hashable) -> "State":
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

    def get_states(self, skill_name: str) -> dict[Hashable, "State"]:
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

    def get_state_type(self, skill_name: str, student_name: Hashable) -> StateType:
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

    def get_state_types(self, skill_name: str) -> dict[Hashable, StateType]:
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
        
    def update_state(self, skill_name: str, student_name: Hashable, correct: int | IntArray,
                     **kwargs: "Unpack[UpdateOptions]") -> "State":
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

    def update_states(self, skill_name: str, corrects: Mapping[StudentT, int | IntArray],
                      **kwargs: "Unpack[UpdateOptions]") -> dict[Hashable, "State"]:
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

    def add_student(self, skill_name: str, student_name: Hashable,
                    initial_state: StateType = StateType.DEFAULT_STATE) -> None:
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

    def add_students(self, skill_name: str, student_names: list[StudentT],
                     initial_states: StateType | list[StateType] = StateType.DEFAULT_STATE) -> None:
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

    def remove_student(self, skill_name: str, student_name: Hashable) -> None:
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

    def remove_students(self, skill_name: str, student_names: Iterable[Hashable]) -> None:
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

    def get_model(self) -> Model | None:
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

    def set_model(self, model: Model) -> None:
        """
        Sets BKT model that the roster uses.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)

        """
        self.model = model
        if not isinstance(model, Model):
            raise TypeError("invalid model, must be of type pyBKT.models.Model")
        for s in self.skill_rosters:
            self.skill_rosters[s].set_model(model)

    def get_mastery_state(self) -> float:
        """
        Gets the mastery probability threshold that the roster uses to denote mastery of a skill.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> roster.get_mastery_state()
        0.95

        """
        return self.mastery_state

    def set_mastery_state(self, mastery_state: float) -> None:
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
    def __repr__(self) -> str:
        """
        Returns a Python representation of the Roster object.

        >>> roster = Roster(['Morgan', 'Bob'], 'Calculate unit rate')
        >>> model = Model()
        >>> model.fit(data_path = 'ct.csv', skills = 'Calculate unit rate')
        >>> roster.set_model(model)

        """
        count = self.students if isinstance(self.students, int) else len(self.students)
        return f'Roster({count!r}, {self.skill_rosters.keys()!r}, {self.mastery_state!r}, {self.track_progress!r}, {self.model!r})'

class SkillRoster:
    def __init__(self, students: int | list[StudentT], skill: str, mastery_state: float = 0.95,
                 track_progress: bool = False, model: Model | None = None) -> None:
        self.model = model
        self.students: dict[Hashable, State] = {}
        self.mastery_state = mastery_state
        self.track_progress = track_progress
        self.skill = skill
        if isinstance(students, int):
            self.add_students(list(range(1, students + 1)))
        elif isinstance(students, list):
            self.add_students(students)

    # STATE BASED METHODS

    def reset_state(self, student_name: Hashable) -> None:
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        self.students[student_name] = State(StateType.DEFAULT_STATE, roster = self)

    def reset_states(self) -> None:
        for s in self.students:
            self.reset_state(s)

    def get_mastery_prob(self, student_name: Hashable) -> float:
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].get_mastery_prob()

    def get_mastery_probs(self) -> dict[Hashable, float]:
        return {s: self.students[s].get_mastery_prob() for s in self.students}

    def get_correct_prob(self, student_name: Hashable) -> float:
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].get_correct_prob()

    def get_correct_probs(self) -> dict[Hashable, float]:
        return {s: self.students[s].get_correct_prob() for s in self.students}

    def get_state(self, student_name: Hashable) -> "State":
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name]

    def get_states(self) -> dict[Hashable, "State"]:
        return self.students

    def get_state_type(self, student_name: Hashable) -> StateType:
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        return self.students[student_name].state_type

    def get_state_types(self) -> dict[Hashable, StateType]:
        return {s: self.get_state_type(s) for s in self.students}
        
    def update_state(self, student_name: Hashable, correct: int | IntArray,
                     **kwargs: "Unpack[UpdateOptions]") -> "State":
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        self.students[student_name].update(correct, kwargs) 
        return self.get_state(student_name)

    def update_states(self, corrects: Mapping[StudentT, int | IntArray],
                      **kwargs: "Unpack[UpdateOptions]") -> dict[Hashable, "State"]:
        if any(i not in self.students for i in corrects):
            raise ValueError("student name not found in roster for this skill")
        for s in corrects:
            self.update_state(s, corrects[s], **kwargs)
        return self.get_states()

    # STUDENT BASED METHODS

    def add_student(self, student_name: Hashable, initial_state: StateType = StateType.DEFAULT_STATE) -> None:
        self.students[student_name] = State(initial_state, roster = self)

    def add_students(self, student_names: list[StudentT],
                     initial_states: StateType | list[StateType] = StateType.DEFAULT_STATE) -> None:
        if not isinstance(student_names, list):
            raise TypeError("student names must be a list")
        if not isinstance(initial_states, list):
            initial_states = [initial_states] * len(student_names)
        for i, s in enumerate(student_names):
            self.add_student(s, initial_states[i])

    def remove_student(self, student_name: Hashable) -> None:
        if student_name not in self.students:
            raise ValueError("student name not found in roster for this skill")
        del self.students[student_name]

    def remove_students(self, student_names: Iterable[Hashable]) -> None:
        for s in student_names:
            self.remove_student(s)

    # MISCELLANEOUS FUNCTIONS

    def get_model(self) -> Model | None:
        return self.model

    def set_model(self, model: Model) -> None:
        self.model = model
        for s in self.students:
            self.students[s].update(-1, {'multigs': self.model.model_type[-1], 'multilearn': self.model.model_type[0]}, append = False)

    def get_mastery_state(self) -> float:
        return self.mastery_state

    def set_mastery_state(self, mastery_state: float) -> None:
        self.mastery_state = mastery_state
        for s in self.students:
            self.students[s].refresh()

    # NATIVE PYTHON FUNCTIONS
    def __repr__(self) -> str:
        return f'SkillRoster({len(self.students)!r}, {self.skill!r}, {self.mastery_state!r}, {self.track_progress!r}, {self.model!r})'

class State:
    def __init__(self, state_type: StateType, state: StatePredictions | None = None,
                 roster: "SkillRoster | None" = None) -> None:
        self.state_type = state_type
        self.roster = roster
        self.tracked_states: list[StatePredictions] = []
        self.current_state: StatePredictions
        if state is not None:
            self.current_state = state
            return
        model = self._roster().model
        if model is not None and model.fit_model and self._roster().skill in model.fit_model:
            self.current_state = {'correct_prediction': -1, 'state_prediction': model._fitted_model(self._roster().skill)['prior']}
            self.update(-1, {}, append = False)
        else:
            self.current_state = {'correct_prediction': -1, 'state_prediction': -1}

    def _roster(self) -> "SkillRoster":
        assert self.roster is not None, "a State needs its SkillRoster to make predictions"
        return self.roster

    def get_mastery_prob(self) -> float:
        return self.current_state['state_prediction']

    def get_correct_prob(self) -> float:
        return self.current_state['correct_prediction']

    def update(self, correct: int | IntArray, kwargs: UpdateOptions, append: bool = True) -> None:
        if isinstance(correct, int):
            data = self.process_data(np.array([correct]), kwargs, append = append)
        elif isinstance(correct, np.ndarray):
            data = self.process_data(correct, kwargs, append = append)
        else:
            raise TypeError("need to pass int or np.ndarray")
        roster = self._roster()
        # process_data has already raised ValueError if the roster has no model.
        assert roster.model is not None
        correct_predictions, state_predictions = self.predict(roster.model, roster.skill, data, self.current_state)
        self.current_state['correct_prediction'] = correct_predictions[-1]
        self.current_state['state_prediction'] = state_predictions[-1]

        if roster.track_progress:
            self.tracked_states.append(self.current_state.copy())
        self.refresh()

    def process_data(self, corrects: IntArray, kwargs: UpdateOptions, append: bool = True) -> SequenceData:
        roster = self._roster()
        if roster.model is None:
            raise ValueError("model not specified")
        multilearn, multigs = [kwargs.get(t, False) for t in ('multilearn', 'multigs')]
        fitted = roster.model._fitted_model(roster.skill)
        gs_ref = fitted['gs_names']
        resource_ref = fitted['resource_names']
        if append:
            corrects = np.append(corrects, [-1])
        if set(corrects) - {-1, 0, 1} != set():
            raise ValueError("data must be binary")
        data = corrects + 1
        lengths = np.array([len(corrects)], dtype=np.int64)
        starts = np.array([1], dtype=np.int64)
        
        if multilearn:
            if isinstance(kwargs['multilearn'], list):
                resargs = kwargs['multilearn']
            else:
                resargs = [kwargs['multilearn']]
            resources = np.array([resource_ref[x] for x in resargs])
            if any(k not in resource_ref for k in resargs):
                raise ValueError("specified learn class must be one of:", resource_ref.keys())
        else:
            resources = np.ones(len(data), dtype=np.int64)

        if multigs:
            if isinstance(kwargs['multigs'], list):
                gsargs = kwargs['multigs']
            else:
                gsargs = [kwargs['multigs']]
            if any(k not in gs_ref for k in gsargs):
                raise ValueError("specified guess/slip class must be one of:", gs_ref.keys())
            data_ref = np.array([gs_ref[x] for x in gsargs])
            data_temp = np.zeros((len(gs_ref), len(corrects)))
            for i in range(len(data_temp[0]) - 1):
                data_temp[data_ref[i]][i] = data[i]
            data = np.asarray(data_temp, dtype='int')
        else:
            data = np.asarray([data], dtype='int')

        Data: SequenceData = {'starts': starts, 'lengths': lengths, 'resources': resources, 'data': data}
        return Data

    def predict(self, model: Model, skill: str, data: SequenceData,
                state: StatePredictions) -> tuple[FloatArray, FloatArray]:
        fitted = model._fitted_model(self._roster().skill)
        if state['state_prediction'] > 0:
            old_prior = fitted['pi_0']
            fitted['pi_0'] = np.array([[1 - state['state_prediction']], [state['state_prediction']]])
            fitted['prior'] = fitted['pi_0'][1][0]
        correct_predictions, state_predictions = model._predict(model._fitted_model(skill), data)
        fitted['pi_0'] = old_prior if state['state_prediction'] > 0 else fitted['pi_0']
        fitted['prior'] = fitted['pi_0'][1][0]
        return correct_predictions, state_predictions[1]

    def refresh(self) -> None:
        if self.current_state['state_prediction'] == -1:
            self.state_type = StateType.DEFAULT_STATE
        elif self.current_state['state_prediction'] >= self._roster().mastery_state:
            self.state_type = StateType.MASTERED
        else:
            self.state_type = StateType.UNMASTERED

    def __repr__(self) -> str:
        stype = repr(self.state_type)
        stype = stype[stype.index('<') + 1: stype.index(':')]
        return 'State({}, {}, {})'.format(stype, repr(self.current_state), 'Roster(...)')
