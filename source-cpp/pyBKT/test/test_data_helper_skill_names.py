import pandas as pd

from pyBKT.util.data_helper import convert_data


SPECIAL_SKILL = "Order of Operations +,-,/,* () positive reals"
ORDINARY_SKILL = "Basic Addition"


def make_dataframe():
    return pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "user_id": [101, 101, 102, 102],
        "skill_name": [
            SPECIAL_SKILL,
            ORDINARY_SKILL,
            SPECIAL_SKILL,
            ORDINARY_SKILL,
        ],
        "correct": [1, 0, 0, 1],
    })


def test_skill_names_with_regex_characters_are_matched_literally():
    result = convert_data(
        make_dataframe(),
        [SPECIAL_SKILL, ORDINARY_SKILL],
    )
    assert SPECIAL_SKILL in result
    assert ORDINARY_SKILL in result


def test_string_skill_pattern_keeps_regex_behavior():
    result = convert_data(make_dataframe(), ".*")
    assert SPECIAL_SKILL in result
    assert ORDINARY_SKILL in result
