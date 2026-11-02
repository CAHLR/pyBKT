import pandas as pd

from pyBKT.util.data_helper import convert_data


def test_skill_names_with_regex_characters():
    skill = "Order of Operations +,-,/,* () positive reals"

    df = pd.DataFrame({
        "order_id": [1, 2],
        "skill_name": [skill, "Other skill"],
        "correct": [1, 0],
        "user_id": [1, 1],
    })

    result = convert_data(df, [skill])

    assert list(result.keys()) == [skill]


def test_string_skill_name_remains_regex():
    df = pd.DataFrame({
        "order_id": [1, 2],
        "skill_name": ["Order 123", "Order ABC"],
        "correct": [1, 0],
        "user_id": [1, 1],
    })

    result = convert_data(df, "Order.*")

    assert list(result.keys()) == ["Order 123", "Order ABC"]
