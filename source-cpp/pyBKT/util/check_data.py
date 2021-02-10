def check_data(data):
    if data["data"].shape[1] != data["resources"].shape[0]: raise IndexError("data and resource sizes must match")
    if not all(data["starts"] + data["lengths"] - 1 <= data["data"].shape[1]): raise IndexError("data lengths don't match its shape")
