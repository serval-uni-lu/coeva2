def get_data_by_month(data, a_month):
    month_df = data[data["issue_d"] == a_month]
    a_y = month_df.pop("charged_off").to_numpy()
    a_X = month_df.to_numpy()
    return a_X, a_y


def filter_initial_states(x, start, size):
    return x[start : start + size]
