import bisect


def find_biggest_element_less_than_x(sorted_list, x):
    # Find index of the rightmost element less than x
    index = bisect.bisect_left(sorted_list, x) - 1

    # Return the biggest element smaller than x
    return index