def filter_and_store(points):

    X_LIMIT = 100
    Y_LIMIT = 120
    Z_LIMIT = 300

    result = []

    for x, y, z in points:

        # Rule 1: If x or y exceeds limit -> skip
        if x > X_LIMIT or y > Y_LIMIT:
            continue

        # Rule 2: Clamp z if needed
        if z > Z_LIMIT:
            z = Z_LIMIT

        # Store valid point
        result.extend([x, y, z])

    # Prepend count (number of valid xyz sets)
    count = len(result) // 3
    result.insert(0, count)

    return result


# input_points = [
#     (50, 80, 200),
#     (90, 110, 350),
#     (150, 50, 100),
#     (60, 130, 200),
#     (40, 60, 290) 
# ]


input_points = [
    50, 80, 200,
    90, 110, 350,
    150, 50, 100,
    60, 130, 200,
    40, 60, 290
]

output = filter_and_store(input_points)
print(output)
