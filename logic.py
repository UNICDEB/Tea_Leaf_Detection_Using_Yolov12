def build_xyz_points(flat_list):

    # Convert flat list [x,y,z,x,y,z,...] into [(x,y,z), ...]

    if len(flat_list) % 3 != 0:
        raise ValueError("Input list length must be multiple of 3")

    points = []
    for i in range(0, len(flat_list), 3):
        x = flat_list[i]
        y = flat_list[i + 1]
        z = flat_list[i + 2]
        points.append((x, y, z))

    return points


def filter_and_store(points):

    X_LIMIT = 100
    Y_LIMIT = 120
    Z_LIMIT = 300

    result = []

    for x, y, z in points:

        # Rule 1: Skip if x or y exceeds limit
        if x > X_LIMIT or y > Y_LIMIT:
            continue

        # Rule 2: Clamp z value
        if z > Z_LIMIT:
            z = Z_LIMIT

        # Store valid xyz
        result.extend([x, y, z])

    # Prepend count
    count = len(result) // 3
    result.insert(0, count)

    return result



input_points = [
    50, 80, 200,
    90, 110, 350,
    150, 50, 100,
    60, 130, 200,
    40, 60, 290
]


xyz_points = build_xyz_points(input_points)
output = filter_and_store(xyz_points)

print("Final Output:", output)
