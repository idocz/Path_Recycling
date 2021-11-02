
@cuda.jit(device=True)
def travel_to_voxels_border_old(current_point, current_voxel, direction, voxel_size, next_voxel):
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    t_x = 2.1
    t_y = 2.2
    t_z = 2.3
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + (inc_x - 1) / 2) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + (inc_y - 1) / 2) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]) - current_point[2]) / direction[2]

    assign_3d(next_voxel, current_voxel)
    if t_x <= t_y and t_x <= t_z:
        # collision with x
        next_voxel[0] += inc_x
        current_point[0] = (current_voxel[0] + 1 + (inc_x - 1) / 2) * voxel_size[0]
        current_point[1] = current_point[1] + t_x * direction[1]
        current_point[2] = current_point[2] + t_x * direction[2]
        return t_x
        # t_min = t_x
            # t_min = t_x
        # else:
        #     # collision with z
        #     next_voxel[2] += inc_z
        #     current_point[0] = current_point[0] + t_z * direction[0]
        #     current_point[1] = current_point[1] + t_z * direction[1]
        #     current_point[2] = (current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]
        #     # t_min = t_z
    elif t_y <= t_z:
        # collision with y
        next_voxel[1] += inc_y
        current_point[0] = current_point[0] + t_y * direction[0]
        current_point[1] = (current_voxel[1] + 1 + (inc_y - 1) / 2) * voxel_size[1]
        current_point[2] = current_point[2] + t_y * direction[2]
        return t_y
        # t_min = t_y
    else:
        # collision with z
        next_voxel[2] += inc_z
        current_point[0] = current_point[0] + t_z * direction[0]
        current_point[1] = current_point[1] + t_z * direction[1]
        current_point[2] = (current_voxel[2] + 1 + (inc_z - 1) / 2) * voxel_size[2]
        return t_z
        # t_min = t_z
    # current_point[0] = current_point[0] + t_min * direction[0]
    # current_point[1] = current_point[1] + t_min * direction[1]
    # current_point[2] = current_point[2] + t_min * direction[2]
    # return t_min

    # if t_min < 0:
    #     print("bugg in t_min", t_min)


    # return t_min


@cuda.jit(device=True)
def travel_to_voxels_border_old(current_point, current_voxel, direction, voxel_size, next_voxel):
    inc_x = sign(direction[0])
    inc_y = sign(direction[1])
    inc_z = sign(direction[2])
    voxel_fix_x = (inc_x - 1) / 2
    voxel_fix_y = (inc_y - 1) / 2
    voxel_fix_z = (inc_z - 1) / 2
    t_x = 2.1
    t_y = 2.2
    t_z = 2.3
    if direction[0] != 0:
        t_x = (((current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]) - current_point[0]) / direction[0]
    if direction[1] != 0:
        t_y = (((current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]) - current_point[1]) / direction[1]
    if direction[2] != 0:
        t_z = (((current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]) - current_point[2]) / direction[2]

    t_min = min_3d(t_x, t_y, t_z)
    assign_3d(next_voxel, current_voxel)
    if t_min == t_x:
        next_voxel[0] += inc_x
        current_point[0] = (current_voxel[0] + 1 + voxel_fix_x) * voxel_size[0]
        current_point[1] = current_point[1] + t_min * direction[1]
        current_point[2] = current_point[2] + t_min * direction[2]
    elif t_min == t_y:
        next_voxel[1] += inc_y
        current_point[1] = (current_voxel[1] + 1 + voxel_fix_y) * voxel_size[1]
        current_point[0] = current_point[0] + t_min * direction[0]
        current_point[2] = current_point[2] + t_min * direction[2]


    elif t_min == t_z:
        next_voxel[2] += inc_z
        current_point[2] = (current_voxel[2] + 1 + voxel_fix_z) * voxel_size[2]
        current_point[0] = current_point[0] + t_min * direction[0]
        current_point[1] = current_point[1] + t_min * direction[1]

    if t_min < 0:
        print("bugg in t_min", t_min)


    return t_min

