def find_max_precip_location_new(cube, start_idx, stop_idx, x_offset=0, y_offset=0, mask_2d=None):
    
    # Slice the event window first — much smaller than full year
    #cube_sliced = cube[start_idx:stop_idx,:,:]
    data = np.array(cube.data)
    data[data >= 1e19] = np.nan

    # Then apply mask only to this small slice
    if mask_2d is not None:
        data = np.where(mask_2d, data, np.nan)

    flat_index = np.nanargmax(data)
    # print(f"Max precip: {np.nanmax(data)}")
    t_local, y_idx, x_idx = np.unravel_index(flat_index, data.shape)

    return {
        'max_precip':   float(data[t_local, y_idx, x_idx]),
        't_global':     int(t_local + start_idx),
        't_local':      int(t_local),
        'x_idx':        int(x_idx),
        'y_idx':        int(y_idx),
        'x_idx_global': int(x_idx + x_offset),
        'y_idx_global': int(y_idx + y_offset),
        'x_coord':      cube.coord('projection_x_coordinate').points[x_idx],
        'y_coord':      cube.coord('projection_y_coordinate').points[y_idx],}