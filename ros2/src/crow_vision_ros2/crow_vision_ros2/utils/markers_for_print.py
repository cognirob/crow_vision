import cv2
import png
import numpy as np

def print_markers(random_seed, marker_ids, markers_dict_num=54, markers_size=4, marker_size_mm=60, dpi=300):
    if markers_dict_num <= max(marker_ids):
        print('Error: The amount of markers in the dict is smaller than some of the requested ids!')
        return
    # paper and aruco size
    dpmm = dpi / 25.4
    paper_mm = np.array([210, 297])
    marker_size_px = np.round(marker_size_mm * dpmm).astype(np.int64)
    # setup marker dict
    aruco_dict = cv2.aruco.Dictionary_create(markers_dict_num, markers_size, random_seed)
    # generate marker images
    markers = [aruco_dict.drawMarker(marker_id, marker_size_px) for marker_id in marker_ids]
    paper_px = np.round(paper_mm * dpmm).astype(np.int)
    image = np.ones(paper_px, dtype=np.uint8) * 255

    locations = []
    offset = [50, 30]
    for i in range(offset[0], image.shape[0] - (marker_size_px+offset[0]), marker_size_px+offset[0]):
        for j in range(offset[1], image.shape[1] - (marker_size_px+offset[1]), marker_size_px+offset[1]):
            locations.append((i, j))

    for i, marker in enumerate(markers):
        location = locations[i]
        s_row, s_col = np.c_[location, location + np.r_[marker.shape]]
        image[slice(*s_row), slice(*s_col)] = marker
    
    writer = png.Writer(*np.flip(image.shape), greyscale=True, compression=0, x_pixels_per_unit=int(dpmm*1000), y_pixels_per_unit=int(dpmm*1000), unit_is_meter=True)
    outfile_name = 'markers_{}_{}_to_{}.png'.format(random_seed, marker_ids[0], marker_ids[-1])
    with open(outfile_name, "wb") as file:
        writer.write(file, [row for row in image])
    print('done')

print_markers(66, np.arange(10))