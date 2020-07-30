import numpy as np
import cv2

import pylab
import png


# input setup
marker_size_mm = 36  # length of a side of the markers in mm
square_size_mm = 42
num_of_diamonds = 12
markers_bits = 4  # bits per side of the markers data matrix

safety_margin_mm = np.r_[10, 10, 10, 10]  # offsets around the paper due to printing (top, down, left, right)
use_paper = "a4"  # render to A4 or A3 paper
dpi = 2400  # DPI
final_dpi = int(dpi / 4)

bitdepth = 8  # used when saving to PNG; 8=0-255 values, 1=0/1
marker_tag_size_mm = np.r_[5, 10]  # size of the marker tag (text with marker ID)
random_seed = 65536  # random seed value for reproducibility

# internal vars
markers_dict_size = num_of_diamonds * 4  # size of the dictionary used to generate the markers
rnd_generator = np.random.RandomState(random_seed)
aruco_dict = cv2.aruco.Dictionary_create(markers_dict_size, markers_bits, random_seed)

dpmm = dpi / 25.4
final_dpmm = final_dpi / 25.4
final_scale = int(dpi // final_dpi)
a4dim_mm = np.array([210, 297])
a3dim_mm = np.array([297, 420])

if use_paper.lower() == "a4":
    paper_mm = a4dim_mm
elif use_paper.lower() == "a3":
    paper_mm = a3dim_mm
else:
    raise Exception(f"Unknown paper format: {use_paper}.")

paper_px = np.round(paper_mm * dpmm).astype(np.int)
center_px = np.round(paper_px / 2).astype(np.int)
image_center = tuple(np.flip(center_px))
marker_size_px = np.round(marker_size_mm * dpmm).astype(np.int)
square_size_px = np.round(square_size_mm * dpmm).astype(np.int)
marker_tag_size_px = np.round(marker_tag_size_mm * dpmm).astype(np.int)

font_scale = final_scale * 3
font_thickness = 6

# for printing and stacking
safety_topleft = (np.round(safety_margin_mm[[0, 2]] * dpmm).astype(np.int)) // final_scale
safety_margin_sum_px = (np.round(safety_margin_mm.reshape(2, 2).sum(axis=1) * dpmm).astype(np.int)) // final_scale
diamond_size_2d_px = (np.round(np.r_[square_size_mm, square_size_mm] * 3 * dpmm).astype(np.int)) // final_scale
paper_stack_size = np.int32((paper_px // final_scale - safety_margin_sum_px * 2) / diamond_size_2d_px)
paper_stack_length = paper_stack_size[0] * paper_stack_size[1]
stack_division_px = (paper_px // final_scale - safety_margin_sum_px) / paper_stack_size
stack_offsets_px = (stack_division_px - diamond_size_2d_px) / 2

def generate_aruco(marker_id, marker_size_px):
    return aruco_dict.drawMarker(marker_id, marker_size_px)


def get_formated_text(array):
    return np.array2string(np.array(array), precision=2, separator=', ', suppress_small=True, formatter={'float_kind': lambda x: f"{x:0.2f}"})


def putTextChequers(img, text, position):
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, font_thickness)
    return cv2.putText(img, text, tuple((position - np.r_[text_size[0]] / 2 + np.r_[0, text_size[1]]).astype(int)), cv2.FONT_HERSHEY_COMPLEX_SMALL, font_scale, 255, font_thickness)


grid = np.mgrid[0:square_size_px * 2.9:square_size_px, 0:square_size_px * 2.9:square_size_px].reshape(2, -1).T
square_shape_px = np.r_[square_size_px, square_size_px]
marker_offset_px = (square_shape_px - marker_size_px) / 2

paper_stack = []
id_stack = []
for diamond_idx in range(num_of_diamonds):
    chequers = np.ones([square_size_px * 3] * 2, dtype=np.uint8) * 255
    marker_ids = np.array(range(diamond_idx * 4, diamond_idx * 4 + 4))
    markers = [generate_aruco(marker_id, marker_size_px) for marker_id in marker_ids]
    for i, coords in enumerate(grid):
        if i % 2 == 0:  # square
            cv2.rectangle(chequers, tuple(coords.astype(int)), tuple((coords + square_shape_px).astype(int)), (0, 0, 0), -1)
        else:  # marker
            s_row, s_col = np.c_[coords + marker_offset_px, coords + square_shape_px - marker_offset_px].astype(int)
            chequers[slice(*s_row), slice(*s_col)] = markers[(i - 1) // 2]

    diamond_name = f"D {get_formated_text(marker_ids)}"
    chequers = putTextChequers(chequers, diamond_name, square_shape_px * 1.5)
    chequers = putTextChequers(chequers, f"square: {square_size_mm}mm", square_shape_px * 0.5)
    chequers = putTextChequers(chequers, f"marker: {marker_size_mm}mm", square_shape_px * 2.5)
    chequers = putTextChequers(chequers, "Dict:", np.r_[square_size_px * 2.5, square_size_px * 0.5 - font_scale * 45])
    chequers = putTextChequers(chequers, f"N={markers_dict_size}", np.r_[square_size_px * 2.5, square_size_px * 0.5 - font_scale * 15])
    chequers = putTextChequers(chequers, f"bits={markers_bits}", np.r_[square_size_px * 2.5, square_size_px * 0.5 + font_scale * 15])
    chequers = putTextChequers(chequers, f"seed={random_seed}", np.r_[square_size_px * 2.5, square_size_px * 0.5 + font_scale * 45])
    chequers = putTextChequers(chequers, f"{diamond_idx}", np.r_[square_size_px * 0.5, square_size_px * 2.5])

    chequers = np.pad(chequers, ((int(3 * dpmm), ) * 2, ) * 2, "constant", constant_values=255)
    chequers = np.pad(chequers, ((int(0.5 * dpmm), ) * 2, ) * 2, "constant", constant_values=0)
    # scale chequers
    chequers = cv2.resize(chequers, tuple(np.r_[chequers.shape] // final_scale), interpolation=cv2.INTER_AREA)
    # chequers = cv2.resize(chequers, tuple(np.flip(square_shape_px * 3) // final_scale), interpolation=cv2.INTER_AREA)

    # chequers format conversion
    if bitdepth == 1:
        chequers = np.clip(chequers, 0, 1)
    # threshold to remove smooting
    ret, chequers = cv2.threshold(chequers, 127, 255, cv2.THRESH_BINARY)

    # show chequers
    pylab.imshow(chequers)

    # diamond marker stacking
    paper_stack.append(chequers)
    id_stack.append(diamond_idx)

    if len(paper_stack) == paper_stack_length:
        image = np.ones(paper_px // final_scale, dtype=np.uint8) * 255
        for i in range(paper_stack_size[0]):
            for j in range(paper_stack_size[1]):
                idx = i * paper_stack_size[0] + j
                img = paper_stack[idx]
                loc = safety_topleft + np.r_[i, j] * stack_division_px + stack_offsets_px
                s_row, s_col = np.c_[loc, loc + img.shape].astype(np.int32)
                image[slice(*s_row), slice(*s_col)] = img

        outfile_name = f"chd_stack_{get_formated_text(id_stack)}.png"
        writer = png.Writer(*np.flip(image.shape), greyscale=True, bitdepth=bitdepth, compression=0, x_pixels_per_unit=int(final_dpmm * 1000), y_pixels_per_unit=int(final_dpmm * 1000), unit_is_meter=True)

        with open(outfile_name, "wb") as file:
            writer.write(file, [row for row in image])

        print(f"Generated and saved diamond stack {outfile_name}.")

        paper_stack = []
        id_stack = []

    # save chequers
    outfile_name = f"ch_diamond_{get_formated_text(marker_ids)}.png"
    writer = png.Writer(*np.flip(chequers.shape), greyscale=True, bitdepth=bitdepth, compression=0, x_pixels_per_unit=int(final_dpmm * 1000), y_pixels_per_unit=int(final_dpmm * 1000), unit_is_meter=True)

    with open(outfile_name, "wb") as file:
        writer.write(file, [row for row in chequers])

    print(f"Generated and saved diamond {diamond_name}.")
