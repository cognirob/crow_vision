import numpy as np


def ftl_pcl2numpy(pcl_msg, uses_stupid_bgr_format=True):
    # warning: this assumes that the machine and PCL have the same endianness!
    step = int(pcl_msg.point_step / 4)  # point offset in dwords
    rgb_offset = int(next((f.offset for f in pcl_msg.fields if f.name == "rgb")) / 4)  # rgb offset in dwords
    dwordData = np.frombuffer(pcl_msg.data, dtype=np.byte).view(np.float32).reshape(-1, step)  # organize the bytes into double words (4 bytes)
    xyz = dwordData[:, :3]  # assuming x, y, z are the first three dwords
    rgb_raw = dwordData[:, rgb_offset]  # extract rgb data
    rgb3d = rgb_raw.view(np.uint32).copy().view(np.uint8).reshape(-1, 4)[:, slice(1, None) if pcl_msg.is_bigendian else slice(None, 3)]
    if uses_stupid_bgr_format:
        rgb3d = np.fliplr(rgb3d)
    return xyz, rgb3d, rgb_raw
