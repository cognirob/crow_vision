import numpy as np

from sensor_msgs.msg import PointCloud2, PointField


def ftl_pcl2numpy(pcl_msg, uses_stupid_bgr_format=True):
    """
    convert ROS2 PointCloud2 pcl_msg to numpy ndarray
    """
    # warning: this assumes that the machine and PCL have the same endianness!
    step = int(pcl_msg.point_step / 4)  # point offset in dwords
    dwordData = np.frombuffer(pcl_msg.data, dtype=np.byte).view(np.float32).reshape(-1, step)  # organize the bytes into double words (4 bytes)
    xyz = dwordData[:, :3]  # assuming x, y, z are the first three dwords
    if "rgb" in [f.name for f in pcl_msg.fields]:
        rgb_offset = int(next((f.offset for f in pcl_msg.fields if f.name == "rgb")) / 4)  # rgb offset in dwords    
        rgb_raw = dwordData[:, rgb_offset]  # extract rgb data
        rgb3d = rgb_raw.view(np.uint32).copy().view(np.uint8).reshape(-1, 4)[:, slice(1, None) if pcl_msg.is_bigendian else slice(None, 3)]
        if uses_stupid_bgr_format:
            rgb3d = np.fliplr(rgb3d)
        return xyz, rgb3d, rgb_raw
    else:
        return xyz, None, None


def ftl_numpy2pcl(xyz, orig_header, rgb=None):
    """
    inverse method to ftl_pcl2numpy. 
    Converts np array to ROS2 PointCloud2. 

    @arg rgb: additional np array with color. Then use xyz+rgb pointcloud. Otherwise only xyz is used (default).
    """
    itemsize = np.dtype(np.float32).itemsize
    assert xyz.shape[0] == 3, "xyz must be only 'xyz' data in shape (3,N)"
    num_points = xyz.shape[1]

    if rgb is not None:
        assert num_points == len(rgb), "color must have same number of points"
        fields = [PointField(name=n, offset=i*itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(list('xyz') + ['rgb'])]
        fields[-1].offset = 16
        k = 5
        dataa = np.concatenate((xyz.T, np.zeros((num_points, 1), dtype=np.float32), rgb[:, np.newaxis]), axis=1)
    else:
        fields = [PointField(name=n, offset=i*itemsize, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(list('xyz'))]
        fields[-1].offset = 16 #FIXME probably not 16 is in xyz+rgb version
        k = 3
        dataa = xyz.T

    #fill PointCloud2 correctly according to https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0#file-dragon_pointcloud-py-L32
    pcl = PointCloud2(
        header=orig_header,
        height=1,
        width=num_points,
        fields=fields,
        point_step=(itemsize * k),  #=xyz + padding + rgb
        row_step=(itemsize * k * num_points),
        data=dataa.tobytes()
        )
    return pcl

