from get_camera_transformation import get_camera_transformation

#unknown serial nr
res = get_camera_transformation('aaa')
print(res)

#known serial nr
res = get_camera_transformation('cam2')
print(res)