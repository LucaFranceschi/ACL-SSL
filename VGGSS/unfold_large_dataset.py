'''
Docstring for VGGSS.unfold_large_dataset

Meant to modify total_video_frames into frames, by taking the center frame from each video and copying it to frames with the correct name.
'''
import os, shutil

if __name__ == '__main__':

    src_data_dir = '/data/upftfg27/lfranceschi/vggsound/total_video_frames'
    dst_data_dir = '/data/upftfg27/lfranceschi/vggsound/frames'

    os.makedirs(dst_data_dir, exist_ok=True)

    video_names = os.listdir(src_data_dir)

    for v_name in video_names:
        frames = os.listdir(os.path.join(src_data_dir, v_name))
        center_frame = frames[round(len(frames)/2)]

        shutil.copy(os.path.join(src_data_dir, v_name, center_frame), os.path.join(dst_data_dir, v_name, '.jpg'))
