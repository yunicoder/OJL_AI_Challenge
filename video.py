import argparse
from moviepy.editor import ImageSequenceClip
import os

def main():
    parser = argparse.ArgumentParser(description='creat driving movie.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='image folder pass'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='video frame'
    )
    args = parser.parse_args()

    #画像のパスのリストを作成する
    image_list = [os.path.abspath('{}'.format(args.image_folder)) + '/' + path for path in sorted(os.listdir(args.image_folder))]

    #画像パスリストをImageSequenceClipに渡す
    clip = ImageSequenceClip(image_list, fps=args.fps)
    video_file_name = args.image_folder + '.mp4'
    
    print('make movie. file name is {}.mp4'.format(args.image_folder))
    clip.write_videofile(video_file_name)
    
if __name__ == '__main__':
    main()


