import argparse
from moviepy.editor import ImageSequenceClip
import os


def main():
    parser = argparse.ArgumentParser(description='create driving movie.')
    parser.add_argument(
        'data_folder_name',
        type=str,
        default='',
        help='data_each_situation/[THIS ARG]/IMG'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='video frame'
    )
    args = parser.parse_args()


    image_folder = './data_each_situation/' + args.data_folder_name + '/IMG'

    ### 画像のパスのリストを作成する
    image_list = [os.path.abspath('{}'.format(image_folder)) + '/'
                  + path for path in sorted(os.listdir(image_folder))]

    ### 画像パスリストをImageSequenceClipに渡す
    clip = ImageSequenceClip(image_list, fps=args.fps)
    video_file = './data_each_situation/' + args.data_folder_name + '/' + args.data_folder_name + '.mp4'
    
    print('make movie. file name is {}.mp4'.format(args.data_folder_name))
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()
