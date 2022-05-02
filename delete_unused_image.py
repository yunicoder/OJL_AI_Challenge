import os
import argparse
import csv

def getImageName(path):
    return os.path.basename(path)

def getUseDataPath(data_path):
    rows = []
    with open(data_path + '/driving_log.csv') as csvFile:
        reader = csv.reader(csvFile)            
        for row in reader:
            rows.append(row)

    use_image_name = []
    for row in rows:
        use_image_name.append(getImageName(row[0]))
        use_image_name.append(getImageName(row[1]))
        use_image_name.append(getImageName(row[2]))

    return use_image_name

def deleteUnuseImage(data_path, use_image_name):
    files = os.listdir(data_path+'/IMG')

    cnt = 0
    for file in files:
        if file not in use_image_name:
                cnt += 1

#     print('{}/IMG/内の使用していないファイル({}件)を本当に削除しますか？(y/n) : '.format(data_path, cnt), end='')
    print('do you wanna delete {} files (path: {}/IMG)?(y/n):'.format(cnt, data_path), end='')
    s = input()

    cnt = 0
    if s == "y":
        for file in files:
            if file not in use_image_name:
                os.remove(data_path+'/IMG/'+file)
                cnt += 1

        print("delete {} files".format(cnt))
        
def main(data_path):
    use_image_name = getUseDataPath(data_path)
    deleteUnuseImage(data_path, use_image_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        '--data_path',
        default='data',
        type=str,
        help='dataのあるパス'
    )
    args = parser.parse_args()

    main(args.data_path)
