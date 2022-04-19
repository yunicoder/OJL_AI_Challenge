#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def main():
    import sys, subprocess, getpass, os, json, requests
    from urllib import request, error
    from os import path
    
    argv = sys.argv
    argc = len(argv)
    Course = argv[1]
    username = argv[2]
    activationKey = argv[3]
    
    if(os.path.isfile('./submit.txt')):
        f = open('./submit.txt', 'r')
        alltxt = f.readlines()
        f.close()
        lap = alltxt[len(alltxt)-1].strip()  
    else:
        print('submit.txt not found.')
        return False

    url = "https://"
    url+= "zero2one.jp/wp-json/wp/v2/sdcarwritemysql"
    url+= "/{username}".format(username=username)
    url+= "/{activationKey}".format(activationKey=activationKey)
    url+= "/{lap}".format(lap=lap)

    try:
        html = requests.get(url).json()
    except Exception as e:
        print(e)
        return False
    
    print(html['message'])
    return html

if __name__ == '__main__':
    main()
    
