# -*- encoding: utf-8 -*-
# @Author  :  Liang Hao
# @Time    :  2023/09/11 18:50:08
# @FileName:  down_icvl_hsi.py
# @Contact :  lianghao02@megvii.com


import os
import requests
import threading

from tqdm import tqdm
from queue import Queue
from bs4 import BeautifulSoup

def download_file(url, path):
    response = requests.get(url, stream=True, timeout=10)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, 'wb') as file:
        for data in tqdm(response.iter_content(32 * 1024), 
                         total=total_size // (32 * 1024), 
                         unit='KB', unit_scale=True):
            file.write(data)
    

def get_icvl_dataset(file_type='mat'):
    files = []
    url = 'https://icvl.cs.bgu.ac.il/hyperspectral/'
    response = requests.get(url)
    
    if response.status_code != 200:
        raise f"response : {response.status_code}"
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    for _, hs_box in enumerate(tqdm(soup.find_all('div', class_='hs_box'), desc='search')):
        file_name = hs_box.find('h4', class_='hs_title').text
        
        for a_tag in hs_box.find('div', class_='hs-files').find_all('a'):
            type_ = a_tag.text.strip()
            if file_type.lower() in type_.lower():
                files.append((file_name + type_, a_tag['href']))
                break
    
    return files


def worker(queue_obj, save_dir):
    print(f"start downloading : {queue_obj.qsize()}")
    while not queue_obj.empty():
        url = queue_obj.get()
        path = os.path.join(save_dir, url[0])
        download_file(url[1], path)
        queue_obj.task_done()
        

def main(file_type, save_dir):
    files = get_icvl_dataset(file_type)
    
    num_threads = min(len(files), 4)
    url_queue = Queue()
    
    for file in files:
        url_queue.put(file)
    
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(url_queue, save_dir))
        t.daemon = True
        t.start()
    url_queue.join()


if __name__ == '__main__':
    file_type = '.mat'
    save_dir = '/home/Public/Train/denoise/HSI/ICVL'
    
    main(file_type, save_dir)