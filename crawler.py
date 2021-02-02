import json
import time
import re

from selenium import webdriver
from bs4 import BeautifulSoup

from os import listdir
import numpy as np


class Crawler:
    def __init__(self, max_cnt=5000, max_wait_thr=10):
        self.driver = None
        self.init_driver()
        self.BASE_URL = 'https://academic.microsoft.com/paper/'
        self.PAPER_RGX = r'paper/(\d+)(/reference)?'
        self.max_cnt = max_cnt
        self.max_wait_thr = max_wait_thr
        self.queue = []
        self.init_queue()
        self.explored = set()
        self.extend_queue = True

    def init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) ' +
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"')
        self.driver = webdriver.Chrome(options=options)

    def init_queue(self):
        with open('crawler/start.txt') as file:
            init_urls = file.readlines()
        for url in init_urls:
            paper_id = re.search(self.BASE_URL + r'(\d+)', url.strip())
            if paper_id:
                self.queue.append(paper_id.group(1))

    def crawl(self):
        cnt = 0
        while cnt < self.max_cnt:
            paper_id = self.queue.pop(0)
            if paper_id in self.explored:
                continue

            paper = self.get_paper(paper_id)
            if paper is None:
                print(f'Driver fault - Paper{cnt:4}/{self.max_cnt:4} - {paper_id}')
                self.driver.quit()
                self.init_driver()
                self.queue.insert(0, paper_id)
                continue

            self.explored.add(paper_id)
            cnt += 1
            if cnt % 100 == 0:
                print(f'Paper {cnt:4}/{self.max_cnt} - {paper_id}')
            if not paper['references']:
                print(paper_id)
            if self.extend_queue:
                self.queue.extend(paper['references'])
                self.extend_queue = len(set(self.queue + list(self.explored))) < self.max_cnt
        self.driver.quit()

    def wait(self):
        cnt = 0
        loaded = False
        while not loaded and cnt < self.max_wait_thr:
            time.sleep(1.5)
            cnt += 1
            loaded = '<div class="primary_paper">' in self.driver.page_source
            # print(loaded)
        return loaded

    def get_html(self, paper_id):
        self.driver.get(self.BASE_URL + paper_id)
        loaded = self.wait()
        if loaded:
            return BeautifulSoup(self.driver.page_source, features='html.parser')
        else:
            return None

    def make_json(self, paper_id, parsed_html):
        paper = {'id': paper_id,
                 'title': parsed_html.head.find('title').text,
                 'abstract': parsed_html.body.find('p', attrs={'class': None}).text,
                 'date': parsed_html.body.find('span', attrs={'class': 'year'}).text,
                 'authors': self.get_authors(parsed_html),
                 'references': self.get_references(parsed_html)}
        return paper

    def get_paper(self, paper_id):
        parsed_html = self.get_html(paper_id)
        if parsed_html is None:
            return None
        paper = self.make_json(paper_id, parsed_html)
        with open(f'crawler/papers/{paper_id}.json', 'w') as file:
            json.dump(paper, file, indent=2)
        return paper

    def get_authors(self, parsed_html):
        authors_div = parsed_html.find('div', attrs={'class': 'authors'})
        return [a.text for a in authors_div.find_all('a', attrs={'class': 'au-target author link'})]

    def get_references(self, parsed_html):
        ref_id_lst = []
        ref_cnt = 0
        primary_ref_lst = parsed_html.find_all('div', attrs={'class': 'primary_paper'})
        for ref in primary_ref_lst:
            if ref_cnt >= 10:
                break
            ref = ref.find('a', attrs={'class': 'title au-target'})
            if ref is None or ref.get('href', None) is None:
                continue
            ref_id = re.search(self.PAPER_RGX, ref['href'])
            if ref_id:
                ref_id_lst.append(ref_id.group(1))
                ref_cnt += 1

        return ref_id_lst


class PageRank:
    def __init__(self, alpha, max_iter):
        self.alpha = alpha
        self.max_iter = max_iter
        self.paper_id_lst = None
        self.A = None
        self.init_paper_id_lst()
        self.init_adjacency_matrix()

    def init_paper_id_lst(self):
        file_name_lst = listdir('crawler/papers')
        self.paper_id_lst = [file_name[:-5] for file_name in file_name_lst]

    def init_adjacency_matrix(self):
        n = len(self.paper_id_lst)
        idx = dict(zip(self.paper_id_lst, list(range(n))))
        self.A = np.zeros((n, n))
        for paper_id in self.paper_id_lst:
            with open(f'crawler/papers/{paper_id}') as file:
                paper = json.load(file)
            for ref in paper['references']:
                if ref in idx:
                    self.A[idx[paper_id], idx[ref]] = 1.


if __name__ == '__main__':
    crawler = Crawler(max_cnt=50)
    crawler.crawl()
    # crawler.get_paper('2981549002')
