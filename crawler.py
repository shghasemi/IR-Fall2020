import json
import time
import re

from selenium import webdriver
from bs4 import BeautifulSoup


class Crawler:
    def __init__(self):
        self.driver = self.init_driver()
        self.BASE_URL = 'https://academic.microsoft.com/paper/'
        self.PAPER_RGX = r'paper/(\d+)(/reference)?'

    def init_driver(self):
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--user-agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) ' +
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.96 Safari/537.36"')
        return webdriver.Chrome(options=options)

    def get_html(self, paper_id):
        self.driver.get(self.BASE_URL + paper_id)
        time.sleep(2)
        html = self.driver.page_source
        return BeautifulSoup(html, features='html.parser')

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
        paper = self.make_json(paper_id, parsed_html)
        with open(f'papers/{paper_id}.json', 'w') as file:
            json.dump(paper, file, indent=2)

    def get_authors(self, parsed_html):
        authors_div = parsed_html.find('div', attrs={'class': 'authors'})
        return [a.text for a in authors_div.find_all('a', attrs={'class': 'au-target author link'})]

    def get_references(self, parsed_html):
        ref_lst = [a.get('href', None) for a in parsed_html.find_all('a', attrs={'class': 'title au-target'})]
        ref_ids = []
        ref_cnt = 0
        for ref in ref_lst:
            if ref_cnt >= 10:
                break
            if ref is None:
                continue
            ref_id = re.search(self.PAPER_RGX, ref)
            if ref_id:
                ref_ids.append(ref_id.group(1))
                ref_cnt += 1

        return ref_ids


if __name__ == '__main__':
    crawler = Crawler()
    crawler.get_paper('2981549002')
