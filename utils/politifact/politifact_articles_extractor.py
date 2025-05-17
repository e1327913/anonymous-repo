import requests
import logging
import re
import csv
import os
import argparse
import datetime
import json
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_html_from_url(args):
    url, save_path = args['url'], args['save_path']
    logging.info(f"Mining Information from {url}")
    grab = requests.get(url)
    logging.info(f"Soupify {url} html")
    soup = BeautifulSoup(grab.text, 'html.parser')
    url_stubs = re.sub('https://www.politifact.com/', '', url)
    url_year = url_stubs.split('/')[1]
    logging.info(f'URL YEAR: {url_year}')
    is_article_not_dead = soup.find("h1", {"class": "m-notfound__title"}) is None
            
    if is_article_not_dead:
        logging.info(f"{url} article is valid. Saving to queue")
        HTML_FILE_NAME = '-'.join(url_stubs[:-1].split('/'))
        HTML_SAVE_PATH = os.path.join(save_path, f'{HTML_FILE_NAME}.html')
        output = {"url": url, 'html_save_path': HTML_SAVE_PATH, 'html': grab.content}
        return output
    else:
        logging.info(f'Skipping {url} due to old age or dead links')

def extract_politifact_articles_from_urls(args):
    SAVE_PATH = args.save_path
    ARTICLE_LINKS_FOLDER = args.article_links_folder
    ARTICLE_LINKS_PATH = os.path.join(ARTICLE_LINKS_FOLDER, 'politifact_article_links.csv')
    FILE_OUTPUT_PATH = os.path.join(SAVE_PATH, 'politifact_raw_articles.csv')

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    # Retrieve Article Links
    article_links = []
    with open(ARTICLE_LINKS_PATH, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for item in reader:
            article_links.append({
                'url': item[0],
                'save_path': SAVE_PATH
            })
    
    # Create Multiple processes
    with Pool() as pool:
        outputs = pool.map(extract_html_from_url, article_links)
        outputs = [output for output in outputs if output is not None]
        # Save output to csv
        logging.info(f'All processes are done. Saving to {FILE_OUTPUT_PATH}')
        with open(FILE_OUTPUT_PATH, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['url', 'html_file_location'])
            for output in tqdm(outputs, desc='Saving Item'):
                writer.writerow([output['url'], output['html_save_path']])
                with open(output['html_save_path'], 'wb+') as html_file:
                    html_file.write(output['html'])
                    html_file.close()
            file.close()

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'politifact-raw-article-html', timestamp), help='Script output location')
    parser.add_argument('--article_links_folder', type=str, default=os.path.join('script_outputs', 'politifact-raw', '2025-03-19-22-37-52'), help='politifact_articles_links.csv location')

    args = parser.parse_args()
    extract_politifact_articles_from_urls(args)