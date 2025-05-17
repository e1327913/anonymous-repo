import requests
import logging
import re
import csv
import os
import argparse
import datetime
import json
import time
import random
import spacy
import spacy_fastlang # Do not remove this. This is for language_detector
from bs4 import BeautifulSoup
from multiprocessing import Pool

POLITIFACT_CONFIG = {
    'urls': [
        'https://www.politifact.com/factchecks/list/?ruling=true',
        'https://www.politifact.com/factchecks/list/?ruling=mostly-true',
        'https://www.politifact.com/factchecks/list/?ruling=half-true',
        'https://www.politifact.com/factchecks/list/?ruling=barely-true',
        'https://www.politifact.com/factchecks/list/?ruling=false',
        'https://www.politifact.com/factchecks/list/?ruling=pants-fire', 
        'https://www.politifact.com/factchecks/list/',
        'https://www.politifact.com/factchecks/list/?category=health-check',
        'https://www.politifact.com/factchecks/list/?category=coronavirus',
        'https://www.politifact.com/factchecks/list/?category=2024-senate-elections',
        'https://www.politifact.com/factchecks/list/?category=abortion',
        'https://www.politifact.com/factchecks/list/?category=afghanistan',
        'https://www.politifact.com/factchecks/list/?category=agriculture',
        'https://www.politifact.com/factchecks/list/?category=animals',
        'https://www.politifact.com/factchecks/list/?category=ask-politifact',
        'https://www.politifact.com/factchecks/list/?category=ad-watch',
        'https://www.politifact.com/factchecks/list/?category=after-the-fact',
        'https://www.politifact.com/factchecks/list/?category=alcohol',
        'https://www.politifact.com/factchecks/list/?category=artificial-intelligence',
        'https://www.politifact.com/factchecks/list/?category=autism',
        'https://www.politifact.com/factchecks/list/?category=bankruptcy',
        'https://www.politifact.com/factchecks/list/?category=bipartisanship',
        'https://www.politifact.com/factchecks/list/?category=bush-adminstration',
        'https://www.politifact.com/factchecks/list/?category=baseball',
        'https://www.politifact.com/factchecks/list/?category=border-security',
        'https://www.politifact.com/factchecks/list/?category=campaign-finance',
        'https://www.politifact.com/factchecks/list/?category=cap-and-trade',
        'https://www.politifact.com/factchecks/list/?category=children',
        'https://www.politifact.com/factchecks/list/?category=city-budget',
        'https://www.politifact.com/factchecks/list/?category=civil-rights',
        'https://www.politifact.com/factchecks/list/?category=congress',
        'https://www.politifact.com/factchecks/list/?category=consumer-safety',
        'https://www.politifact.com/factchecks/list/?category=corporations',
        'https://www.politifact.com/factchecks/list/?category=county-budget',
        'https://www.politifact.com/factchecks/list/?category=crime',
        'https://www.politifact.com/factchecks/list/?category=candidate-biography',
        'https://www.politifact.com/factchecks/list/?category=census',
        'https://www.politifact.com/factchecks/list/?category=china',
        'https://www.politifact.com/factchecks/list/?category=city-government',
        'https://www.politifact.com/factchecks/list/?category=climate-change',
        'https://www.politifact.com/factchecks/list/?category=constitutional-amendments',
        'https://www.politifact.com/factchecks/list/?category=corrections-and-updates',
        'https://www.politifact.com/factchecks/list/?category=county-government',
        'https://www.politifact.com/factchecks/list/?category=criminal-justice',
        'https://www.politifact.com/factchecks/list/?category=death-penalty',
        'https://www.politifact.com/factchecks/list/?category=debt',
        'https://www.politifact.com/factchecks/list/?category=disability',
        'https://www.politifact.com/factchecks/list/?category=debates',
        'https://www.politifact.com/factchecks/list/?category=deficit',
        'https://www.politifact.com/factchecks/list/?category=drugs',
        'https://www.politifact.com/factchecks/list/?category=ebola',
        'https://www.politifact.com/factchecks/list/?category=education',
        'https://www.politifact.com/factchecks/list/?category=energy',
        'https://www.politifact.com/factchecks/list/?category=ethics',
        'https://www.politifact.com/factchecks/list/?category=economy',
        'https://www.politifact.com/factchecks/list/?category=elections',
        'https://www.politifact.com/factchecks/list/?category=environment',
        'https://www.politifact.com/factchecks/list/?category=facebook-fact-checks',
        'https://www.politifact.com/factchecks/list/?category=families',
        'https://www.politifact.com/factchecks/list/?category=financial-regulation',
        'https://www.politifact.com/factchecks/list/?category=florida-amendments',
        'https://www.politifact.com/factchecks/list/?category=food-safety',
        'https://www.politifact.com/factchecks/list/?category=fake-news',
        'https://www.politifact.com/factchecks/list/?category=federal-budget',
        'https://www.politifact.com/factchecks/list/?category=fires',
        'https://www.politifact.com/factchecks/list/?category=food',
        'https://www.politifact.com/factchecks/list/?category=foreign-policy',
        'https://www.politifact.com/factchecks/list/?category=gambling',
        'https://www.politifact.com/factchecks/list/?category=good-enough-to-be-true',
        'https://www.politifact.com/factchecks/list/?category=guns',
        'https://www.politifact.com/factchecks/list/?category=gas-prices',
        'https://www.politifact.com/factchecks/list/?category=government-regulation',
        'https://www.politifact.com/factchecks/list/?category=health-care',
        'https://www.politifact.com/factchecks/list/?category=history',
        'https://www.politifact.com/factchecks/list/?category=homeless',
        'https://www.politifact.com/factchecks/list/?category=human-rights',
        'https://www.politifact.com/factchecks/list/?category=health-check',
        'https://www.politifact.com/factchecks/list/?category=homeland-security',
        'https://www.politifact.com/factchecks/list/?category=housing',
        'https://www.politifact.com/factchecks/list/?category=immigration',
        'https://www.politifact.com/factchecks/list/?category=income',
        'https://www.politifact.com/factchecks/list/?category=iran',
        'https://www.politifact.com/factchecks/list/?category=islam',
        'https://www.politifact.com/factchecks/list/?category=impeachment',
        'https://www.politifact.com/factchecks/list/?category=infrastructure',
        'https://www.politifact.com/factchecks/list/?category=iraq',
        'https://www.politifact.com/factchecks/list/?category=israel',
        'https://www.politifact.com/factchecks/list/?category=jan-6',
        'https://www.politifact.com/factchecks/list/?category=jobs',
        'https://www.politifact.com/factchecks/list/?category=kagan-nomination',
        'https://www.politifact.com/factchecks/list/?category=katrina',
        'https://www.politifact.com/factchecks/list/?category=labor',
        'https://www.politifact.com/factchecks/list/?category=lgbtq',
        'https://www.politifact.com/factchecks/list/?category=legal-issues',
        'https://www.politifact.com/factchecks/list/?category=lottery',
        'https://www.politifact.com/factchecks/list/?category=marijuana',
        'https://www.politifact.com/factchecks/list/?category=medicaid',
        'https://www.politifact.com/factchecks/list/?category=message-machine',
        'https://www.politifact.com/factchecks/list/?category=message-machine-2014',
        'https://www.politifact.com/factchecks/list/?category=marriage',
        'https://www.politifact.com/factchecks/list/?category=medicare',
        'https://www.politifact.com/factchecks/list/?category=message-machine-2012',
        'https://www.politifact.com/factchecks/list/?category=military',
        'https://www.politifact.com/factchecks/list/?category=natural-disasters',
        'https://www.politifact.com/factchecks/list/?category=new-hampshire-2012',
        'https://www.politifact.com/factchecks/list/?category=negative-campaigning',
        'https://www.politifact.com/factchecks/list/?category=nuclear',
        'https://www.politifact.com/factchecks/list/?category=obama-birth-certificate',
        'https://www.politifact.com/factchecks/list/?category=oil-spill',
        'https://www.politifact.com/factchecks/list/?category=occupy-wall-street',
        'https://www.politifact.com/factchecks/list/?category=party-support',
        'https://www.politifact.com/factchecks/list/?category=pensions',
        'https://www.politifact.com/factchecks/list/?category=politifacts-top-promises',
        'https://www.politifact.com/factchecks/list/?category=pop-culture',
        'https://www.politifact.com/factchecks/list/?category=poverty',
        'https://www.politifact.com/factchecks/list/?category=public-health',
        'https://www.politifact.com/factchecks/list/?category=public-service',
        'https://www.politifact.com/factchecks/list/?category=patriotism',
        'https://www.politifact.com/factchecks/list/?category=polls',
        'https://www.politifact.com/factchecks/list/?category=population',
        'https://www.politifact.com/factchecks/list/?category=privacy-issues',
        'https://www.politifact.com/factchecks/list/?category=public-safety',
        'https://www.politifact.com/factchecks/list/?category=pundits',
        'https://www.politifact.com/factchecks/list/?category=race-ethnicity',
        'https://www.politifact.com/factchecks/list/?category=redistricting',
        'https://www.politifact.com/factchecks/list/?category=religion',
        'https://www.politifact.com/factchecks/list/?category=russia',
        'https://www.politifact.com/factchecks/list/?category=recreation',
        'https://www.politifact.com/factchecks/list/?category=regulation',
        'https://www.politifact.com/factchecks/list/?category=retirement',
        'https://www.politifact.com/factchecks/list/?category=science',
        'https://www.politifact.com/factchecks/list/?category=sexuality',
        'https://www.politifact.com/factchecks/list/?category=social-security',
        'https://www.politifact.com/factchecks/list/?category=space',
        'https://www.politifact.com/factchecks/list/?category=state-budget',
        'https://www.politifact.com/factchecks/list/?category=stimulus',
        'https://www.politifact.com/factchecks/list/?category=second-term-promise',
        'https://www.politifact.com/factchecks/list/?category=small-business',
        'https://www.politifact.com/factchecks/list/?category=sotomayor-nomination',
        'https://www.politifact.com/factchecks/list/?category=sports',
        'https://www.politifact.com/factchecks/list/?category=states',
        'https://www.politifact.com/factchecks/list/?category=supreme-court',
        'https://www.politifact.com/factchecks/list/?category=10-news-tampa-bay',
        'https://www.politifact.com/factchecks/list/?category=technology',
        'https://www.politifact.com/factchecks/list/?category=2018-california-governors-race',
        'https://www.politifact.com/factchecks/list/?category=tourism',
        'https://www.politifact.com/factchecks/list/?category=transparency',
        'https://www.politifact.com/factchecks/list/?category=taxes',
        'https://www.politifact.com/factchecks/list/?category=terrorism',
        'https://www.politifact.com/factchecks/list/?category=abc-news-week',
        'https://www.politifact.com/factchecks/list/?category=trade',
        'https://www.politifact.com/factchecks/list/?category=transportation',
        'https://www.politifact.com/factchecks/list/?category=ukraine',
        'https://www.politifact.com/factchecks/list/?category=urban',
        'https://www.politifact.com/factchecks/list/?category=unions',
        'https://www.politifact.com/factchecks/list/?category=veterans',
        'https://www.politifact.com/factchecks/list/?category=voting-record',
        'https://www.politifact.com/factchecks/list/?category=voter-id-laws',
        'https://www.politifact.com/factchecks/list/?category=water',
        'https://www.politifact.com/factchecks/list/?category=weather',
        'https://www.politifact.com/factchecks/list/?category=women',
        'https://www.politifact.com/factchecks/list/?category=wealth',
        'https://www.politifact.com/factchecks/list/?category=welfare',
        'https://www.politifact.com/factchecks/list/?category=workers',
    ],
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def extract_all_politifact_links(args):
    url, PAGE_LIMIT = args['url'], args['page_limit']
    try:
        list1 = [3, 5, 7]
        sleep_timer = random.choice(list1)
        time.sleep(sleep_timer)
        html = requests.get(url)
        soup = BeautifulSoup(html.text, 'html.parser')
        personalities = set()
        articles = set()
        # Gather Personalities and Articles from all HREF first
        for list_item in soup.find_all("li", {"class": "o-listicle__item"}):
            hrefs = list_item.find_all("a")
            personality = hrefs[0].get("href")
            article = hrefs[1].get("href")
            stubs = re.split('/', article)
            assert len(stubs) == 8
            try:
                # is_valid = translator.detect(stubs[6]).lang == 'en'
                nlp = spacy.load('en_core_web_sm')
                nlp.add_pipe("language_detector") 
                text = ' '.join(stubs[6].split('-')[:-1])
                doc = nlp(text)
                is_valid = doc._.language == 'en'
                logging.info(f"Stub: {stubs[6]} | Formatted Stub: {text} | Language: {doc._.language} | Valid: {is_valid} ")
                if is_valid:
                    personalities.add(personality)
                    articles.add(article)
            except:
                raise ValueError(article)
        
        # Get next page link and place it in the array
        new_stub_url = None
        page_number = None
        for button in soup.find_all("a", {"class": "c-button c-button--hollow"}):
            page_number_results = re.findall('\\?page=[0-9]*', button.get("href"))
            if len(page_number_results) > 0 :
                page_number = int(re.sub('\\?page=', '', page_number_results[0]))
            if button.text == 'Next' and page_number is not None and page_number <= PAGE_LIMIT: # 50 pages because it is not that old yet
                new_stub_url = "https://www.politifact.com/factchecks/list/" + button.get("href")
                
        return new_stub_url, new_stub_url is not None, personalities, articles, page_number
    except Exception as e:
        raise e
    
def get_all_politifact_links(args):

    SAVE_PATH = args.save_path
    PAGE_LIMIT = args.page_limit

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()

    urls = POLITIFACT_CONFIG['urls']
    personalities = set()
    articles = set()
    is_valid = True
    current_page_number = 1
    with Pool() as pool:
        while is_valid:
            inputs = [{'url': url, 'page_limit': PAGE_LIMIT} for url in urls]
            outputs = pool.map(extract_all_politifact_links, inputs)
            none_arr = []
            new_urls = []
            for url, is_next_url_available, extracted_personalities, extracted_articles, page_number in outputs:
                none_arr.append(is_next_url_available)
                if is_next_url_available:
                    new_urls.append(url)
                    personalities = personalities.union(extracted_personalities)
                    articles = articles.union(extracted_articles)
                    current_page_number = page_number
                    
            if len(new_urls) == 0:
                is_valid = False
            else:
                logging.info(f"New URLs found. Continuing the process. Page: {current_page_number} / {PAGE_LIMIT}")
                urls = new_urls

    logging.info(f'Finish mining links for Politifact. Page limit is {PAGE_LIMIT} for all classes')

    # Save Personalities
    logging.info(f"Saving Personalities under politifact_personalities_links.csv")
    with open(os.path.join(SAVE_PATH, 'politifact_personalities_links.csv'), 'w') as file:
        writer = csv.writer(file)
        for url_stub in personalities:
            writer.writerow(['https://www.politifact.com' + url_stub])
        file.close()
    logging.info(f"Done saving to politifact_personalities_links.csv")

    # Save Articles
    logging.info(f"Saving Articles under politifact_article_links.csv")
    with open(os.path.join(SAVE_PATH, 'politifact_article_links.csv'), 'w') as file:
        writer = csv.writer(file)
        for url_stub in articles:
            writer.writerow(['https://www.politifact.com' + url_stub])
        file.close()
    logging.info(f"Done saving to politifact_article_links.csv")

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'politifact-raw'), help='Script output location')
    parser.add_argument('--page_limit', type=int, default=50, help='Page Limit for Scrapping in PolitiFact')

    args = parser.parse_args()
    get_all_politifact_links(args)