import numpy as np
import pandas as pd
import logging
import re
import csv
import os
import argparse
from datetime import datetime
from uuid import uuid4
import json
from bs4 import BeautifulSoup
from multiprocessing import Pool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
        
def extract_details_from_o_stage(o_stage):
    instigator = o_stage.find_all("a", {"class": "m-statement__name"})[0]
    assert instigator is not None
    statement_date = o_stage.find_all("div", {"class": "m-statement__desc"})[0]
    assert statement_date is not None
    instigator_claim_date = re.findall('([a-zA-Z]* [0-9]{1,2}, [0-9]{4})', statement_date.text)[0]
    assert instigator_claim_date is not None
    claim = o_stage.find_all("div", {"class": "m-statement__quote"})[0]
    assert claim is not None
    class_flag = o_stage.find_all("img", {"class": "c-image__original"})
    assert len(class_flag) == 2
    class_flag = class_flag[len(class_flag) - 1].get("alt")
    tags = o_stage.find_all("a", {"class": "c-tag"})
    assert len(tags) > 0
    tags = ','.join(tag.get("title") for tag in tags)
    
    # Special Edge case. Article is in English but Date is in Spanish
    if 'Mayo' in instigator_claim_date:
        instigator_claim_date = instigator_claim_date.replace('Mayo', 'May')
    
    date = datetime.strptime(instigator_claim_date, '%B %d, %Y')
    
    output = {
        'claim_owner': instigator.get("title"),
        'claim_date': instigator_claim_date,
        'year': date.year,
        'month': date.month,
        'day': date.day,
        'claim': claim.get_text(),
        'label': class_flag.lower(),
        'tags': tags
    }
    
    return output 

def extract_if_your_time_is_short(if_your_time_is_short):
    soup = BeautifulSoup(if_your_time_is_short.prettify(), 'html.parser')
    short_on_time = soup.find('div', {"class": 'short-on-time'})
    assert short_on_time is not None
    texts = []
    raw_texts = short_on_time.get_text().strip()
    raw_texts = re.sub('•', '', raw_texts)
    raw_texts = re.sub(r'\s{2,}?', '<kekw>', raw_texts)
    raw_texts = re.split('<kekw>', raw_texts)
    texts = [text for text in raw_texts if len(text) > 0]
    return {
        'html': short_on_time,
        'text': texts if len(texts) > 0 else None
    }

def extract_year_from_url(url):
    url_stubs = re.sub('https://www.politifact.com/', '', url)
    url_year = url_stubs.split('/')[1]
    assert url_year is not None
    return url_year

def remove_unrelated_classes_from_politifact(soup):
    # Remove these classes
    html_classes = [
        {"section": {"class": "o-pick"}},
        {"div": {"class": "m-borderbox__content sms-box"}}, # SMS Box
        {"div": {"class": "artembed"}}, # Infographics
        {"div": {"class": "u-push--bottom"}} # SMS Box
    ]
    for html_class in html_classes:
        dict_key = list(html_class.keys())[0]
        for tag in soup.find_all(dict_key, html_class[dict_key]):
            tag.decompose()

    for data in soup(['style', 'script']):
        # Remove tags
        data.decompose()
    
    return soup

def extract_main_article_title_from_politifact(c_title):
    title = c_title.get_text()
    assert title is not None
    return {
        'html': c_title,
        'text': title.strip()
    }
    
def extract_main_article_from_politifact(m_textblock):
    output = []
    # New Articles use this format
    if m_textblock.html is not None:
        assert m_textblock.html.body is not None
        for html_element in m_textblock.html.body:
            text = html_element.get_text().strip()
            if len(text) > 0:
                output.append(text)
    # Older Articles use this format
    elif m_textblock.div is not None:
        assert m_textblock.div is not None
        for html_element in m_textblock:
            text = html_element.get_text().strip()
            if len(text) > 0:
                output.append(text)

    if len(output) == 0:
        output = None
    return {
        'html': m_textblock,
        'text': output
    }

def extract_our_sources_from_politifact(our_sources):
    soup = BeautifulSoup(our_sources.prettify(), 'html.parser')
    paragraphs = soup.find_all('p')
    assert paragraphs is not None
    outputs = []
    for paragraph in paragraphs:
        reference_text = re.sub(r'\n', '', paragraph.get_text().strip())
        reference_text = re.sub(r'\s{2,}?', '', reference_text)
        output = None
        if paragraph.a is not None:
            output = {
                'text': reference_text,
                'url': paragraph.a.get('href')
            }
        else:
            output = {
                'text': reference_text,
                'url': None
            }
        outputs.append(output)
    return {
        'html': paragraphs,
        'text': outputs
    }

def extract_details_from_politifact(args):
    url, html_file_location, save_path = args['url'], args['html_file_location'], args['save_path']
    try:
        file = None
        logging.info(f'Reading {url} html from {html_file_location}')
        with open(html_file_location, 'r') as file:
            html = file.read()
            file.close()
        soup = BeautifulSoup(html, 'html.parser')
        # Remove unrelated classes first
        logging.info(f'Removing unrelated html for {url}')
        soup = remove_unrelated_classes_from_politifact(soup)
        # Get high level details from o_stage
        o_stage = soup.find("section", {"class": "o-stage"})
        assert o_stage is not None
        logging.info(f'Extracting o-stage for {url}')
        o_stage_info = extract_details_from_o_stage(o_stage)
        # Get details from If Your Time is Short
        m_callouts = soup.find_all('div', {"class": "m-callout"})
        assert len(m_callouts) > 0
        if_your_time_is_short = None
        for m_callout in m_callouts:
            for html_element in m_callout:
                if html_element.name == 'h4' and html_element.get_text().strip() == 'If Your Time is short':
                    if_your_time_is_short = m_callout
        if_your_time_is_short_dict = {
            'html': None, 'text': None
        }
        if if_your_time_is_short is not None:
            logging.info(f'Extracting m-callout for {url}')
            if_your_time_is_short_dict = extract_if_your_time_is_short(if_your_time_is_short)
        else:
            logging.info(f'Skip extracting o-stage for {url}')
        # Extract Main Article Title
        c_title = soup.find('h1', {"class": "c-title"})
        assert c_title is not None
        title_dict = extract_main_article_title_from_politifact(c_title)
        # Extract Main Article
        m_textblock = soup.find('article', {"class": "m-textblock"})
        assert m_textblock is not None
        logging.info(f'Extracting main article content for {url}')
        article_dict = extract_main_article_from_politifact(m_textblock)
        # Extract Sources
        our_sources = soup.find('section', {'id': 'sources'})
        assert our_sources is not None
        logging.info(f'Extracting our-sources for {url}')
        sources_dict = extract_our_sources_from_politifact(our_sources)
        logging.info(f'Processing done for {url}')

        # Save Filtered HTMl texts
        url_stubs = re.sub('https://www.politifact.com/', '', url)
        FILE_NAME = '-'.join(url_stubs[:-1].split('/'))
        TITLE_FILE_PATH = os.path.join(save_path,f'{FILE_NAME}_title.html')
        with open(TITLE_FILE_PATH, 'w') as title_file:
            title_file.write(f"{title_dict['html']}")
        title_file.close()
        ARTICLE_FILE_PATH = os.path.join(save_path,f'{FILE_NAME}_article.html')
        with open(ARTICLE_FILE_PATH, 'w') as article_file:
            article_file.write(f"{article_dict['html']}")
        article_file.close()
        SOURCES_FILE_PATH = os.path.join(save_path,f'{FILE_NAME}_sources.html')
        with open(SOURCES_FILE_PATH, 'w') as sources_file:
            sources_file.write(f"{sources_dict['html']}")
        sources_file.close()

        json_text_output = {
            'url': url,
            **o_stage_info,
            'if_your_time_is_short': None, 
            'if_your_time_is_short_html_path': None,
            'main_article_title': title_dict['text'],
            'main_article_title_html_path': TITLE_FILE_PATH,
            'main_article': article_dict['text'],
            'main_article_html_path': ARTICLE_FILE_PATH,
            'our_sources': sources_dict['text'],
            'our_sources_html_path': SOURCES_FILE_PATH
        }

        TIME_IS_SHORT_FILE_PATH = os.path.join(save_path, f'{FILE_NAME}_time_is_short.html')
        if if_your_time_is_short_dict['html'] is not None:
            with open(TIME_IS_SHORT_FILE_PATH, 'w') as time_is_short_file:
                time_is_short_file.write(f"{if_your_time_is_short_dict['html']}")
            time_is_short_file.close()
            json_text_output['if_your_time_is_short'] = if_your_time_is_short_dict['text']
            json_text_output['if_your_time_is_short_html_path'] = TIME_IS_SHORT_FILE_PATH

        return json_text_output
    
    except Exception as e:
        raise ValueError([url, e])
        
def extract_articles_from_html(args):
    SAVE_PATH = args.save_path
    ARTICLE_RAW_HTML_FOLDER = args.article_raw_html_folder
    ARTICLE_RAW_HTML_PATH = os.path.join(ARTICLE_RAW_HTML_FOLDER, 'politifact_raw_articles.csv')
    SAVE_FILE_PATH = os.path.join(SAVE_PATH, 'politifact_extracted_articles.json')

    paths = [SAVE_PATH]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(SAVE_PATH)

    # Save current args to output
    with open(f"{SAVE_PATH}/argsparse_config.json", 'w') as file:
        json.dump(vars(args), file)
        file.close()
    
    # Blacklisted URL
    blacklisted_urls = [
        'https://www.politifact.com/factchecks/2024/dec/02/facebook-posts/no-elon-musk-no-demando-a-the-view-y-whoopi-goldbe/'
    ]

    # Retrieve Article HTMLs
    raw_article_details = []
    with open(ARTICLE_RAW_HTML_PATH) as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for index, item in enumerate(reader):
            if index > 0 and item[0] not in blacklisted_urls:
                raw_article_details.append({
                    'url': item[0],
                    'html_file_location': item[1],
                    'save_path': SAVE_PATH
                })
        file.close()

    
    with Pool() as pool:
        outputs = pool.map(extract_details_from_politifact, raw_article_details)
        logging.info('Writing output to JSON')
        df = pd.DataFrame(outputs)
        df.to_json(SAVE_FILE_PATH, orient='records')
        
if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    parser = argparse.ArgumentParser(description='Politifact Link Extractor. Extract Personalities and Article Links from targeted URLs')
    parser.add_argument('--save_path', type=str, default=os.path.join('script_outputs', 'politifact-extracted-articles'), help='Script output location')
    parser.add_argument('--article_raw_html_folder', type=str, default=os.path.join('script_outputs', 'politifact-raw-article-html'), help='politifact_articles_links.csv location')

    args = parser.parse_args()
    extract_articles_from_html(args)