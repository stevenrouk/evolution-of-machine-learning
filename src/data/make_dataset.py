import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import re
import time

import requests

SCRIPT_DIRECTORY = os.path.split(os.path.realpath(__file__))[0]
SRC_DIRECTORY = os.path.split(SCRIPT_DIRECTORY)[0]
ROOT_DIRECTORY = os.path.split(SRC_DIRECTORY)[0]
DATA_DIRECTORY_RAW = os.path.join(ROOT_DIRECTORY, 'data', 'raw')
QUEUE_FILEPATH = os.path.join(SCRIPT_DIRECTORY, 'queue.txt')
BASE_URL = 'http://export.arxiv.org/oai2'

def build_url(verb=None, metadata_prefix=None, set_spec=None, resumption_token=None):
    url = [BASE_URL, '?']
    if verb:
        url.extend(['verb=', verb, '&'])
    if metadata_prefix:
        url.extend(['metadataPrefix=', metadata_prefix, '&'])
    if set_spec:
        url.extend(['set=', set_spec, '&'])
    if resumption_token:
        url.extend(['resumptionToken=', resumption_token, '&'])

    if url[-1] == '&':
        url.pop()
    if url[-1] == '?':
        url.pop()
    
    return ''.join(url)

def build_first_call_url(set_spec=None):
    """Format: http://export.arxiv.org/oai2?verb=ListRecords&set=stat&metadataPrefix=oai_dc"""
    return build_url(verb='ListRecords', metadata_prefix='oai_dc', set_spec=set_spec)

def build_resumption_url(resumption_token):
    """Format: http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=3902257|1001"""
    return build_url(verb='ListRecords', resumption_token=resumption_token)

def get_resumption_token(response_text):
    # resumption token substring
    resump_text = response_text[response_text.find('resumptionToken'):]

    return re.search(r'>(\d+\|\d+)<', resump_text).groups()[0]

def make_name_file_safe(filename):
    return "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()


if __name__ == "__main__":
    # # Get first API request and write to file
    # print('Getting first url')
    # url = build_first_call_url()
    # output_filename = make_name_file_safe(f'FIRST') + '.xml'
    # output_filepath = os.path.join(DATA_DIRECTORY_RAW, output_filename)
    # r = requests.get(url)
    # with open(output_filepath, 'w', encoding='utf-8') as f:
    #     f.write(r.text)

    # # Get resumption token
    # resumption_token = get_resumption_token(r.text)

    resumption_token = '3923570|591001'

    # while we have a resumption token, get API request and write to file
    # if we need to wait, wait
    try:
        fail_count = 0
        while resumption_token:
            if int(resumption_token.split('|')[1]) > 1700001:
                # Going to cut it here just since I don't know what happens
                # at the last API pull
                print(f'breaking the script at {resumption_token}')
                break
            print(f'trying {resumption_token}')
            url = build_resumption_url(resumption_token)
            output_filename = make_name_file_safe(f'{resumption_token}') + '.xml'
            output_filepath = os.path.join(DATA_DIRECTORY_RAW, output_filename)
            r = requests.get(url)
            if r.status_code == 200:
                with open(output_filepath, 'w', encoding='utf-8') as f:
                    f.write(r.text)
                print('success')
                time.sleep(3)
                resumption_token = get_resumption_token(r.text)
                fail_count = 0
            elif r.status_code == 503:
                fail_count += 1
                if fail_count < 3:
                    print('503 response -- waiting 7 seconds')
                    time.sleep(7)
                elif fail_count < 7:
                    print('503 response -- waiting 60 seconds')
                    time.sleep(60)
                elif fail_count < 10:
                    print('503 response -- waiting 600 seconds')
                    time.sleep(600)
                elif fail_count < 20:
                    print('503 response -- waiting 6000 seconds')
                    time.sleep(6000)
                else:
                    print(f'Fail count is at {fail_count} -- stopping execution at token {resumption_token}')
    except Exception as e:
        # if anything fails, write the resumption token that failed to a file
        print(e)
        with open('failed_token.txt', 'a') as f:
            f.write(resumption_token)
    
    import pdb; pdb.set_trace()
