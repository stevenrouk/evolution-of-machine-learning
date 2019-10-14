import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import requests

from src.data.arxiv_sets import ARXIV_SETS

BASE_URL = 'http://export.arxiv.org/oai2'

def build_url(verb=None, metadata_prefix=None, set_spec=None, resumption_token=None):
    url = [BASE_URL, '?']
    if verb:
        url.extend(['verb=', verb, '&'])
    if metadata_prefix:
        url.extend(['metadata_prefix=', metadata_prefix, '&'])
    if set_spec:
        url.extend(['set=', set_spec, '&'])
    if resumption_token:
        url.extend(['resumption_token=', resumption_token, '&'])

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

if __name__ == "__main__":
    url = build_first_call_url()
    print(url)
    next_url = build_resumption_url(resumption_token='3902257|1001')
    print(next_url)
