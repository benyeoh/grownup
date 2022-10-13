import os
from bs4 import BeautifulSoup
from langdetect import detect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException


def verify_format(path):
    return ((path[-4:] == 'html') or (path[-3:] == 'htm'))


def verify_length(length, ref_length):
    return int(length, base=10) > ref_length


def verify_tld(tld, ref_tld):
    return tld in set(ref_tld)


def verify_domain_length(domain, ref_domain_length):
    return len(domain) < ref_domain_length


def verify_domain_pattern(domain):
    # return true if no '-' found
    return domain.find('-') == -1


def verify_subdomain_length(subdomain, ref_subdomain_length):
    return len(subdomain) < ref_subdomain_length


def verify_path_pattern(path):
    # Return True if path not None and no ',' found
    return path.find(',') == -1


def verify_path_cctld(path, ref_cctld_lst):
    lst_path = path.split('/')
    for element in lst_path:
        if element.lower() in ref_cctld_lst or element in ref_cctld_lst:
            return False
    return True


def filter_lang(content, lang=['en']):
    """
    Access the language of body content. Return True if it is English
    Utilize the langdetect package described here: https://pypi.org/project/langdetect/
    """
    # Languagedetect is non-determinstic, introduce seed = 0 can produce same result for same string everytime.
    DetectorFactory.seed = 0

    try:
        soup = BeautifulSoup(content, 'html.parser')
        # [s.decompose() for s in soup('script')]  # Remove <script> elements.
        # Note: Commoncrawl archives only the HTML without the page dependencies (images, videos, JavaScripts and CSS).

        if soup.body:
            body_text = soup.body.get_text()
            try:
                return detect(body_text) in lang
            except LangDetectException:
                return False

    # If UnboundLocalError captured, content cant be parsered hence wont pass language filter
    except UnboundLocalError as e:
        print('UnboundLocalError captured.')
        return False

    # If TypeError captured
    except TypeError as e:
        print('TypeError captured. Traverse None or pass None to multiple variables.')
        return False
