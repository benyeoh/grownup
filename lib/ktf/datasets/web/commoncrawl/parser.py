import gzip
import zlib
import json
import csv
import os
from json.decoder import JSONDecodeError
from urllib.parse import urlparse

import hashlib
import regex
import tldextract
from warcio.archiveiterator import ArchiveIterator

from . import filters
from . import constants


class ParserCDX:
    """Parses raw CommonCrawl dataset described here: https://commoncrawl.org/the-data/get-started/
    The objective is to process raw input (cdx.gz) to get intermediate compressed web archive \n
    (in .arc or .warc format) in bulk.

    The requirement for output
    1. Informative content, in English
    2. Traverse all qualified domains and for each domain, keep 0-4 subdomains.
    3. For every `subdomain-domain` range, keep 2 .html pages with highest similarity.
    * Similarity is measured on the path (read from left to right): identical string between every pair of '/'.

    The parser looks through individual raw input (from 2008 - 2020) and:
    1. Apply filters from bucket on the fly
    2a. Group by domains and subdomains, select the pairs with enough content
    2b. Sorted the path by layers(number of '/') and similarity
    3. Form list of urls based on (2) and proceed with .arc/.warc fetching

    Glossary [example]: www.google.com/jp/abc-def/12345/demo.html
    1. top-level-domain(TLD): .com
    2. domain: google
    3. subdomain: www
    4. country-code-TLD (ccTLD): jp
    5. path: /jp/abc-def/12345/demo.html
    6. netloc: www.google.com
    7. url: www.google.com/jp/abc-def/12345/demo.html
    """

    def __init__(self, cdx_path):
        """Initializer
            Args:
                cdx_path: Path to the raw cdx.gz. Action: read
        """
        self._cdx_path = cdx_path

        # Nested dict to save raw web crawl infomation
        # Format: {cleaned_url: {'digest': digest, 'address': address, 'group': netloc-based-group-name}}
        self._web_crawl_raw_dict = {}
        # Nested dict to save grouped domain and subdomain. Format: {domain: {subdomain: [paths]}}
        self._domain_dict = {}

    def _read_cdx(self, line):
        """
        Parse index JSON in cdx block. Convert byte obj to dictionary.
        """
        line = line.decode('utf-8', 'ignore')
        pattern = regex.compile(r'\{(?:[^{}]|(?R))*\}')
        line_json = pattern.findall(line)
        line_dict = json.loads(line_json[-1])  # -1 used to prevent if more than 1 JSON returned

        return line_dict

    def _local_filter(self, subdomain, domain, tld, path, length):
        """
        Filters of basic page info and url. Apply on the fly.
        Remove: non-html and short pages, not from assigned TLD, unwanted country code pages and
        messy domain / subdomain (names)
        """

        rules = [filters.verify_format(path),
                 filters.verify_length(length, ref_length=constants.MIN_PAGE_LENGTH),
                 filters.verify_tld(tld, ref_tld=constants.SELECTED_TLD),
                 filters.verify_path_cctld(path, ref_cctld_lst=constants.COUNTRY_CODE_TO_REMOVE),
                 filters.verify_domain_length(domain, ref_domain_length=constants.MAX_DOMAIN_LENGTH),
                 filters.verify_domain_pattern(domain),
                 filters.verify_subdomain_length(subdomain, ref_subdomain_length=constants.MAX_DOMAIN_LENGTH)
                 ]

        return all(rules)

    def _global_filter(self, web_crawl_raw_dict, domain_dict):
        """
        Filters on the nested dict of grouped domain and subdomain. Apply after consolidate all info from each cdx
        """

        # Sampling paths
        reduced_domain_dict = self._remove_duplicate_path(domain_dict)
        selected_domain_dict = self._reduce_subdomain_qty(reduced_domain_dict)
        updated_domain_dict = self._reduce_path_qty(selected_domain_dict)

        # Generate download_dict for download
        web_crawl_download_dict = self._get_web_crawl_path(web_crawl_raw_dict, updated_domain_dict)

        return web_crawl_download_dict

    def _get_cdx_components(self, line_dict):
        """
        Fetch values from CDX JSON index that describes url and address to download web crawl file
        """
        length = line_dict['length']
        filename = line_dict['filename']
        offset = line_dict['offset']
        # Concatenate parameters to form a str containing info to download web crawl file
        address = '-r ' + str(offset) + '-' + str(int(offset) + int(length) - 1) + ' ' + \
            '"https://commoncrawl.s3.amazonaws.com/' + filename + '"'

        url = urlparse(line_dict['url'])
        path = url.path
        subdomain, domain, tld = tldextract.extract(url.netloc)

        return address, subdomain, domain, tld, path, length

    def _get_web_crawl_component(self, path, subdomain, domain, tld):
        """
        Prepare info needed to name and group web crawl that will be downloaded.
        Also return web crawl format (in ARC or WARC) for subsequent web crawl parser
        """

        # Example cdx_path: path_to_cdx/raw/CC-MAIN-2008-2009/indexes/cdx_id.gz, aim to get '2008'
        cur_year = (self._cdx_path.split('/')[-3]).split('-')[2]
        cur_cdx_filename = os.path.splitext(self._cdx_path.split('/')[-1])[0]
        group_name = subdomain + '.' + domain + '.' + tld + '_' + cur_year + '_' + cur_cdx_filename
        '''
        Note: cur_year + cur_cdx_filename helps to:
        (1) label the resource
        (2) make it unique if the same webpage been crawled in another year
        '''
        cleaned_url = subdomain + '.' + domain + '.' + tld + path

        # Determine the web_crawl_format from cur_year. According to CommonCrawl announcement:
        # If before 2012, it is ARC, afterwards are WARC
        web_crawl_format = 'ARC' if int(cur_year) <= 2012 else 'WARC'

        # Apply MD5 to convert url into digest
        md5 = hashlib.md5()
        md5.update(cleaned_url.encode('utf-8'))
        digest = md5.hexdigest()

        return group_name, cleaned_url, digest, web_crawl_format

    def _add_domain_dict(self, domain, subdomain, tld, path):
        """
        Create and update dict that groups domain and its subdomains. value is list of paths
        """
        domain_full = domain + '.' + tld

        if domain_full not in self._domain_dict.keys():
            self._domain_dict[domain_full] = {subdomain: [path]}
        elif subdomain not in self. _domain_dict[domain_full].keys():
            self._domain_dict[domain_full][subdomain] = [path]
        elif subdomain in self._domain_dict[domain_full].keys():
            self._domain_dict[domain_full][subdomain].append(path)

    def _remove_duplicate_path(self, domain_dict):
        """
        Apply set() to remove duplicate path from raw domain dict.
        """
        reduced_domain_dict = {}

        for domain_full in domain_dict.keys():
            reduced_domain_dict[domain_full] = {}
            for subdomain, paths in domain_dict[domain_full].items():
                reduced_domain_dict[domain_full][subdomain] = list(set(paths))

        return reduced_domain_dict

    def _reduce_subdomain_qty(self, reduced_domain_dict):
        """
        Prepare a dict saving limited pairs of subdomain-domain
        """
        selected_domain_dict = {}

        for domain_full in reduced_domain_dict.keys():
            count_cutoff = 0  # Count for subdomain to not exceed expected value (max_subdomain_count)
            if domain_full not in selected_domain_dict.keys():
                selected_domain_dict[domain_full] = {}

            for subdomain, lst_path in reduced_domain_dict[domain_full].items():
                # Add subdomain to new dict if page_count > 2 and stop adding if exceeding max_subdomain_count
                if len(lst_path) > constants.MIN_PAGE_COUNT \
                        and count_cutoff < constants.MAX_SUBDOMAIN_COUNT:
                    count_cutoff += 1
                    selected_domain_dict[domain_full][subdomain] = lst_path
                else:
                    continue

        return selected_domain_dict

    def _reduce_path_qty(self, nested_dict):
        """
        Further reduce the dict by saving only 2 path with most '/' and hightest similarity
        """
        updated_domain_dict = {}

        for domain_full in nested_dict.keys():
            updated_domain_dict[domain_full] = {}

            for subdomain, lst_path in nested_dict[domain_full].items():

                # Filter only path count >= 2
                if subdomain and (len(lst_path) >= constants.MIN_PAGE_COUNT):

                    # Sorted by count_slash
                    lst_path = sorted(lst_path, key=lambda lst_path: (lst_path.count('/'), lst_path), reverse=True)

                    # Initiate path list
                    reduced_lst_path = lst_path[:constants.MIN_PAGE_COUNT]
                    last_elem = []

                    # Update by similarity, search two paths with identical substrings except page name (after last '/')
                    # First find out the substr and starting point
                    idx_start = 0
                    for i in range(len(lst_path)):
                        substr = lst_path[i].replace(lst_path[i].split('/')[-1], '')
                        if sum(substr in elem for elem in lst_path) >= constants.MIN_PAGE_COUNT:
                            idx_start = i
                            break

                    # Next form the list of last element, update list of path if fulfilling the requirement
                    for i in range(idx_start, len(lst_path)):
                        if substr == lst_path[i].replace(lst_path[i].split('/')[-1], ''):
                            last_elem.append(lst_path[i].split('/')[-1])

                    if len(last_elem) >= constants.MIN_PAGE_COUNT:
                        reduced_lst_path = []
                        for elem in sorted(last_elem)[:constants.MIN_PAGE_COUNT]:
                            reduced_lst_path.append(substr + elem)

                    updated_domain_dict[domain_full][subdomain] = reduced_lst_path
                else:
                    continue

        return updated_domain_dict

    def _get_web_crawl_path(self, web_crawl_raw_dict, updated_domain_dict):
        """
        Generate dict with arc to download. Corresponding url are derived after applying global filter(s)
        """
        web_crawl_download_dict = {}  # Format {cleaned_url: {'digest': digest, 'address': address, 'group':  group}}

        for domain_full in updated_domain_dict.keys():
            for subdomain, paths in updated_domain_dict[domain_full].items():
                for path in paths:
                    lst = subdomain + '.' + domain_full + path
                    web_crawl_download_dict[lst] = web_crawl_raw_dict[lst]

        return web_crawl_download_dict

    def parse(self, max_indices, apply_filters):
        """
        Parse raw CDX file, to get ARC information after apply filters.
        """
        # Used in partial scan with max_indice step, also used in exception handling msg to show the progress
        count = 0

        with gzip.open(self._cdx_path, 'rb') as stream:
            try:
                for line in stream:
                    try:
                        line_dict = self._read_cdx(line)

                        # Only (1)'mime' tag = 'text/html', and (2) containing MUST HAVE tags
                        if ((line_dict['mime'] == 'text/html') and (set(constants.CDX_TAGS) <= set(line_dict.keys()))):
                            address, subdomain, domain, tld, path, length = self._get_cdx_components(line_dict)

                            # Check all Tags can be decoded: value not None
                            if None not in [address, domain, tld, path, length]:
                                # Subdomain is allowed to be empty
                                group_name, cleaned_url, digest, web_crawl_format = self._get_web_crawl_component(
                                    path, subdomain, domain, tld)
                                is_filtered = apply_filters and not self._local_filter(
                                    subdomain, domain, tld, path, length)

                                # Add index into dicts when
                                # (1) Pass filters with apply-filters is True; or (2) apply filters is False
                                if ((not is_filtered) and (cleaned_url not in self._web_crawl_raw_dict.keys())):
                                    self._web_crawl_raw_dict[cleaned_url] = {
                                        'digest': digest, 'address': address, 'group': group_name}
                                    # Format: {domain: {subdomain: [path1, path2, ....]}}
                                    self._add_domain_dict(domain, subdomain, tld, path)

                            count += 1
                            if max_indices is not None and count >= max_indices:
                                break

                    except (JSONDecodeError) as e:
                        print('Json decode failed on {}'.format(count))
                        '''
                        Count is helped in showing the parsing CDX is running. It does not refer to the number of JSON.
                        Hence, if multiple same number printed. It means some JSON didnt pass local filter,
                        Then the counts remains
                        '''
                        pass

                    except (KeyError) as e:
                        print('Key mime could not find on {}'.format(count))
                        pass

                    except Exception:
                        print()
                        print('Parsing CDX was interrupted.')
                        raise

            except (zlib.error) as e:
                print('Error -3 while decompressing data: invalid block type on {}'.format(count))
                pass

        print(' - There are {} web crawl files to be indexed.'.format(len(self._web_crawl_raw_dict)))
        web_crawl_download_dict = self._global_filter(self._web_crawl_raw_dict, self._domain_dict)
        print(' - There are {} web crawl files to be downloaded.'.format(len(web_crawl_download_dict)))

        return web_crawl_download_dict, web_crawl_format


class ParserArchive:
    """Parse web crawl files(format ARC or WARC) to extract HTML content with (optional) filters applied."""

    def __init__(self, web_crawl_dir, web_crawl_format):
        """Initializer
        Args:
            web_crawl_dir: Directory saving .arc or .warc which are downloaded from AWS S3
            format: ARC or WARC, derived from CDX parser and determined by the year of crawling by Common Crawl team
        """
        self._web_crawl_dir = web_crawl_dir
        self._format = web_crawl_format

    def _get_content_type(self, record):
        """content_type are saved with different tag for ARC and WARC"""
        if self._format == 'ARC':
            return record.rec_headers.get_header('content-type')
        elif self._format == 'WARC':
            return record.http_headers.get_header('content-type')

    def _extract_html(self, content):
        """
        Extract HTML content labeled by declaration tag. This is to avoid XML parsing error, which is not starting
        from '<!DOCTYPE '
        """
        # TODO - a bit hardcoded, a smarter way is expected.
        if content.find('<!DOCTYPE') != -1:
            start = content.index('<!DOCTYPE')
        else:
            start = 0
        end = content.index('</html>')

        return content[start: end]

    def parse(self, apply_filters):
        """
        Parse web crawl files, to get HTML content after applying filters
        """
        for web_crawl_filename in sorted(os.listdir(self._web_crawl_dir)):
            # Iterate every compressed web crawl file
            if web_crawl_filename.endswith('.gz'):
                web_crawl_path = os.path.join(self._web_crawl_dir, web_crawl_filename)

                # Streaming every web crawl file
                with open(web_crawl_path, 'rb') as stream:
                    for record in ArchiveIterator(stream):
                        if record.rec_type == 'response':
                            content = record.content_stream().read()  # Content is bytes object
                            content = content.decode('utf-8', 'ignore')

                            if ((self._get_content_type(record) == 'text/html') and (content.find('</html>') != -1)):
                                html_str = self._extract_html(content)
                                html_id = str(web_crawl_filename[: -3])

                                # Apply language and content length filters
                                if apply_filters:
                                    if not ((filters.filter_lang(html_str)) and
                                            (len(html_str) > constants.MIN_CONTENT_LENGTH)):
                                        continue
                                yield html_str, html_id
                        else:
                            print('Broken HTTP requests on page {}'.format(web_crawl_filename))
