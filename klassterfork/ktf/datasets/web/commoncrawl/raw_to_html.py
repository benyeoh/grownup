#!/usr/bin/env python
import argparse
import csv
import os
import glob
import sys
import stat
import shlex
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../..'))

import magic

import ktf.datasets.web.commoncrawl
from ktf.datasets.web.commoncrawl.parser import ParserCDX, ParserArchive
from ktf.datasets.web.commoncrawl import filters


def download_web_crawl(web_crawl_download_dict, web_crawl_dir):
    """
    Form a HTTP range request to download web crawl files from AWS.
    """
    # CURL mode: silent (-s), show error (-S), fail silently on HTTP error (--fail), output (-output)
    curl_prefix = 'curl -sS --fail '
    curl_postfix = ' --output '

    st = os.stat(web_crawl_dir)
    # Set the updated permission: execute by owner, group and others
    os.chmod(web_crawl_dir, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Download web crawl files with exception handle and completion check
    for url in sorted(web_crawl_download_dict.keys()):
        web_crawl_filename = web_crawl_download_dict[url]['group'] + \
            '_' + web_crawl_download_dict[url]['digest'] + '.gz'
        if web_crawl_filename not in sorted(os.listdir(web_crawl_dir)):

            # Prepare arg for subprocess
            web_crawl_path_tmp = os.path.join(web_crawl_dir, web_crawl_download_dict[url]['digest'] + '.gz.tmp')
            cmd = curl_prefix + web_crawl_download_dict[url]['address'] + curl_postfix + web_crawl_path_tmp
            args = shlex.split(cmd)

            # Retry after exceptions. Tentative attempts 3
            count = 0
            while count < 3:
                try:
                    process = subprocess.check_call(args, shell=False)

                    if process == 0:  # Complete download
                        web_crawl_path = os.path.join(
                            web_crawl_dir, web_crawl_download_dict[url]['group'] + '_' +
                            web_crawl_download_dict[url]['digest'] + '.gz')
                        os.rename(web_crawl_path_tmp, web_crawl_path)

                        # Status update
                        if len(os.listdir(web_crawl_dir)) % 500 == 0:
                            print('Downloaded {} web crawl files.'.format(len(os.listdir(web_crawl_dir))))
                    break

                except (subprocess.CalledProcessError) as e:  # Not able to connect or reconnect
                    if count == 2:
                        print('Connection issues during download. Retry but failed after {} times.'.format(count + 1))
                        raise
                    count += 1
                except (KeyboardInterrupt) as e:
                    print('Download web crawl files was interrupted.')
                    print()
                    raise


def web_crawl_to_html(web_crawl_parser, html_dir, apply_filters):
    """
    Process web crawl files to parse HTML content and then save HTML pages
    """
    st = os.stat(html_dir)
    # Set the updated permission: execute by owner, group and others
    os.chmod(html_dir, st.st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # Initializer
    id_to_html = {}
    group_to_ids = {}

    # Parse web crawl files and save into HTML
    try:
        for html_str, id in web_crawl_parser.parse(apply_filters):

            # Only process if HTML page was not existed and confirmed accessible (with file type check)
            html_filename = id + '.html'
            if ((html_filename not in sorted(os.listdir(html_dir))) and
                    (magic.from_buffer(html_str)[:4] == 'HTML')):
                id_to_html[id] = html_str
                group = id.replace(id.split('_')[-1], '')

                if group not in group_to_ids:
                    group_to_ids[group] = [id]
                else:
                    group_to_ids[group].append(id)

                if len(group_to_ids[group]) >= 2:
                    # Write to files
                    if len(group_to_ids[group]) == 2:
                        for cur_id in group_to_ids[group]:
                            # Write both html content in the group
                            # (There should be only 2 at this point)
                            write_str_to_html(html_dir, cur_id, id_to_html[cur_id])
                            id_to_html.pop(cur_id, {})

                    else:
                        # Write all html content in the group
                        cur_id = group_to_ids[group][len(group_to_ids[group]) - 1]
                        write_str_to_html(html_dir, cur_id, id_to_html[cur_id])
                        id_to_html.pop(cur_id, {})

    except(KeyboardInterrupt, ConnectionError) as e:
        print('Conversion of HTML was interrupted.')
        print()
        raise


def reorg_dict(web_crawl_download_dict, html_dir):
    """
    From HTML filename fetch URL, to prep the output csv with URL, filename and group name
    """
    cleaned_html_dict = {}

    for url in sorted(web_crawl_download_dict.keys()):
        group_name = web_crawl_download_dict[url]['group']
        html_name = group_name + '_' + web_crawl_download_dict[url]['digest'] + '.html'
        html_path = os.path.join(html_dir, html_name)
        if html_path in sorted(glob.glob(os.path.join(html_dir, '*.html'))):
            html_name = html_path.split('/')[-1]
            cleaned_html_dict[html_name] = {url: group_name}

    return cleaned_html_dict


def get_cdx_lst(all_orig, num_task, id_seq):
    """
    Calculate CDX list based on number of total task and task instance id
    """
    num_task = int(num_task)
    id_seq = int(id_seq)

    # At least processing 1 CDX per task
    if ((num_task > 0 and num_task <= len(all_orig)) and
            (id_seq <= num_task and id_seq > 0)):

        avg = len(all_orig) / float(num_task)
        orig_lst = []
        last = 0.0

        while last < len(all_orig):
            orig_lst.append(sorted(all_orig)[int(last): int(last + avg)])
            last += avg

        return orig_lst[id_seq - 1]

    elif num_task > len(all_orig) or num_task < 0:
        raise ValueError('Incorrect value input of {}.'.format(num_task))
    elif id_seq > num_task or id_seq <= 0:
        raise ValueError('Incorrect value input of {}.'.format(id_seq))


def write_nested_dict_to_csv(path, dict):
    with open(path, 'w') as fd:
        content_writer = csv.writer(fd, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for session in dict:
            for key in dict[session]:
                content_writer.writerow([session, key, dict[session][key]])


def write_str_to_html(html_dir, html_id, content):
    path = os.path.join(html_dir, html_id + '.html')
    with open(path, 'w') as fd:
        fd.write(content)
        fd.close


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description='--Help message for parsing CDX into HTML--')
    arg_parser.add_argument('-i', '--input', dest='input_dir', required=True, metavar='DIR',
                            help='Input directory for CDX from AWS S3')
    arg_parser.add_argument('-o', '--output', dest='output_dir', required=True, metavar='DIR',
                            help='Output directory for HTML and other results')
    arg_parser.add_argument('--apply-filters', action='store_false', default=True,
                            help='Apply filters on URL and content with default True')
    arg_parser.add_argument('--num-task', type=int, default=1, metavar='NUM',
                            help='Number of tasks to parse input CDX with default value 1. '
                            'Choose a value less than the number of CDX in input directory.')
    arg_parser.add_argument('--id-seq', type=int, default=1, metavar='NUM',
                            help='Instances for current task with a default value 1. Choose a value less than num-task')
    arg_parser.add_argument('--max-indices', default=None, metavar='NUM',
                            help='Process entire CDX (default) or a partial scan with input steps')

    (opt_args, args) = arg_parser.parse_known_args()

    if len(args) > 0:  # If additional arguments are given
        print('Error with additional arguments provided')
        arg_parser.print_help()
        exit()

    # Concatenate all raw input cdx
    all_orig = glob.glob(os.path.join(opt_args.input_dir, 'indexes', 'cdx-' + ('[0-9]' * 5) + '.gz'))
    # Compute the list of raw CDX to process in the current task
    cur_orig_lst = get_cdx_lst(all_orig, opt_args.num_task, opt_args.id_seq)

    # Create output_dir if not exist
    os.makedirs(opt_args.output_dir, exist_ok=True)
    # Load processed_cdx_lst from output dir. Skip files not with ext. '.csv'
    processed_cdx_lst = list(filename[:-4] for filename in sorted(os.listdir(opt_args.output_dir))
                             if filename.endswith('csv'))

    for path in sorted(cur_orig_lst):
        cur_cdx_filename = os.path.basename(path).split('.')[0]

        if cur_cdx_filename not in processed_cdx_lst:

            # Set path
            cdx_path = path
            max_indices = opt_args.max_indices if opt_args.max_indices is None else int(opt_args.max_indices)
            # Should change to web_crawl. However, to keep consistent folder structure and avoid duplicate downloading
            # of web crawl files, in this scripts keep folder name arc and will apply changes in a batch after
            # task complete.
            web_crawl_dir = os.path.join(opt_args.output_dir, cur_cdx_filename, 'arc')
            html_dir = os.path.join(opt_args.output_dir, cur_cdx_filename)
            for path in [web_crawl_dir, html_dir]:
                os.makedirs(path, exist_ok=True)

            print()
            print('Processing: ' + cur_cdx_filename)
            print(('- With {} step'.format(max_indices)) if max_indices is not None else 'With all indexes.')
            cdx_parser = ParserCDX(cdx_path)
            web_crawl_download_dict, web_crawl_format = cdx_parser.parse(max_indices, opt_args.apply_filters)

            print()
            if (len(os.listdir(web_crawl_dir)) == 0):
                print('Downloading web_crawl files of: ' + cur_cdx_filename)
            else:
                print('Resuming downloading of: ' + cur_cdx_filename)
            download_web_crawl(web_crawl_download_dict, web_crawl_dir)

            print()
            print('Converting HTML pages of: ' + cur_cdx_filename)
            web_crawl_parser = ParserArchive(web_crawl_dir, web_crawl_format)
            web_crawl_to_html(web_crawl_parser, html_dir, opt_args.apply_filters)

            # Get URL of each page and save in dict with filename and group name
            cleaned_html_dict = reorg_dict(web_crawl_download_dict, html_dir)
            print(' - Rebuilt {} HTML pages.'.format(len(cleaned_html_dict)))

            processed_cdx_path = os.path.join(opt_args.output_dir, cur_cdx_filename + '.csv')
            write_nested_dict_to_csv(processed_cdx_path, cleaned_html_dict)
            print()
            print('Completed on {}'.format(cur_cdx_filename))
            print()

        else:
            print('{} has been processed.'.format(cur_cdx_filename))
