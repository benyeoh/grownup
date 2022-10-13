# Fixed parameters: a complete index JSON in each cdx block should contains these tags
CDX_TAGS = ['status', 'url', 'filename', 'length', 'mime', 'offset', 'digest']

# Fixed parameters: country code top-level-domain (ccTLD) used on internationalized domain name (IDN) level.
# Ref: https://en.wikipedia.org/wiki/List_of_Internet_top-level_domains#Country_code_top-level_domains
COUNTRY_CODE_TO_REMOVE = ["ac", "as", "at", "bd", "bg", "br", "ch", "cl", "cn", "de", "dk", "ee", "es",
                          "eu", "fi", "fr", "gr", "gt", "hk", "ht", "hu", "id", "ie", "il", "in", "ir", "is",
                          "it", "jp", "kr", "li", "lk", "lt", "lu", "lv", "my", "no", "nu", "pe", "pl", "pm",
                          "pw", "re", "ro", "rs", "se", "sh", "si", "su", "tf", "th", "tm", "tn", "to", "tr",
                          "tw", "vn", "wf", "ws", "yt", "nz", "be", "ca", "sa"]

# Variables: current local filter only proceed top-level-domain from the list below
SELECTED_TLD = ['edu', 'org', 'com']

MIN_PAGE_COUNT = 2
MIN_PAGE_LENGTH = 5000
MAX_SUBDOMAIN_COUNT = 4
MAX_DOMAIN_LENGTH = 10
MIN_CONTENT_LENGTH = 10000
