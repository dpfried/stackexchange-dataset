import sys
import csv
import tqdm
import xml.etree.ElementTree as etree
from collections import defaultdict, Counter
from bs4 import BeautifulSoup, PageElement

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

from pairer import CodePreservingBeautifulSoup

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_xml')
    args = parser.parse_args()

    if "Comments" in args.input_xml:
        # 11K it/s
        columns = ["Id", "PostId", "Score", "Text", "CreationDate", "UserId", "ContentLicense"]
        cols_to_html_parse = {"Text"}
        cols_to_code_preserve_html_parse = set()
        num_rows = 82_037_744 - 3
    elif "Posts" in args.input_xml:
        # 2.7K it/s
        columns = [
            # question fields
            "Id", "PostTypeId", "AcceptedAnswerId", "CreationDate", "Score", "ViewCount", "Body", "OwnerUserId", "LastEditorUserId", "LastEditorDisplayName", "LastEditDate", "LastActivityDate", "Title", "Tags", "AnswerCount", "CommentCount", "FavoriteCount", "CommunityOwnedDate", "ContentLicense",
            # answer fields (that aren't already in the list above)
            "ParentId",
            ]
        cols_to_html_parse = {"Body"}
        cols_to_code_preserve_html_parse = {"Title"}
        num_rows = 59_949_888 - 3

    def make_parsed_key(col):
        return f'{col}Parsed'

    for col in cols_to_html_parse | cols_to_code_preserve_html_parse:
        columns.append(make_parsed_key(col))

    writer = csv.DictWriter(sys.stdout, columns)
    writer.writeheader()

    for event, elem in tqdm.tqdm(etree.iterparse(args.input_xml, events=('end',)), ncols=80, total=num_rows):
        attribs = defaultdict(lambda: None, {k: v for k, v in elem.attrib.items() if k in columns})
        for col in cols_to_html_parse:
            if attribs[col] != None:
                text = attribs[col]
                try:
                    attribs[make_parsed_key(col)] = BeautifulSoup(text, "html.parser").get_text()
                except Exception as e:
                    print(e)
                    elem.clear()
                    continue
        for col in cols_to_code_preserve_html_parse:
            if attribs[col] != None:
                text = attribs[col]
                try:
                    attribs[make_parsed_key(col)] = CodePreservingBeautifulSoup(text, "html.parser").get_text()
                except Exception as e:
                    print(e)
                    elem.clear()
                    continue
        try:
            writer.writerows([attribs])
        except Exception as e:
            print(e)
        elem.clear()
