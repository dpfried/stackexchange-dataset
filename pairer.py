import re
import traceback
import xml.etree.ElementTree as etree
from collections import defaultdict, Counter
from bs4 import BeautifulSoup, PageElement
from tqdm import tqdm
import pprint

from utils import *

from typing import Set

class QA_Pairer():

    tag_split_re = re.compile(r"[\<\>]+")

    def __init__(self, xml_path, name=None, out_folder="out", min_score=3, max_responses=3, out_format="txt", archiver=None,
                 tokenizer=None):
        """Makes a text dataset from StackExchange dumps"""
        self.xml_path = xml_path
        if name is None:
            self.name = os.path.dirname(xml_path).replace("dumps/", "")
        else:
            self.name = name
        # dict to save questions
        self.questions = defaultdict(lambda: None, {})
        # folder to save txt files to
        self.out_folder = out_folder
        # min_score required to parse an answer
        self.min_score = min_score
        self.max_responses = max_responses
        assert out_format in ["txt", "lm_dataformat", "zip", "none"], "Out format not recognized"
        self.out_format = out_format
        if out_format in ["lm_dataformat", "zip"]:
            assert archiver is not None
            self.ar = archiver

        self.tag_counter = Counter()

        self.tokenizer = tokenizer
        self.token_counter = Counter()
        self.token_count = 0

        self.question_count = 0
        self.answer_count = 0

    def main(self):
        """iterates through SE xmls and:

        - stores PostTypeId="1" with AcceptedAnswerIds / Answers.
        - when an AcceptedAnswerId or Answer > min_score is reached, it should:
            > concat the Question & Accepted answer
            > Clean markup / HTML
            > Output to txt file
            > Delete from memory

        """
        os.makedirs(self.out_folder, exist_ok=True)
        for event, elem in tqdm(etree.iterparse(self.xml_path, events=('end',)), desc="Parsing {} XML file".format(self.name), ncols=80):
            if elem.tag == "row":
                try:
                    attribs = defaultdict(lambda: None, elem.attrib)
                    if is_question(attribs):
                        if has_answers(attribs):
                            trim_attribs(attribs, "question")
                            self.questions[attribs["Id"]] = attribs
                        else:
                            # if the question has no answers, discard it
                            continue
                    elif is_answer(attribs):
                        # if is accepted answer, append answer Body to relevant questions "AcceptedAnswer" field
                        # if the answer's score > min_score
                        # append the answer to the relevant question's OtherAnswers dict
                        self.add_answer(attribs)
                        self.check_complete(attribs)
                    elem.clear()
                except:
                    traceback.print_exc()
        print("processing complete")
        self.print_status()


    def is_above_threshold(self, a_attribs):
        """
        Determines whether an answer is above the min_score threshold

        :param a_attribs: Answer's attribute dict
        :return:
        """
        assert is_answer(a_attribs), "Must be an answer to be above threshold"
        if a_attribs["Score"] is not None:
            if int(a_attribs["Score"]) >= self.min_score:
                return True
        return False

    def add_answer(self, a_attribs):
        """
        Adds answer to its parent question in self.questions if it's either an accepted answer or above self.min_score.
         If answer is an accepted answer, it gets appended to the AcceptedAnswer field, otherwise it gets appended to
         OtherAnswers.

         Also increments the question's 'ParsedAnswers' field. When ParsedAnswers = AnswerCount, the question is deleted
         from memory and saved to a text file.

        :param a_attribs: Answer's attribute dict
        """
        assert is_answer(a_attribs), "Must be an answer to add to parent"
        if a_attribs is not None and self.questions[a_attribs["ParentId"]] is not None:
            if is_accepted_answer(a_attribs, self.questions[a_attribs["ParentId"]]):
                self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            elif self.is_above_threshold(a_attribs):
                if a_attribs["Id"] is not None:
                    parent = self.questions[a_attribs["ParentId"]]
                    if parent is not None:
                        self.questions[a_attribs["ParentId"]]["Answers"][a_attribs["Id"]] = trim_attribs(a_attribs, "answer")
                        self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
                else:
                    self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            else:
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1

    def write(self, out_name, str_rep):
        if self.out_format == "none":
            pass
        elif self.out_format == "txt":
            with open("{}/{}".format(self.out_folder, out_name), 'w') as f:
                try:
                    f.write(filter_newlines(out_str))
                except:
                    f.write(filter_newlines(handle_unicode_errors(out_str)))
        elif self.out_format == "zip":
            try:
                self.ar.writestr(out_name, filter_newlines(out_str))
            except:
                self.ar.writestr(out_name, filter_newlines(handle_unicode_errors(out_str)))
        elif self.out_format == "lm_dataformat":
            try:
                self.ar.add_data(filter_newlines(out_str), meta={
                    'name': out_name})
            except:
                self.ar.add_data(filter_newlines(handle_unicode_errors(out_str)), meta={
                    'name': out_name})

    @classmethod
    def get_tags(cls, attrib):
        if "Tags" not in attrib:
            return []
        tags = cls.tag_split_re.split(attrib["Tags"])
        return [t for t in tags if bool(t)]

    def update_tag_and_token_counts(self, tags, out_str):
        self.tag_counter.update(tags)
        if self.tokenizer is not None:
            tokens = self.tokenizer(out_str)['input_ids']
            token_count = len(tokens)
            for tag in tags:
                self.token_counter[tag] += token_count
            self.token_count += token_count

    def print_status(self):
        print(f"{self.question_count:_} questions")
        print(f"{self.answer_count:_} answers")
        print(f"{self.answer_count / self.question_count:.2f} answers / question")

        print("common tags:")
        underscore_print_counter(self.tag_counter, n=20)
        if self.tokenizer is not None:
            print(f"total tokens: {self.token_count:_}")
            underscore_print_counter(self.token_counter, n=20)
        print()

    def check_complete(self, a_attribs):
        """
        checks if the parent question of the previously added answer has no future answers, and if so,
        removes from dict and prints to file.
        """
        keys_to_del = []
        parent = self.questions[a_attribs["ParentId"]]
        if a_attribs is not None and parent is not None:
            if parent["AnswerCount"] is not None and parent["ParsedAnswers"] is not None:
                if int(parent["ParsedAnswers"]) == int(parent['AnswerCount']):
                    keys_to_del.append(a_attribs["ParentId"])
                    if parent["Answers"] is not None and len(parent["Answers"]) > 0:
                        out_name = "{}_{}.txt".format(self.name, parent["Id"].zfill(10))
                        out_str = ""
                        out_str += 'Q:\n\n'
                        if parent["Title"] is not None:
                            out_str += '{}\n\n'.format(BeautifulSoup(parent["Title"], "html.parser").get_text())
                        if parent["Body"] is not None:
                            out_str += '{}\n\n'.format(BeautifulSoup(parent["Body"], "html.parser").get_text())
                        if parent["Answers"] is not None:
                            key_score_dict = {}
                            for k, a in parent["Answers"].items():
                                key_score_dict[k] = int(a["Score"])
                            key_score_dict = {k: v for k, v in sorted(key_score_dict.items(), key=lambda item: item[1], reverse=True)}
                            count = 0
                            for k in key_score_dict:
                                if count >= self.max_responses:
                                    break
                                out_str += 'A:\n\n{}\n\n'.format(BeautifulSoup(parent["Answers"][k]["Body"], "html.parser").get_text())
                                count += 1
                                self.answer_count += 1

                        self.question_count += 1
                        tags = self.get_tags(parent)
                        self.update_tag_and_token_counts(tags, out_str)
                        self.write(out_name, out_str)

                        if self.question_count % 100_000 == 0:
                            self.print_status()

        for key in keys_to_del:
            self.questions.pop(key, None)

from bs4 import BeautifulSoup, NavigableString, CData, Tag


class CodePreservingBeautifulSoup(BeautifulSoup):
    """
    modified from https://stackoverflow.com/a/42802393, with changes for beautifulsoup 4.10
    """
    tags_to_keep = {'code'}

    def _all_strings(self, strip=False, types=BeautifulSoup.default):# strip=False, types=(NavigableString, CData)):

        if types is self.default:
            types = self.interesting_string_types

        for descendant in self.descendants:
            # return inner text within keep_tags, if we encounter them
            if isinstance(descendant, Tag) and descendant.name in self.tags_to_keep:
                yield f"<|{descendant.name}|>{descendant.get_text()}</|{descendant.name}|>"

            # skip an inner text node inside "a"
            if isinstance(descendant, NavigableString) and descendant.parent.name in self.tags_to_keep:
                1/0
                continue

            # default behavior
            if (types is None and not isinstance(descendant, NavigableString)):
                continue
            descendant_type = type(descendant)
            if isinstance(types, type):
                if descendant_type is not types:
                    # We're not interested in strings of this type.
                    1/0
                    continue
            elif types is not None and descendant_type not in types:
                # We're not interested in strings of this type.
                1/0
                continue
            if strip:
                descendant = descendant.strip()
                if len(descendant) == 0:
                    1/0
                    continue
