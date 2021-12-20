import re
import traceback
import xml.etree.ElementTree as etree
from collections import defaultdict, Counter
from bs4 import BeautifulSoup, PageElement, NavigableString, CData, Tag
from tqdm import tqdm
import pprint
import csv
import numpy as np

from utils import *

from typing import Set

class QA_Pairer():

    tag_split_re = re.compile(r"[\<\>]+")

    # remove @token if it occurs at the beginning of the string or preceeded by whitespace
    remove_username_re = re.compile(r"(^|\s+)@\w+")

    threshold_lower_bounds = {
        # ('stackoverflow', 'comments'): [0, 0, 0, 1, 1, 2], # not enough have a positive score to use
        ('stackoverflow', 'questions'): [0, 1, 2, 3, 6, 11],
        ('stackoverflow', 'answers'): [0, 1, 2, 4, 7, 14],
    }

    def __init__(self, post_path,
                name=None,
                out_folder="out",
                out_format="txt",
                archiver=None,
                in_format="xml",
                comment_path=None, 
                max_responses=3,
                max_comments=5,
                min_score=3,
                attribute_move_probability=0.5,
                shard_number=None,
                num_shards=None, 
                tokenizer=None,
                count_tokens=False):
        """Makes a text dataset from StackExchange dumps"""
        self.post_path = post_path
        self.comment_path = comment_path
        if name is None:
            self.name = os.path.dirname(post_path).replace("dumps/", "")
        else:
            self.name = name
        # dict to save questions
        self.questions = defaultdict(lambda: None, {})
        # folder to save txt files to
        self.out_folder = out_folder
        # min_score required to parse an answer
        self.min_score = min_score
        self.max_responses = max_responses
        self.max_comments = max_comments
        self.attribute_move_probability = attribute_move_probability
        assert in_format in ["csv", "xml"], "In format not recognized"
        self.in_format = in_format
        assert out_format in ["txt", "lm_dataformat", "zip", "none"], "Out format not recognized"
        self.out_format = out_format
        if out_format in ["lm_dataformat", "zip", "fairseq"]:
            assert archiver is not None
            self.ar = archiver

        self.tag_counter = Counter()

        self.count_tokens = count_tokens
        self.tokenizer = tokenizer
        self.token_counter = Counter()
        self.token_count = 0

        self.question_count = 0
        self.answer_count = 0

        self.shard_number = shard_number
        self.num_shards = num_shards

        # either None or (if comment_path was passed) a dict Dict[PostId: str, (score: int, text: str)
        self.comment_dict = self.parse_comments()

    def make_iter(self, file):
        if self.in_format == 'csv':
            with open(file, 'r') as f:
                f = (line.replace('\0', '') for line in f)
                reader = csv.DictReader(f)
                for row in reader:
                    record = defaultdict(lambda: None, {k: None if v == '' else v for k, v in row.items()})
                    yield record
        else:
            for event, elem in etree.iterparse(file, events=('end',)):
                if elem.tag == 'row':
                    record = defaultdict(lambda: None, elem.attrib)
                    yield record
                    elem.clear()

    def parse_comments(self):
        comment_dict = defaultdict(list)
        for record in tqdm(self.make_iter(self.comment_path), desc="Parsing {} comment file".format(self.name), ncols=120):
            text = record["Text"]
            if text is None:
                continue
            if self.in_format == 'xml':
                text = BeautifulSoup(text, "html.parser").get_text()
            text = self.remove_username_re.sub("", text)
            post_id = record["PostId"]
            comment_dict[post_id].append(text)
        self.comment_dict = comment_dict
        return comment_dict

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
        if self.shard_number is not None and self.num_shards is not None:
            question_ids = [int(record["Id"]) for record in tqdm(self.make_iter(self.post_path), desc="Get post ids for sharding", ncols=120) if is_question(record)]
            shard_question_ids = set(np.array_split(question_ids, self.num_shards)[self.shard_number])
        else:
            shard_question_ids = None

        for record in tqdm(self.make_iter(self.post_path), desc="Parsing {} posts".format(self.name), ncols=120):
            try:
                if is_question(record):
                    if shard_question_ids is not None:
                        question_id = int(record["Id"])
                        if question_id not in shard_question_ids:
                            continue
                        shard_question_ids.remove(question_id)
                    if has_answers(record):
                        trim_attribs(record, "question")
                        self.questions[record["Id"]] = record
                    else:
                        # if the question has no answers, discard it
                        continue
                elif is_answer(record):
                    # if is accepted answer, append answer Body to relevant questions "AcceptedAnswer" field
                    # if the answer's score > min_score
                    # append the answer to the relevant question's OtherAnswers dict
                    self.add_answer(record)
                    self.check_complete(record)
            except:
                traceback.print_exc()
        print("processing complete")
        self.print_status()

        if shard_question_ids is not None and len(shard_question_ids) != 0:
            print("warning: did not find {len(shard_question_ids)} questions ids that should have been in this shard (below):")
            print(' '.join(str(x) for x in sorted(shard_question_ids)))


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

    def write(self, out_name, out_str):
        if self.out_format == "none":
            pass
        elif self.out_format == "fairseq":
            assert self.tokenizer is not None
            raw_file, bpe_file = self.ar
            try:
                line = filter_newlines(out_str)
            except:
                line = filter_newlines(handle_unicode_errors(out_str))
            raw_file.write(line)
            raw_file.write("\n\n")

            bpe_file.write(' '.join(str(ix) for ix in self.tokenizer(line).ids))
            bpe_file.write("\n\n")
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
        if "Tags" not in attrib or attrib["Tags"] is None:
            return []
        tags = cls.tag_split_re.split(attrib["Tags"])
        return [t for t in tags if bool(t)]

    def update_tag_and_token_counts(self, tags, out_str):
        self.tag_counter.update(tags)
        if self.count_tokens and self.tokenizer is not None:
            tokens = self.tokenizer(out_str).ids
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
                        out_strs = []

                        question_body = ""

                        question_attrs = {}
                        tags = self.get_tags(parent)
                        random.shuffle(tags)
                        tag_str = ','.join(tags)
                        if tag_str:
                            question_attrs['tags'] = tag_str
                        
                        if (self.name, 'questions') in self.threshold_lower_bounds:
                            question_votes = int(parent['Score'])
                            question_attrs['dscore'] = threshold(self.threshold_lower_bounds[(self.name, 'questions')], question_votes)

                        if parent["TitleParsed"] is not None:
                            title_parsed = parent["TitleParsed"]
                            question_body += title_parsed
                        elif parent["Title"] is not None:
                            title_parsed = BeautifulSoup(parent["Title"], "html.parser").get_text()
                            question_body += title_parsed

                        if parent["BodyParsed"] is not None:
                            body_parsed = parent["BodyParsed"]
                            if question_body:
                                question_body += '\n\n{}'.format(body_parsed)
                            else:
                                question_body = body_parsed
                        elif parent["Body"] is not None:
                            body_parsed = CodePreservingBeautifulSoup(parent["Body"], "html.parser").get_text()
                            if question_body:
                                question_body += '\n\n{}'.format(body_parsed)
                            else:
                                question_body = body_parsed
                        
                        question_body = self.remove_username_re.sub("", question_body)
                        out_strs.append(make_tagged("q", question_body, question_attrs, attribute_move_probability=self.attribute_move_probability))

                        def add_comments(post_id):
                            if self.comment_dict is not None:
                                comments = self.comment_dict[parent["Id"]][:self.max_comments]
                                comment_str = '\n'.join(make_tagged('c', comment, {}) for comment in comments)
                                if comment_str:
                                    out_strs.append(comment_str)
                        
                        add_comments(parent["Id"])

                        if parent["Answers"] is not None:
                            answers = sorted(parent["Answers"].items(), lambda t: int(t[1]["Score"]), reverse=True)
                            count = 0
                            for key, answer in answers:
                                if count >= self.max_responses:
                                    break
                                if answer["BodyParsed"] is not None:
                                    answer_body_parsed = answer["BodyParsed"]
                                elif answer["Body"] is not None:
                                    answer_body_parsed = CodePreservingBeautifulSoup(answer["Body"], "html.parser").get_text()
                                else:
                                    continue

                                answer_body_parsed = self.remove_username_re.sub("", answer_body_parsed)

                                answer_attrs = {}
                                
                                if (self.name, 'answers') in self.threshold_lower_bounds:
                                    answer_votes = int(answer['Score'])
                                    answer_attrs['dscore'] = threshold(self.threshold_lower_bounds[(self.name, 'answers')], answer_votes)

                                if tag_str:
                                    answer_attrs['tags'] = tag_str

                                out_strs.append(make_tagged("a", answer_body_parsed, answer_attrs, attribute_move_probability=self.attribute_move_probability))

                                add_comments(answer["Id"])
                                
                                count += 1
                                self.answer_count += 1
                        
                        out_str = '\n'.join(out_strs)

                        self.question_count += 1
                        tags = self.get_tags(parent)
                        self.update_tag_and_token_counts(tags, out_str)
                        self.write(out_name, out_str)

                        if self.question_count % 100_000 == 0:
                            self.print_status()

        for key in keys_to_del:
            self.questions.pop(key, None)


class CodePreservingBeautifulSoup(BeautifulSoup):
    """
    modified from https://stackoverflow.com/a/42802393, with changes for beautifulsoup 4.10
    """
    tags_to_keep = {'code'}
    keep_only_with_newlines = True

    def _all_strings(self, strip=False, types=BeautifulSoup.default):# strip=False, types=(NavigableString, CData)):

        if types is self.default:
            types = self.interesting_string_types

        for descendant in self.descendants:
            # return inner text within keep_tags, if we encounter them
            if isinstance(descendant, Tag) and descendant.name in self.tags_to_keep and \
                ((not self.keep_only_with_newlines) or ('\n' in str(descendant))):

                #yield f"<|{descendant.name}|>{descendant.get_text()}</|{descendant.name}|>"
                yield str(descendant)

            # skip an inner text node inside "a"
            if isinstance(descendant, NavigableString) and descendant.parent.name in self.tags_to_keep and \
                ((not self.keep_only_with_newlines) or ('\n' in str(descendant))):
                continue

            # default behavior
            if (types is None and not isinstance(descendant, NavigableString)):
                continue
            descendant_type = type(descendant)
            if isinstance(types, type):
                if descendant_type is not types:
                    # We're not interested in strings of this type.
                    continue
            elif types is not None and descendant_type not in types:
                # We're not interested in strings of this type.
                continue
            if strip:
                descendant = descendant.strip()
                if len(descendant) == 0:
                    continue
            yield descendant
