import os, re
import random

class Mean:
    def __init__(self):
        self.total = 0.0
        self.count = 0.0

    def add(self, value, weight=1.0):
        self.total += value
        self.count += weight

    @property
    def mean(self):
        if self.count == 0.0:
            return 0.0
        else:
            return self.total / self.count


def header_info(xml_path):
    os.system("head {}".format(xml_path))


def handle_unicode_errors(txt):
    return txt.encode('utf-8', 'replace').decode()


def is_question(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "1":
            return True
    return False


def is_answer(elem_attribs):
    if elem_attribs["PostTypeId"] is not None:
        if elem_attribs["PostTypeId"] == "2":
            return True
    return False


def filter_newlines(text):
    # replace three or more \n with \n\n
    return re.sub("\n{3,}", "\n\n", text)


def is_accepted_answer(a_attribs, q_attribs):
    assert is_question(q_attribs), "Must be a question to have an accepted answer"
    assert is_answer(a_attribs), "Must be an answer to be an accepted answer"
    if q_attribs["AcceptedAnswerId"] is not None:
        if q_attribs["AcceptedAnswerId"] == a_attribs["Id"]:
            return True
    else:
        return False


def has_answers(elem_attribs):
    assert is_question(elem_attribs), "Must be a question to have answers"
    if elem_attribs["AnswerCount"] is not None:
        if int(elem_attribs["AnswerCount"]):
            return True
    return False


def trim_attribs(elem_attribs, attrib_type="question"):
    """deletes non-useful data from attribs dict for questions / answers, returns remaining"""
    if attrib_type == "question":
        to_keep = ['Id', 'Body', 'Title', 'Tags', 'AnswerCount', 'AcceptedAnswerId', 'PostTypeId']
        to_delete = [x for x in elem_attribs.keys() if x not in to_keep]
        [elem_attribs.pop(x, None) for x in to_delete]
        elem_attribs["ParsedAnswers"] = 0
        elem_attribs["Answers"] = {}
    elif attrib_type == "answer":
        to_keep = ['Id', 'Body', 'Score']
        new_dict = {}
        for item in to_keep:
            new_dict[item] = elem_attribs[item]
        return new_dict
    else:
        raise Exception('Unrecognized attribute type - please specify either question or answer')

def underscore_print_counter(counter, n=None, prefix="\t"):
    for key, value in counter.most_common(n=n):
        print(f"{prefix}{key}:\t{value:_}")

def threshold(lower_bounds, value):
    # make sure we're not doing string comparisons
    assert float(value) == value
    assert all(float(x) == x for x in lower_bounds)

    if value < lower_bounds[0]:
        return 0

    disc_threshold = 0
    while disc_threshold < len(lower_bounds) and lower_bounds[disc_threshold] <= value:
        disc_threshold += 1
    disc_threshold -=1
    assert 0 <= disc_threshold < len(lower_bounds)
    return disc_threshold

def make_tagged(tag, inner, attributes={}, insert_newlines=True, attribute_move_probability=None):
    if attributes:
        attr_strs = [f'{k}={v}' for k, v in attributes.items()]
        if attribute_move_probability is not None:
            assert 0 <= attribute_move_probability <= 1.0
            begin_attr_strs = []
            end_attr_strs = []
            for x in attr_strs:
                if random.random() < attribute_move_probability:
                    end_attr_strs.append(x)
                else:
                    begin_attr_strs.append(x)
        else:
            begin_attr_strs = attr_strs
            end_attr_strs = []
    else:
        begin_attr_strs = []
        end_attr_strs = []
    del attr_strs
    if begin_attr_strs:
        random.shuffle(begin_attr_strs)
        begin_attr_string = f" {' '.join(begin_attr_strs)}"
    else:
        begin_attr_string = ''
    if end_attr_strs:
        random.shuffle(end_attr_strs)
        end_attr_string = f" {' '.join(end_attr_strs)}"
    else:
        end_attr_string = ''
    if insert_newlines:
        return f'<| {tag}{begin_attr_string} |>\n{inner}\n<|/ {tag}{end_attr_string} |>'
    else:
        return f'<| {tag}{begin_attr_string} |> {inner} <|/ {tag}{end_attr_string} |>'