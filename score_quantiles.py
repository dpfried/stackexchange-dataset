import tqdm
import lxml.etree
import numpy as np
import pickle

def zeno(num_vals):
    last = 0
    vals = [last]
    while len(vals) < num_vals:
        last = 1 - ((1 - last) / 2)
        vals.append(last)
    return np.array(vals)

def stackexchange_reader(filename, rng, yield_rate=None, parse_html=True):
    basename = filename.split('/')[-1]
    if basename == 'Comments.xml':
        text_field = 'Text'
        num_rows = 82_037_744 - 3
    elif basename == 'Posts.xml':
        text_field = 'Body'
        num_rows = 53_949_888 - 3
    else:
        raise ValueError(f"unrecognized basename {basename}")

    with open(filename, 'rb') as f:
        for event, element in tqdm.tqdm(
            lxml.etree.iterparse(f), ncols=80, total=num_rows, desc=basename
        ):
            if event == 'end' and element.tag == 'row':
                if yield_rate is not None and rng.random() > yield_rate:
                    continue
                if text_field not in element.attrib:
                    continue
                text = element.attrib[text_field]
                score = element.attrib["Score"]
                is_answer = (element.attrib.get("PostTypeId") == "2")
                yield (int(score), is_answer)
                # if parse_html:
                #     from bs4 import BeautifulSoup
                #     parsed = BeautifulSoup(text, "html.parser")
                #     yield parsed.get_text()
                # else:
                #     yield text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--log_spacing", action='store_true')
    parser.add_argument("--buckets", type=int, default=6)

    args = parser.parse_args()

    filename = args.filename

    print(filename)

    question_or_comment_scores = []
    answer_scores = []

    for score, is_answer in stackexchange_reader(filename, None):
        (answer_scores if is_answer else question_or_comment_scores).append(score)

    for name, scores in [("question_or_comment", question_or_comment_scores), ("answer", answer_scores)]:
        scores = np.array(scores)
        scores = scores[scores >= 0]

        num_buckets = args.buckets

        if args.log_spacing:
            qs = zeno(num_buckets)
        else:
            qs = np.arange(num_buckets+1)/num_buckets

        print(name)
        for q, v in zip(qs, np.quantile(scores, qs)):
            print(f"{q:0.3f}: {v}")
        print()
