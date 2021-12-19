import tqdm
import lxml.etree
import numpy as np
import pickle

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
                yield int(score)
                # if parse_html:
                #     from bs4 import BeautifulSoup
                #     parsed = BeautifulSoup(text, "html.parser")
                #     yield parsed.get_text()
                # else:
                #     yield text

if __name__ == "__main__":
    #filename = 'dumps/stackoverflow/Comments.xml'
    filename = 'dumps/stackoverflow/Posts.xml'
    out_fname = filename+"_scores-filt.pkl"

    print(filename)

    scores = np.array(list(stackexchange_reader(filename, None)))

    scores = scores[scores > 0]

    def make_qs(num_buckets):
        return np.arange(num_buckets+1)/num_buckets

    with open(out_fname, 'wb') as f:
        pickle.dump(scores, f)

    for buckets in [5, 10]:
        qs = make_qs(buckets)
        for q, v in zip(qs, np.quantile(scores, qs)):
            print(f"{q:0.3f}: {v}")
        print()
