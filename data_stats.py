import lxml.etree
import tqdm
import humanize
from transformers import GPT2TokenizerFast
import argparse
from bs4 import BeautifulSoup
from collections import Counter, defaultdict
import numpy as np

def readable(x, is_size=False):
    if isinstance(x, float):
        return f"{x:.2f}"
    else:
        if is_size:
            return humanize.naturalsize(x)
        else:
            return f"{x:_}"

def print_counter(counter, human_readable=True, limit=None, is_size=False):
    for k, v in counter.most_common(limit):
        if human_readable:
            v = readable(v, is_size=is_size)
        print(f"\t{k}:\t{v}")
    print()

def print_aggregated_values(value_dict, text="", human_readable=True, limit=None, is_size=False,
                            quantiles=np.array([0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]),
                            keys_to_print=None):
    d = {
        k: np.array(v) for k, v in value_dict.items()
    }
    if text:
        text = f"{text} "
    print(f"{text}mean:")
    mean = Counter({k: int(v.mean()) for k, v in d.items()})
    print_counter(mean, human_readable=human_readable, limit=limit, is_size=is_size)
    print(f"{text}quantiles{quantiles}:")
    if keys_to_print is None:
        keys_to_print = [k for k, _ in mean.most_common(limit)]
    for k in keys_to_print:
        q = np.quantile(d[k], quantiles)
        readable_q = ' | '.join(readable(int(x), is_size=is_size) for x in q)
        print(f"\t{k}:\t{readable_q}")
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default='dumps/stackoverflow/Comments.xml')
    parser.add_argument("--subsample", type=int)
    args = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=False)

    no_text_count = 0

    total_tokens = 0
    total_text_size = 0
    num_entries = 0

    filename = args.filename
    
    num_rows = {
        'dumps/stackoverflow/Comments.xml': 82_037_744 - 3,
        'dumps/stackoverflow/Posts.xml': 53_949_888 - 3,
    }.get(args.filename, None)

    basename = args.filename.split('/')[-1]

    if basename == 'Comments.xml':
        text_field = 'Text'
        parse_html = False
    elif basename == 'Posts.xml':
        text_field = 'Body'
        parse_html = True
    else:
        raise ValueError("should be {Comments,Posts}.xml")

    if parse_html:
        tag_counter = Counter()
        size_per_tag = defaultdict(list)
        total_size_per_tag = Counter()

    with open(filename, 'rb') as f:
        for event, element in tqdm.tqdm(lxml.etree.iterparse(f), ncols=80,
                                        total=num_rows):
            if event == 'end' and element.tag == 'row':
                num_entries += 1
                if args.subsample is not None and num_entries % args.subsample == 0:
                    continue
                if text_field not in element.attrib:
                    no_text_count += 1
                    continue
                text = element.attrib[text_field]
                if parse_html:
                    parsed = BeautifulSoup(text, "html.parser")
                    text = parsed.get_text()
                    for tag in parsed.findAll():
                        tag_counter[tag.name] += 1
                        tag_size = len(tag.get_text())
                        size_per_tag[tag.name].append(tag_size)
                        total_size_per_tag[tag.name] += tag_size

                total_text_size += len(text)
                tokens = tokenizer(text)['input_ids']
                total_tokens += len(tokens)
                if num_entries % 100000 == 0:
                    print(f"{num_entries:_} entries:\t{humanize.naturalsize(total_text_size)}\t{total_tokens:_} tokens\t{total_tokens/num_entries:.2f} tokens/entries")
                    if no_text_count > 0:
                        print(f"{no_text_count} entries without text")
                    print()
                    if parse_html:
                        print("top tag counts:")
                        print_counter(tag_counter, is_size=False, limit=20)
                        print("total text size per tag:")
                        print_counter(total_size_per_tag, is_size=True, limit=20)
                        common_tags = [k for k, _ in tag_counter.most_common(20)]
                        print_aggregated_values(size_per_tag, "size per tag", is_size=True, limit=20, keys_to_print=common_tags)
