import sys
import pprint
import argparse, traceback
from multiprocessing import Pool, cpu_count
from utils import *
from downloader import Stack_Exchange_Downloader
from pairer import QA_Pairer
import os
from itertools import repeat
from lm_dataformat import Archive
import zipfile


def download_and_process_single(name, args):
    out_format = args.out_format
    min_score = args.min_score
    max_responses = args.max_responses
    try:
        name = name.strip().lower()
        os.makedirs(args.in_folder, exist_ok=True)
        path_to_posts = "{}/{}/Posts.{}".format(args.in_folder, name, args.in_format)
        path_to_comments = "{}/{}/Comments.{}".format(args.in_folder, name, args.in_format)
        out_folder = args.out_folder
        os.makedirs(out_folder, exist_ok=True)
        if not os.path.isfile(path_to_posts):
            # extract 7z if it's not extracted already
            s = Stack_Exchange_Downloader(name)
            if name != "stackoverflow":
                path_to_7z = "{}/{}.7z".format(args.in_folder, s.sites[name]["url"])
            else:
                path_to_7z = "{}/stackoverflow.com-Posts.7z".format(args.in_folder)
            if not os.path.isfile(path_to_7z):
                # download 7z if it's not downloaded already
                s.download()
            s.extract()
        if out_format == "lm_dataformat":
            archiver = Archive(out_folder)
        elif out_format == "zip":
            archiver = zipfile.ZipFile('{}/{}.zip'.format(out_folder, name), 'a')
        elif out_format == "fairseq":
            raw_folder = os.path.join(out_folder, "raw")
            os.makedirs(raw_folder, exist_ok=True)
            bpe_folder = os.path.join(out_folder, "bpe")
            os.makedirs(bpe_folder, exist_ok=True)
            if args.num_shards is not None and args.shard_num is not None:
                suffix = f"_{args.shard_num}"
            else:
                suffix = ""
            raw_fname = os.path.join(raw_folder, f"{name}{suffix}.raw")
            bpe_fname = os.path.join(raw_folder, f"{name}{suffix}.bpe")
            archiver = open(raw_fname, 'w'), open(bpe_fname, 'w')
        else:
            archiver = None
        if args.count_tokens or out_format == 'fairseq':
            from tokenizers import ByteLevelBPETokenizer

            tokenizer = ByteLevelBPETokenizer.from_file(
                '/checkpoint/dpf/data/tokenizers/github-py+so_psno-True/vocab.json',
                '/checkpoint/dpf/data/tokenizers/github-py+so_psno-True/merges.txt',
                pretokenizer_split_newlines_only=True
                # args.tokenizer_vocab_file,
                # args.tokenizer_merges_file,
                # pretokenizer_split_newlines_only=args.tokenizer_split_newlines_only,
            )
        else:
            tokenizer = None
        qa = QA_Pairer(path_to_posts,
        name=name, 
        out_format=out_format, 
        archiver=archiver, 
        in_format=args.in_format,
        comment_path=path_to_comments, 
        max_responses=max_responses,
        max_comments=args.max_comments,
        min_score=min_score,
        shard_number=args.shard_number,
        num_shards=args.num_shards,
        tokenizer=tokenizer,
        count_tokens=args.count_tokens)
        qa.main()
        if out_format == "lm_dataformat":
            archiver.commit(name)
        elif out_format == "zip":
            archiver.close()
        elif out_format == "fairseq":
            for f in archiver:
                f.close()
        # try:
        #     os.remove(path_to_7z)
        # except FileNotFoundError:
        #     print('ERROR: FileNotFoundError: File {} not found'.format(s.sites[name]["url"]))
        # filelist = [f for f in os.listdir("dumps/{}".format(name)) if f.endswith(".xml")]
        # for f in filelist:
        #     os.remove(os.path.join("dumps/{}".format(name), f))
    except:
        traceback.print_exc()


def main(args):
    names = args.names.split(',')
    if names[0].strip().lower() == "all":
        s = Stack_Exchange_Downloader("all")
        names = []
        for k in s.sites:
            names.append(k)
        # bring stackoverflow to the front so it is always processed first, since it's the largest
        if "stackoverflow" in names:
            names.insert(0, names.pop(names.index("stackoverflow")))
    print('Downloading and processing stackexchange dumps for {}'.format(names))
    # Download & Process
    # init pool with as many CPUs as available
    cpu_no = cpu_count() - 1
    p = Pool(cpu_no)
    #p.starmap(download_and_process_single, zip(names, repeat(args.out_format), repeat(args.min_score), repeat(args.max_responses), repeat(args))
    p.starmap(download_and_process_single, zip(names, repeat(args)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='CLI for stackexchange_dataset - A tool for downloading & processing stackexchange dumps in xml form to a raw '
                    'question-answer pair text dataset for Language Models')
    parser.add_argument('--names', help='names of stackexchanges to download, extract & parse, separated by commas. '
                                        'If "all", will download, extract & parse *every* stackoverflow site',
                        default="stackoverflow",
                        type=str)
    parser.add_argument('--in_format', help='format of in file' 'lm_dataformat, as you will run into number of files per directory limits.',
                        default="xml",
                        choices=["xml", "csv"],
                        type=str)
    parser.add_argument('--out_format', help='format of out file - if you are processing everything this will need to be '
                                             'lm_dataformat, as you will run into number of files per directory limits.',
                        default="lm_dataformat",
                        choices=["txt", "lm_dataformat", "zip", "none"],
                        type=str)
    parser.add_argument('--min_score', help='minimum score of a response in order to be included in the dataset',
                        type=int, default=0)
    parser.add_argument('--max_responses', help='maximum number of responses (sorted by score) to include for each question. ', type=int, default=10)
    parser.add_argument('--max_comments', help='maximum number of comments (sorted consecutively by post time) to include for each question/answer', type=int, default=5)
    parser.add_argument('--num_shards', type=int)
    parser.add_argument('--shard_number', type=int)
    parser.add_argument('--count_tokens', action='store_true')
    parser.add_argument('--out_folder', default='out')
    parser.add_argument('--in_folder', default='dumps')
    # parser.add_argument('--tokenizer_vocab_file', type=str, default='/checkpoint/dpf/data/tokenizers/github-py+so_psno-True/vocab.json')
    # parser.add_argument('--tokenizer_merges_file', type=str, default='/checkpoint/dpf/data/tokenizers/github-py+so_psno-True/merges.txt')
    # parser.add_argument('--tokenizer_split_newlines_only', action='store_true')

    print(' '.join(sys.argv))
    args = parser.parse_args()
    pprint.pprint(vars(args))
    main(args)


