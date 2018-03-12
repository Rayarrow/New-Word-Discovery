# Created by Zhao Xinwei.
# 2017.05.04.
# Some auxiliary functions are implemented here to facilitate printing.

import logging
import os
import re
import sys
from ast import literal_eval
from collections import defaultdict
from os.path import join, splitext

import numpy as np
import pandas as pd

# Default thresholds for stats columns.
# threshold_parameter = namedtuple('threshold_parameter', ['tf', 'agg_coef', 'max_entropy', 'min_entropy'])
# threshold_parameters = dict()
# threshold_parameters[0] = threshold_parameter(100, 2500, 0, 3)
# threshold_parameters[2] = threshold_parameter(100, 60, 0, 2)
# threshold_parameters[3] = threshold_parameter(100, 1000, 0, 2)

# Match the strings that contains at least 1 Chinese characters.
chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')

# Match the strings in which all characters are Chinesee.
chinese_string_pattern = re.compile(r'^[\u4e00-\u9fa5]+$')

# Characters considered to be punctuations.
punctuations = set('，。！？"!、.： ?')

# Configure the logger.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('New Word Discovery')


def load_dictionary(path):
    logger.info('Loading the dictionary...')
    with open(path, 'r', encoding='utf8') as f:
        return [line.split()[0] for line in f]


def load_lines_of_documents(path):
    documents = []
    if not os.path.isdir(path):
        with open(path, 'r', encoding='utf8') as f:
            documents = [line.strip() for line in f]
    else:
        for each_file in os.listdir(path):
            with open(os.path.join(path, each_file), 'r', encoding='utf8') as f:
                documents.extend(line.strip() for line in f)

    return list(set(documents)), os.path.basename(path)


def output_ordered_dict(path, an_ordered_dict, encoding='utf8'):
    """
    Save an `ordered dict` as a two-column table to `path`.
    """
    with open(path, 'w', encoding=encoding) as f:
        for each_unigram, count in an_ordered_dict.items():
            f.write('{} \t\t {}\n'.format(each_unigram, count))


def load_stats(path):
    """
    Read in a `stats` of type `DataFrame` with `encoding`, `index` and `header` specified.
    """
    stats = pd.read_csv(path, sep='\t', encoding='utf8', index_col=0, header=0, na_values=[''])
    logger.info(r'The stats {} are successfully loaded'.format(path))
    # If the index are not unigrams, convert the `str` form grams to `tuple` form.
    if re.match(r'(?:\(\'.*?\', )+\'.*?\'\)', stats.index[0]):
        logger.info(r'The index of {} are not unigrams. Commence the normalization process.'.format(path))
        stats.index = stats.index.map(literal_eval)
        logger.info(r'the index of {} normalized.'.format(path))
    return stats


def modify_stats_path(path, stats):
    """
    Add the specified threshold parameters in the file name. 
    """
    if stats.index.name is not None:
        return stats.index.name.join(splitext(path))
    else:
        return path


def output_stats(path, stats, preserve_grams=True):
    """
    This function do two other things on top of the basic `DataFrame.to_csv()` method:
        1. Specify the `float_format` and `encoding` parameters.
        2. If `preserve_grams` is set to `False`, then the x-grams will be concatenated to a complete string.
        Example. If set to True, then `('王八', '蛋')` will be converted to `'王八蛋'`.
    """
    if not preserve_grams and stats.shape[0] and isinstance(stats.index[0], tuple):
        stats.index, stats.index.name = stats.index.map(lambda x: ''.join(x)), stats.index.name
    stats.to_csv(path, sep='\t', float_format='%.5f', encoding='utf8')
    logger.info(r'Writing to `{}` succeed.'.format(path))


# !!!!!!!!!!!!!!!! Note that the entry in a 1_gram is taken as the unigram itself, not the characters that compose it.
def contain_punc(x_gram):
    """
    Determine if at least one of the entries in the given x-gram are punctuations.
    :return: 
    """
    return any(map(lambda x: x in punctuations, tuple(x_gram)))


def contain_non_chinese(x_gram):
    """
    If at least one grams in the `x_gram` contains non-Chinese character, return True.
    :return: 
    """
    return any(map(lambda x: not chinese_string_pattern.match(x), tuple(x_gram)))


def no_chinese_at_all(x_gram):
    """
    If every entry in the `x_gram` contains no Chinese characters, return True.
    :return: 
    """
    return not any(map(lambda x: chinese_pattern.match(x), tuple(x_gram)))


def verbose_logging(content, idx, length, verbose, *other_para):
    """
    A helper function to logging.
    :param content: the contents to be formatted and logged to the console.
    :param idx: the location of the being processed entry, i.e., the progress of the running function.
    :param length: the number of entries to be processed.
    :param verbose: This field controls the frequency of the logger. The logger log to the console when the process 
    reaches k * `verbose` quantile. In other words, the logger will log 1/`verbose` times in total.
        # Example. When `verbose`=0.02, the logger logs when the currently working function reaches 2%, 4%, 6%, etc.
    :param other_para: variables to be printed except for the progress-related variables (idx, length).
    :return: 
    """
    checkpoint = int(length * verbose)
    # Prevent division by zero.
    if checkpoint == 0:
        checkpoint = 1
    if not idx % checkpoint:
        logger.info(content.format(*other_para, idx, length))


def infer_counter_type(counter):
    """
    Given a counter, infer the type of its entries.
    Example. The type of entries in `discoverer.unigram_counter` is `unigram`.
    """
    counter_type = {1: 'unigram',
                    2: 'bigram',
                    3: 'trigram'}
    if not counter:
        return 'Unknown'
    else:
        return counter_type[len(next(iter(counter)))]


def filter_stats(stats, tf=1, agg=0, max_entropy=0, min_entropy=0, verbose=2, by='tf'):
    """
    Return a `stats` preserving only the words of which attributes reach the thresholds.
    """
    stats = stats.sort_values(by=by, ascending=False)
    stats = stats[
        (stats.tf >= tf) & (stats.agg_coef >= agg) & (stats.max_entropy >= max_entropy) & (stats.min_entropy >= min_entropy)]
    if verbose == 0:
        stats = stats[['tf']]
    elif verbose == 1:
        stats = stats[['tf', 'agg_coef', 'max_entropy', 'min_entropy']]
    elif verbose == 2:
        stats = stats[['tf', 'agg_coef', 'max_entropy', 'min_entropy', 'left_entropy', 'right_entropy']]
    else:
        raise Exception('Invalid `verbose`.')

    # Store the config to its index name field. (`pd.DataFrame` has no `name` field)
    stats.index.name = '{} # {} # {} # {} # {}'.format(tf, agg, max_entropy, min_entropy, verbose)
    return stats


def purify_stats(stats, length=2, pattern=r'[.a-zA-Z\u4e00-\u9fa5]', returning_non_pure=False):
    """
    Select out the rows that the corresponding terms are reasonable characters. Refer to `pure_index` variable below.
    On top of that, `NULL` entries are removed here.
    :param returning_non_pure: If this is true, the stats of the unreasonable terms will also be returned.
    """
    if not stats.shape[0]:
        logger.info(r'Empty stats. Nothing done.')

    # Remove `NULL` entries.
    stats = stats[pd.notnull(stats.index)]

    index = stats.index
    # If the index is not unigram, concatenate the x-grams to a str.
    if not isinstance(index[0], str):
        index = index.map(lambda x: ''.join(x))

    pure_index = (index.str.contains(pattern)) & (index.str.len() >= length)

    if returning_non_pure:
        return stats[pure_index], stats[~pure_index]
    else:
        return stats[pure_index]


def decompose_stats(stats):
    """
    Decompose the stats of Chinese words and Latin words.
    The `stats` of Chinese words are further divided into several blocks based on the length of the words.
    """
    agg_inf_index = (stats.agg_coef == np.inf)
    latin_pure_new_unigram_stats = stats[agg_inf_index]
    chinese_pure_new_unigram_stats = stats[~agg_inf_index]
    return chinese_pure_new_unigram_stats, latin_pure_new_unigram_stats


def generate_report_file_path(output_home, corpus_name, iteration, stats_type):
    """
    Compose a human-readable path from a series of parameters.
    """
    return join(output_home, 'report_{} [{}]_{}.csv'.format(corpus_name, iteration, stats_type))


def generate_report(output_home, new_unigram_stats, bigram_stats, threshold_parameters, preserve_grams=False,
                    corpus_name='default_corpus', unigram_max_len=3, verbose=0, iteration=1):
    """
    Select out the new words based on the given `threshold_parameters`, which are in turn used to generate reports and 
    update the dictionary. (the new words are returned to update the dictionary outside this function)
    """
    new_words = list()
    pure_new_unigram_stats, messy_new_unigram_stats = purify_stats(new_unigram_stats, returning_non_pure=True)

    # output messy new unigram.
    # messy_new_unigram_stats_verbose_2 = filter_stats(messy_new_unigram_stats)
    # output_stats('./output/messy_new_unigram_verbose_2.csv', messy_new_unigram_stats_verbose_2)

    chinese_pure_new_unigram_stats, latin_pure_new_unigram_stats = decompose_stats(pure_new_unigram_stats)
    p = threshold_parameters['latin']
    latin_pure_new_unigram_stats = filter_stats(latin_pure_new_unigram_stats, tf=p.tf, agg=p.agg_coef,
                                                max_entropy=p.max_entropy, min_entropy=p.min_entropy,
                                                verbose=verbose)
    output_stats(generate_report_file_path(output_home, corpus_name, iteration, 'latin'),
                 latin_pure_new_unigram_stats)
    new_words.extend(list(latin_pure_new_unigram_stats.index))

    # Generate the report for unigrams containing Chinese with different length.
    chinese_pure_new_unigram_stats_by_len = chinese_pure_new_unigram_stats.groupby(len)

    chinese_sub_stats_s = defaultdict(lambda: None)
    for each_length in sorted(set(chinese_pure_new_unigram_stats.index.map(
            lambda x: len(x) if len(x) < unigram_max_len else unigram_max_len))):
        p = threshold_parameters[each_length]
        chinese_sub_stats = chinese_pure_new_unigram_stats_by_len.get_group(each_length)
        chinese_sub_stats = filter_stats(chinese_sub_stats, tf=p.tf, agg=p.agg_coef, max_entropy=p.max_entropy,
                                         min_entropy=p.min_entropy, verbose=verbose)
        output_stats(
            generate_report_file_path(output_home, corpus_name, iteration, 'chinese_unigrams@{}'.format(each_length)),
            chinese_sub_stats)
        chinese_sub_stats_s[each_length] = chinese_sub_stats
        new_words.extend(list(chinese_sub_stats.index))

    # Genereate the report for bigrams.
    p = threshold_parameters['bigram']
    bigram_stats = filter_stats(bigram_stats, tf=p.tf, agg=p.agg_coef, max_entropy=p.max_entropy,
                                min_entropy=p.min_entropy, verbose=verbose)
    output_stats(generate_report_file_path(output_home, corpus_name, iteration, 'bigram'), bigram_stats,
                 preserve_grams=preserve_grams)
    new_words.extend(list(bigram_stats.index.map(lambda x: ''.join(x))))

    # return the reports of each invocation of `generate_report()` to comprise a complete report with the result of each
    # iteration merged.
    return new_words, {'latin': latin_pure_new_unigram_stats, 'chinese_unigram': chinese_sub_stats_s,
                       'bigram': bigram_stats}
