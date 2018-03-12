# Created by Zhao Xinwei.
# 2017.05.??.
# Used to load the corpora and execute the new word discovering algorithm.

from argparse import ArgumentParser
from collections import namedtuple

import jieba

from discover_utils import *
from discoverer import Discoverer


# For 4000 lines of corpus.
default_latin = [10, 0, 0, 0]
default_bigram = [10, 50, 0, 1]
default_unigram2 = [10, 2, 0, 1]
default_unigram3 = [10, 2, 0, 1]
default_iteration = 2
default_verbose = 0

arg_parser = ArgumentParser('New Words Discovery',
                            usage='Discover new words from corpus according to term frequency, aggreagation coefficient, min neighboring entropy and max neighboring entropy.')

arg_parser.add_argument('input_path',
                        help='The path to the corpus. It should be a plain text file or a dir containing only plain text files.')
arg_parser.add_argument('output_path', help='The path to generate the reports.')
arg_parser.add_argument('--dictionary_path', default=os.path.join(os.path.dirname(jieba.__file__), 'dict.txt'),
                        help='The path to the dictionary (text), each line of which contains item, POS-tag and frequency, seperated by spaces. Terms satisfying the filter condition but in the dictionary are not considered as new words.')
arg_parser.add_argument('--latin', nargs=4, default=default_latin, type=int,
                        help='The parameters include term frequency, aggreagation coefficient, max neighboring entropy and min neighboring entropy, which also applies for --bigram, --unigram_2 and --unigram_3. This argument set thresholds for latin words, including pure digits, pure letters and the combination of letters and digits such as "iphone 7".')
arg_parser.add_argument('--bigram', nargs=4, default=default_bigram, type=float,
                        help='Bigrams are defined as words that are composed of two unigram terms. Reference argument --latin for further help.')
arg_parser.add_argument('--unigram_2', nargs=4, default=default_unigram2, type=float,
                        help='A term which is composed of two Chinese characters and cannot be divided into other words. Reference argument --latin for further help.')
arg_parser.add_argument('--unigram_3', nargs=4, default=default_unigram3, type=float,
                        help='A term which is composed of three Chinese characters and cannot be divided into other words. Reference argument --latin for further help.')
arg_parser.add_argument('--iteration', default=default_iteration, type=float,
                        help='The next iteration will base its dictionary as the original dictionary plusing the new words discovered in the last iteration.')
arg_parser.add_argument('--verbose', default=default_verbose, choices=[0, 1, 2], type=int,
                        help="Determines the verbosity of the reports. *** 0: only new word items and their term frequency.*** 1: min neighboring entropy and max neighboring entropy are supplemented. *** 2:left and right neighboring entropy are added.")
args = arg_parser.parse_args()

documents, corpus_name = load_lines_of_documents(args.input_path)

output_home = join(args.output_path, corpus_name)
if not os.path.exists(output_home):
    logger.info('Output path does not exists and created.')
    os.makedirs(output_home)

threshold_parameter = namedtuple('threshold_parameter', ['tf', 'agg_coef', 'max_entropy', 'min_entropy'])
threshold_parameters = dict()

threshold_parameters['bigram'] = threshold_parameter(*args.bigram)
threshold_parameters['latin'] = threshold_parameter(*args.latin)
threshold_parameters[2] = threshold_parameter(*args.unigram_2)
threshold_parameters[3] = threshold_parameter(*args.unigram_3)

dictionary = load_dictionary(args.dictionary_path)

discoverer = Discoverer(save_segmentation=False)

# Used to store stats generated in each iteration.
stats_ind = list()

import time

for iteration in range(args.iteration):
    time.sleep(1)
    logger.info("""
   **********************************************************************
    
    commencing iteration {}...
    
   **********************************************************************
    """.format(iteration + 1))
    discoverer.fit(documents, corpus_name + ' [{}]'.format(iteration + 1))
    discoverer.get_new_unigrams(dictionary)

    # Add new words to the `dictionary`.
    new_words, current_stats = generate_report(output_home, discoverer.new_unigram_stats, discoverer.bigram_stats,
                                               threshold_parameters, corpus_name=corpus_name, iteration=iteration + 1,
                                               verbose=args.verbose)
    dictionary += new_words
    stats_ind.append(current_stats)
    for each_new_word in new_words:
        jieba.add_word(each_new_word)

# Output complete reports with the results of each iteration concatenated.
by = 'tf'
overall_latin_new_unigram_stats = pd.concat(
    [each_stats['latin'] for each_stats in stats_ind]).sort_values(by=by, ascending=False)
overall_new_bigrams_stats = pd.concat(
    [each_stats['bigram'] for each_stats in stats_ind]).sort_values(by=by, ascending=False)
output_stats(join(output_home, 'overall_latin.csv'), overall_latin_new_unigram_stats)
output_stats(join(output_home, 'overall_bigrams.csv'), overall_new_bigrams_stats)

for each_length in stats_ind[0]['chinese_unigram']:
    # ====================================================================================================
    # ====================================================================================================

    overall_chinese_sub_unigrams_verbose = pd.concat(
        [each_stats['chinese_unigram'][each_length] for each_stats in stats_ind]).sort_values(by=by,
                                                                                              ascending=False)
    output_stats(join(output_home, 'overall_chinese_unigrams@{}.csv'.format(each_length)),
                 overall_chinese_sub_unigrams_verbose)
