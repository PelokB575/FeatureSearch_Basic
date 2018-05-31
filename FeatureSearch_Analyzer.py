import datetime
import os
import re
import arff
from scipy.io import arff as arff_
import pandas


def feature_search_start(base_dir='.', feat_search=True, make_arff=True, make_stats=True, explicit_arff_path=''):
    list_of_users = next(os.walk(base_dir))[1]
    feature_list_final = {}

    current_user_no = 1

    if feat_search:
        for cpp_folder in list_of_users:
            print(f"Constructing data on {cpp_folder}... ({current_user_no} out of {len(list_of_users)} users)")
            folder_dir = None

            try:
                folder_dir = next(os.walk(os.path.join(base_dir, cpp_folder)))[2]
            except StopIteration:
                return "ERROR WHILE WALKING THROUGH FOLDERS"
            finally:
                feature_list_final.setdefault(cpp_folder, {})
                FL_temp = {}

                for file in folder_dir:
                    if file.endswith(".cpp") and file[0] != '.':
                        FL_temp.setdefault(file, {})
                        FL_temp[file] = get_features_for_file(os.path.join(base_dir, cpp_folder) + '\\' + file,
                                                              cpp_folder)

                feature_list_final[cpp_folder] = FL_temp

            current_user_no += 1

    if make_arff:
        file_path = dump_to_arff(feature_list_final, list_of_users)
    else:
        file_path = explicit_arff_path

    if make_stats:
        stats_list_final = take_stats_from_arff(file_path)
        dump_stats_to_csv(stats_list_final)

    return feature_list_final


def get_features_for_file(file, author):
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
        feat_list = {
            'Non-empty Lines': 0,
            'Words': 0,
            'Chars': 0,
            'Ternary Operations': 0,
            'Comments': 0,
            'Comment Readability': 0,
            'Multi-line to All Comments': 0,
            'Inline to All Comments': 0,
            'Unindented Comments': 0,
            'Macros': 0,
            '#INCLUDE': 0,
            '#DEFINE': 0,
            '#IFNDEF': 0,
            '#IFDEF': 0,
            'Using STL Libraries': 0,
            'Literals': 0,
            'Literal Name Readability': 0,
            'Custom Types': 0,
            'Custom Type Name Readability': 0,
            'Object Definitions': 0,
            'Object Declarations': 0,
            'String Readability': 0,
            'All Keywords': 0,
            'IF Keywords': 0,
            'ELSE Keywords': 0,
            'FOR Keywords': 0,
            'WHILE Keywords': 0,
            'SWITCH Keywords': 0,
            'DO Keywords': 0,
            'Unique Words': 0,
            'Functions': 0,
            'Function Name Readability': 0,
            'Nesting Depth': 0,
            'Average Parameter Count': 0,
            'Parameter Count Deviation': 0,
            'Commands': 0,
            'Average Commands per Line': 0,
            'Commands per Line Deviation': 0,
            'Empty Lines': 0,
            'Tabulators': 0,
            'Spaces': 0,
            'Space Indents': 0,
            'Tab Indents': 0,
            'Prefers Tabs over Spaces': 0,
            'Whitespace to Character Ratio': 0,
            'Average Line Length': 0,
            'Line Length Deviation': 0,
            'Author': str(author)
        }

        line_lengths = []
        unique_words = {}

        function_parameter_count = []

        all_literals = []
        all_functions = []
        all_typedefs = []
        all_strings = []
        all_commented_words = []

        commands_per_non_empty_line = []

        last_row_was_comment = 0
        total_comments = 0
        multiline_comments = 0
        inline_comments = 0
        unindented_comments = 0

        ternaries = 0

        stl_headers = r'(vector|deque|list|set|map|stack|queue|iterator|algorithm|numeric|functional|utility|memory)'

        for line in f.readlines():
            feat_list['Chars'] += len(line)
            if line == '\n':
                feat_list['Empty Lines'] += 1
                continue
            else:
                feat_list['Non-empty Lines'] += 1
                words_in_line = re.findall(r'([A-Z]?[a-z]+|[A-Z]+)', line)
                feat_list['Words'] += len(words_in_line)

                for word in words_in_line:
                    if word.lower() not in unique_words:
                        unique_words.setdefault(word.lower(), 0)
                    unique_words[word.lower()] += 1

                line_lengths.append(len(line))

                feat_list['Spaces'] += len(re.findall(r'\s', line))
                feat_list['Tabulators'] += len(re.findall(r'\t', line))

                if re.findall(r'^\s+', line):
                    feat_list['Space Indents'] += 1

                if re.findall(r'^\t+', line):
                    feat_list['Tab Indents'] += 1

                found_comment = re.findall(r'^[\s\t]*(//|/\*)[\s\t]*', line)
                if found_comment:
                    all_commented_words += re.findall(r'([A-Z]?[a-z]+|[A-Z]+)', found_comment[0])
                    total_comments += 1
                    if last_row_was_comment == 1:
                        multiline_comments += 1
                    last_row_was_comment += 1
                    if line[0] == '/':
                        unindented_comments += 1
                else:
                    last_row_was_comment = 0

                found_comment = re.findall(r'(?:[};{])\s*(?://|/\*)(.*)$', re.sub(r'".*"', '""', line))
                if found_comment:
                    total_comments += 1
                    inline_comments += 1
                    last_row_was_comment = 1
                    all_commented_words += re.findall(r'([A-Z]?[a-z]+|[A-Z]+)', found_comment[0])

                max_nest_depth_in_line = calculate_nesting_depth(re.sub(r'".*"', '""', line))
                if max_nest_depth_in_line > feat_list['Nesting Depth']:
                    feat_list['Nesting Depth'] = max_nest_depth_in_line

                feat_list['Commands'] += len(re.findall(r';', re.sub(r'".*"', '""', line)))
                commands_per_non_empty_line.append(len(re.findall(r';', re.sub(r'".*"', '""', line))))

                found_keyword = re.search(r'^[\s\t]*(if|else|for|while|switch|do)', re.sub(r'".*"', '""', line), re.IGNORECASE)
                if found_keyword:
                    feat_list[f"{found_keyword.group(1).upper()} Keywords"] += 1

                ternaries += len(re.findall(r'(=|==|<|>|<=|>=|!=)[\s\t]*.*\?.*:.*]', line))

                found_macro = re.search(r'^#(include|define|ifdef|ifndef)', line, re.IGNORECASE)
                if found_macro:
                    feat_list['Macros'] += 1
                    feat_list[f"#{found_macro.group(1).upper()}"] += 1
                    if not feat_list['Using STL Libraries'] and re.findall(r'^#include', line, re.IGNORECASE) and re.findall(stl_headers, line, re.IGNORECASE):
                        feat_list['Using STL Libraries'] = 1

                found_literals_raw = [re.split(r',?\s*=?', s[1]) for s in re.findall(r'(?:\w+(\[\d+\]|<\w+>)?\s+)+((?:\w+\s*=?)(?:,\s*(?:\w+\s*=?))*)', line)]
                found_literals = []
                for s in found_literals_raw:
                    if s != '':
                        found_literals += s
                feat_list['Literals'] += len(found_literals)

                all_literals += found_literals

                # Is this, like it or not, the peak of regexp evolution? Probably not
                found_functions = re.findall(r'(?:\w+(?:\[\d+\]|<\w+>)?[*&]?\s+)+(\w+)\(\s*(?:(?:\w+(?:\[\d+\]|<\w+>)?[*&]?\s+)+(\w+))(?:\s*,\s*(?:\w+(?:\[\d+\]|<\w+>)?[*&]?\s+)+(\w+))*\s*\){?', line)
                if found_functions:
                    all_functions.append(found_functions[0][0])
                    function_parameter_count.append(len(found_functions[0]) - 1)
                    feat_list['Functions'] += 1

                found_typedefs = re.findall(r'^\s*(?:\w+(?:\[\d+\]|<\w+>)?[*&]?\s+)*typedef(?:\w+(?:\[\d+\]|<\w+>)?[*&]?\s+)+(\w+)?\s*[;{]', line, re.IGNORECASE) + re.findall(r'^\s*}\s*(\w+)\s*;', line)
                if found_typedefs:
                    feat_list['Custom Types'] += 1
                    all_typedefs += found_typedefs

                feat_list['Object Declarations'] += len(re.findall(r'class', re.sub(r'".*"', '""', line)))
                feat_list['Object Definitions'] += len(re.findall(r'new', re.sub(r'".*"', '""', line)))

                all_strings += [re.findall(r'([A-Z]?[a-z]+|[A-Z]+)', s) for s in re.findall(r'"(.*)"', line)]

                re.purge()

        feat_list['Unique Words'] = len(unique_words) / feat_list['Words'] if feat_list['Words'] > 0 else 0
        # There shouldn't need to be a check for this... Special thanks for strelok1918 who decided to put a swastika as the sole content of their file
        # I would laugh, but I spent ~2 hours running this program just so that I could get a ZeroDivisionError because of you...

        feat_list['All Keywords'] = (feat_list['IF Keywords'] + feat_list['FOR Keywords'] + feat_list['ELSE Keywords'] +
                                     feat_list['WHILE Keywords'] + feat_list['DO Keywords'] + feat_list['SWITCH Keywords']) / feat_list['Non-empty Lines']

        feat_list['Ternary Operations'] = ternaries / feat_list['Non-empty Lines']

        feat_list['IF Keywords'] /= feat_list['Non-empty Lines']
        feat_list['ELSE Keywords'] /= feat_list['Non-empty Lines']
        feat_list['FOR Keywords'] /= feat_list['Non-empty Lines']
        feat_list['WHILE Keywords'] /= feat_list['Non-empty Lines']
        feat_list['SWITCH Keywords'] /= feat_list['Non-empty Lines']
        feat_list['DO Keywords'] /= feat_list['Non-empty Lines']

        feat_list['Comments'] = total_comments / feat_list['Non-empty Lines']
        feat_list['Multi-line to All Comments'] = (multiline_comments / total_comments) / feat_list['Non-empty Lines'] if total_comments > 0 else 0
        feat_list['Inline to All Comments'] = (inline_comments / total_comments) / feat_list['Non-empty Lines'] if total_comments > 0 else 0
        feat_list['Unindented Comments'] = 1 if unindented_comments > total_comments / 2 else 0

        feat_list['Macros'] /= feat_list['Non-empty Lines']
        feat_list['#INCLUDE'] /= feat_list['Non-empty Lines']
        feat_list['#DEFINE'] /= feat_list['Non-empty Lines']
        feat_list['#IFDEF'] /= feat_list['Non-empty Lines']
        feat_list['#IFNDEF'] /= feat_list['Non-empty Lines']

        feat_list['Functions'] /= feat_list['Non-empty Lines']
        feat_list['Average Parameter Count'] = sum(function_parameter_count) / len(function_parameter_count) if function_parameter_count != [] else 0
        feat_list['Parameter Count Deviation'] = sum([abs(i - feat_list['Average Parameter Count']) for i in function_parameter_count]) / len(function_parameter_count) if function_parameter_count != [] else 0

        feat_list['Custom Types'] /= feat_list['Non-empty Lines']

        feat_list['Object Declarations'] /= feat_list['Non-empty Lines']
        feat_list['Object Definitions'] /= feat_list['Non-empty Lines']

        feat_list['Space Indents'] /= feat_list['Non-empty Lines']
        feat_list['Tab Indents'] /= feat_list['Non-empty Lines']

        feat_list['Whitespace to Character Ratio'] = (feat_list['Spaces'] + feat_list['Tabulators']) / feat_list['Chars']

        feat_list['Average Commands per Line'] = feat_list['Commands'] / feat_list['Non-empty Lines']
        feat_list['Commands per Line Deviation'] = sum([abs(i - feat_list['Average Commands per Line']) for i in commands_per_non_empty_line]) / feat_list['Non-empty Lines']

        feat_list['Prefers Tabs over Spaces'] = 1 if feat_list['Tab Indents'] > feat_list['Space Indents'] else 0
        feat_list['Average Line Length'] = sum(line_lengths) / len(line_lengths)
        feat_list['Line Length Deviation'] = sum([abs(i - feat_list['Average Line Length']) for i in line_lengths]) / len(line_lengths)

        feat_list['Literal Name Readability'] = check_readability(format_words(all_literals))
        feat_list['Comment Readability'] = check_readability(format_words(all_commented_words))
        feat_list['String Readability'] = check_readability(format_words(all_strings))
        feat_list['Custom Type Name Readability'] = check_readability(format_words(all_typedefs))
        feat_list['Function Name Readability'] = check_readability(format_words(all_functions))

    return feat_list


def format_words(words):
    formatted = []
    for word in words:
        if word != '':
            if type(word) == list:
                for w in word:
                    if w not in formatted:
                        formatted += [w]
            else:
                if word not in formatted:
                    formatted += [word]
    return formatted


def calculate_nesting_depth(line):
    bracket_contents = re.findall(r'\((.*)\)|\[(.*)\]', line)

    if not bracket_contents:
        return 1

    return 1 + max(calculate_nesting_depth(cont[0] + cont[1]) for cont in bracket_contents)


def check_readability(words):
    if not words:
        return 0

    readable = 0

    with open('words.txt', 'r') as readable_words:

        for word in words:
            is_readable = word in readable_words.read()
            if is_readable:
                readable += 1

    return readable / len(words)


def dump_to_arff(data, users):
    arff_data = []

    for user in data:
        for file in data[user]:
            entry = [data[user][file][v] for v in data[user][file]]
            arff_data.append(entry)

    main_folder_name = base.split('\\')[-1]
    filename = f"{main_folder_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))}.arff"
    with open(filename, "w") as file:
        profiling_data = {
            'relation': 'programmer_profiling',
            'attributes': [
                ('non_empty_lines', 'REAL'),
                ('words', 'REAL'),
                ('chars', 'REAL'),
                ('ternary_operations', 'REAL'),
                ('comments', 'REAL'),
                ('comment_readability', 'REAL'),
                ('multiline_to_all_comments', 'REAL'),
                ('inline_to_all_comments', 'REAL'),
                ('unindented_comments', 'REAL'),
                ('macros', 'REAL'),
                ('#INCLUDE', 'REAL'),
                ('#DEFINE', 'REAL'),
                ('#IFNDEF', 'REAL'),
                ('#IFDEF', 'REAL'),
                ('using_STL_libraries', 'REAL'),
                ('literals', 'REAL'),
                ('literal_name_readability', 'REAL'),
                ('custom_types', 'REAL'),
                ('custom_type_name_readability', 'REAL'),
                ('object_definitions', 'REAL'),
                ('object_declarations', 'REAL'),
                ('string_readability', 'REAL'),
                ('all_keywords', 'REAL'),
                ('IF_keywords', 'REAL'),
                ('ELSE_keywords', 'REAL'),
                ('FOR_keywords', 'REAL'),
                ('WHILE_keywords', 'REAL'),
                ('SWITCH_keywords', 'REAL'),
                ('DO_keywords', 'REAL'),
                ('unique_words', 'REAL'),
                ('functions', 'REAL'),
                ('function_name_readability', 'REAL'),
                ('nesting_depth', 'REAL'),
                ('average_parameter_count', 'REAL'),
                ('parameter_count_deviation', 'REAL'),
                ('commands', 'REAL'),
                ('average_commands_per_line', 'REAL'),
                ('commands_per_line_deviation', 'REAL'),
                ('empty_lines', 'REAL'),
                ('tabulators', 'REAL'),
                ('spaces', 'REAL'),
                ('space_indents', 'REAL'),
                ('tab_indents', 'REAL'),
                ('prefers_tabs_over_spaces', 'REAL'),
                ('whitespace_to_character_ratio', 'REAL'),
                ('average_line_length', 'REAL'),
                ('line_length_deviation', 'REAL'),
                ('author', users)
            ],
            'data': arff_data
        }
        arff.dump(profiling_data, file)
        print('Arff dump completed into ' + filename)
        return filename


def take_stats_from_arff(file):
    data = arff_.loadarff(file)
    df = pandas.DataFrame(data[0])

    stat_df = df[['author', 'empty_lines', 'words', 'chars', 'function_name_readability', 'literal_name_readability', 'custom_type_name_readability', 'string_readability', 'comment_readability']].copy()

    stat_df['lines'] = df['non_empty_lines'] + stat_df['empty_lines']
    # stat_df = stat_df[-1:] + stat_df[:-1]

    stat_df['using_OOP'] = ((df['object_declarations'] > 0) | (df['object_definitions'] > 0))
    stat_df['using_STL'] = (df['using_STL_libraries'] == b'True')
    # stat_df = stat_df[:4] + stat_df[-1:] + stat_df[5:-1]

    stat_df = stat_df[['author', 'lines', 'empty_lines', 'words', 'chars', 'using_OOP', 'using_STL', 'function_name_readability', 'literal_name_readability', 'custom_type_name_readability', 'string_readability', 'comment_readability']]
    stat_df.columns = ['Author', 'Lines', 'Empty Lines', 'Words', 'Characters', 'Uses OOP', 'Uses STL', 'Function Readability', 'Literal Readability', 'Typedef Readability', 'String Readability', 'Comment Readability']

    return stat_df


def dump_stats_to_csv(data):
    main_folder_name = base.split('\\')[-1]
    sorted_by_filename = f"{main_folder_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))}_per_file.csv"
    sorted_by_user = f"{main_folder_name}_{str(datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))}_per_user.csv"

    data.to_csv(sorted_by_filename, header=False)
    data.groupby(['Author']).mean().to_csv(sorted_by_user, header=False)


if __name__ == "__main__":
    # Initialize base values

    # Base directory in which the feature search will be performed
    base = "C:\\Users\\Bence\\Documents\\Lecke\\Diplomamunka\\CPP_Files\\9Files_largescale_onlyCPP"
    # base = "C:\\Users\\Bence\\Documents\\Lecke\\Diplomamunka\\CPP_files_lite"
    # base = "C:\\Users\\Bence\\Documents\\Lecke\\Diplomamunka\\Arff_test"

    arff_file_path = "9Files_largescale_onlyCPP_2018-05-28_23_57.arff"

    # Begin Feature Search
    feature_search_start(base_dir=base, make_arff=False, feat_search=False, explicit_arff_path=arff_file_path)
