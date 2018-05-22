#!/usr/bin/env python3

# main.py
# Created on 2018-02-07
# Author: Daniel Indictor

from logger import Logger

import os
import csv
import argparse
from dateutil.parser import parse as dateparse
import datetime
import warnings
import pprint

import numpy as np
from sklearn.metrics import cohen_kappa_score as kappa
from scipy.stats import kendalltau
from scipy.stats import describe as descriptive_statistics
import matplotlib.pyplot as plt
from krippendorff import krippendorff

DATA_PRE_OBS = '../pre-obs/'
DATA_EXP_PRE_SUR = '../exp-pre-sur/'
DATA_EXP_POS_SUR = '../exp-pos-sur/'
DATA_EXP_OBS_POSTURE = '../exp-obs-posture/'
DATA_EXP_OBS_TYPING = '../exp-obs-typing/'
DATA_EXP_OBS_PRE_TYPING = '../exp-pre-typing/'
DATA_EXP_OBS_PRE_POSTURE = '../exp-pre-posture/'
OUTPUT_DIR = '../data_out/'


def is_csv(path):
  return os.path.isfile(path) and path.lower().endswith('.csv')


def is_txt(path):
  return os.path.isfile(path) and path.lower().endswith('.txt')


def is_int(string):
  try:
    int(string)
  except:
    return False

  return True


def is_float(string):
  try:
    float(string)
  except:
    return False

  return True


def get_csv_files_from_folders(path):
  csv_folders = []
  for folders in [os.path.join(path, folder)
                  for folder in os.listdir(path)
                  if os.path.isdir(os.path.join(path, folder))]:
    csv_folders.append((folders, os.listdir(folders)))

  return csv_folders


# Get 2d table from a csv file, sorted in rows, then columns.
def get_csv_table(path):
  with open(path) as csv_file:
    return [row for row in csv.reader(csv_file)]


def get_table(criteria, files, logger=Logger()):
  # Compile list of files from criteria folders.
  # TODO: Put in try-catch block.

  # The table defined here culminates the ratings from every file for
  # a given criteria. It is a multi-dimensional table where getting the
  # index at [row_number][category] yeilds the number of raters who
  # assigned the subject a certain category. This is the format that our
  # data must be for the Fleiss Kappa Coefficient.
  master_table = []

  logger.log('Found files ')
  logger_files = Logger(logger)
  for file in files:
    logger_files.log(file)

  for path in files:
    # We are only using row[1] and onwards because the subject
    # number has no use here. Tables define here depend on the file.

    for row in get_csv_table(os.path.join(criteria, path)):
      # The first number in each row, the subject number, is ignored-
      # hence the slicing ([1:])
      master_table.append(row)

  return master_table


def write_table(path, table):
  with open(path, 'w') as csv_output:
    csv.writer(csv_output).writerows(table)


def calc_pre_obs(data_path, logger=Logger()):
  global rating
  logger.log('pre-obs')
  for criteria, files in get_csv_files_from_folders(data_path):
    logger_criteria = Logger(logger)
    logger_criteria.log('Criteria ' + os.path.basename(criteria) + ':')
    table = get_table(criteria, files, Logger(logger_criteria))

    # Entries with 2 raters
    filled_table = \
      [list(map(int, row)) for row in table
       if (('' not in row) and len(row) == 2)]

    Logger(logger_criteria).log('Number of participants: ' +
                                str(len(table)))
    Logger(logger_criteria).log('Number of participants with both raters: ' +
                                str(len(filled_table)))

    amount_agreed = len([row for row in filled_table if (row[0] == row[1])])
    percent_agreement = amount_agreed / len(filled_table)
    Logger(logger_criteria).log('Percent Agreement: ' +
                                str(percent_agreement))

    Logger(logger_criteria).log('Krippendorff\'s Alpha: ' +
                                str(krippendorff.alpha(filled_table)))

    x_values, y_values = [x for x in zip(*filled_table)]
    Logger(logger_criteria).log('Cohen\'s Kappa: ' +
                                str(kappa(x_values, y_values)))

    ratings = []
    for row in table:
      for rating in row:
        if is_int(rating):
          ratings.append(int(rating))

    Logger(logger_criteria).log(
      'Frequency of risky behavior: ' + str(np.mean(ratings, dtype=np.float64)))


def get_questions_from_table(iter_rows):
  questions = {}

  columns = zip(*[row for row in iter_rows])
  # next called to ignore participant numbers.
  next(columns)

  for column in columns:
    question_name = column[0]
    answers = [int(answer) for answer in column[1:] if is_int(answer)]
    answer_counts = {}
    for x in range(1, max(answers) + 1):
      answer_counts.update({x: answers.count(x)})

    questions.update({question_name: answer_counts})

  return questions


def questions_from_files(data_path, logger_files=Logger()):
  all_questions = {}
  logger_files.log('Found files')

  for file_name in os.listdir(data_path):
    full_path = os.path.join(data_path, file_name)
    if is_csv(full_path):
      Logger(logger_files).log(os.path.basename(file_name))
      with open(full_path) as csv_file:
        rows = csv.reader(csv_file)
        all_questions.update(get_questions_from_table(rows))

  return all_questions


def calc_survey(data_path, output_path = None, logger=Logger(), num_choices=0):

  questions = questions_from_files(data_path, logger_files=logger)

  logger.log('Question answers:')
  question_logger = Logger(logger)

  for question in questions:
    question_logger.log('Question \"' + question + '\":')

    choices = questions[question]

    if num_choices != 0:
      for x in range(1, num_choices + 1):
        if x not in choices:
          choices.update({x: 0})

    for choice, picks in choices.items():
      Logger(question_logger).log('Choice ' + str(choice) + ' chosen ' + str(picks) + ' times')


def calc_exp_obs_posture(data_path, logger=Logger()):
  logger.log('Postural behavior')
  sub_logger = Logger(logger)
  for folder, files in get_csv_files_from_folders(data_path):
    sub_logger.log('Criteria: \"' + os.path.basename(folder) + '\"')
    folder_logger = Logger(sub_logger)
    folder_logger.log('Found files')

    records_from_table = []
    # Concatenate records from all .csv files
    for file in files:
      file_path = os.path.join(folder, file)
      Logger(folder_logger).log(file_path)
      table = get_csv_table(file_path)
      # First row in table is header row.
      for row in table[1:]:
        for item in range(1, 5):
          if not is_int(row[item]):
            # Negative integers used to represent data that's absent.
            row[item] = -1
        records_from_table.append((dateparse(row[0]),
                                   int(row[1]), int(row[2]),
                                   int(row[3]), int(row[4]),))

    # turn records into a numpy array of the records.
    records_from_table = np.array(records_from_table,
                                  dtype=[('date', datetime.date),
                                         ('participant', int),
                                         ('trial', int),
                                         ('rater 1', int),
                                         ('rater 2', int)])

    # Percent Agreement, Krippendorff's Alpha, and Cohen's Kappa
    logger_criteria = Logger(logger)

    filled_table = [(record['rater 1'], record['rater 2'],)
                    for record in records_from_table
                    if ((record['rater 1'] >= 0) and (record['rater 2'] >= 0))]

    amount_agreed = len([row for row in filled_table if (row[0] == row[1])])
    percent_agreement = amount_agreed / len(filled_table)

    Logger(logger_criteria).log('Percent Agreement: ' +
                                str(percent_agreement))

    Logger(logger_criteria).log('Krippendorff\'s Alpha: ' +
                                str(krippendorff.alpha(filled_table)))

    x_values, y_values = [x for x in zip(*filled_table)]
    Logger(logger_criteria).log('Cohen\'s Kappa: ' +
                                str(kappa(x_values, y_values)))

    # adjusted_records will hold the records where each subject is enumerated based
    # on their day rather than the date.
    adjusted_records = []

    current_participant = -1
    current_date = datetime.datetime(1, 1, 1)
    day_number = 0

    for record in np.sort(records_from_table,
                          order=['participant', 'date', 'trial']):
      # reset of on next participant
      if current_participant != record['participant']:
        current_participant = record['participant']
        day_number = 0
        current_date = datetime.datetime(1, 1, 1)
      # reset if on same participant but next day
      if current_date < record['date']:
        current_date = record['date']
        day_number += 1

      adjusted_records.append((day_number,
                               record['participant'],
                               record['trial'],
                               record['rater 1'],
                               record['rater 2'], ))

    # redefine adjusted_records as numpy table.
    adjusted_records = np.array(adjusted_records, dtype=[('day', int),
                                                         ('participant', int),
                                                         ('trial', int),
                                                         ('rater 1', int),
                                                         ('rater 2', int), ])

    participants = {} # maps day numbers on to list of participants.
    for record in adjusted_records:
      if record['day'] in participants:
        if record['participant'] not in participants[record['day']]:
          participants[record['day']].append(record['participant'])
      else:
        participants.update({record['day']: [record['participant']]})

    ratings = {} # maps day numbers onto list of ratings for that day.
    for record in adjusted_records:
      if record['day'] not in ratings:
        ratings[record['day']] = []

      if record['rater 1'] >= 0:
        ratings[record['day']].append(record['rater 1'])
      if record['rater 2'] >= 0:
        ratings[record['day']].append(record['rater 2'])

    statistic_logger = Logger(Logger(folder_logger))

    for day in participants:
      Logger(folder_logger).log('Day ' + str(day) + ':')

      statistic_logger.log('Number of participants: ' + str(len(participants[day])))

      statistic_logger.log('Frequency of risky behavior: ' + str(np.average(ratings[day])))


def calc_exp_obs_typing(data_path, logger=Logger()):
  logger.log('Typing behavior')

  sub_logger = Logger(logger)

  for folder, files in get_csv_files_from_folders(data_path):
    sub_logger.log('Category: \"' + os.path.basename(folder) + '\"')
    folder_logger = Logger(sub_logger)
    file_logger = Logger(sub_logger)
    file_logger.log('Found files')

    records_from_table = []

    # Concatenate records from all .csv files
    for file in files:
      file_path = os.path.join(folder, file)
      Logger(file_logger).log(file_path)
      table = get_csv_table(file_path)
      # First row in table is header row.
      for row in table[1:]:
        records_from_table.append((dateparse(row[0]), int(row[1]), int(row[2]),
                                   np.mean([float(x) for x in row[3:] if is_float(x)], dtype=np.float64),))

    # turn records into a numpy array of the records.
    records_from_table = np.array(records_from_table, dtype=[('date', datetime.date),
                                                             ('participant', int),
                                                             ('trial', int),
                                                             ('measurement', np.float64)])

    # adjusted_records will hold the records where each subject is enumerated based
    # on their day rather than the date.
    adjusted_records = []

    current_participant = -1
    current_date = datetime.datetime(1, 1, 1)
    day_number = 0

    for record in np.sort(records_from_table, order=['participant', 'date', 'trial']):
      # reset of on next participant
      if current_participant != record['participant']:
        current_participant = record['participant']
        day_number = 0
        current_date = datetime.datetime(1, 1, 1)
      # reset if on same participant but next day
      if current_date < record['date']:
        current_date = record['date']
        day_number += 1

      # filter out records with invalid measurements. Measurements less than 0 were used as stand ins.
      if record['measurement'] >= 0:
        adjusted_records.append((day_number, record['participant'], record['trial'], record['measurement'],))

    # redefine adjusted_records as numpy table.
    adjusted_records = np.array(adjusted_records, dtype=[('day', int),
                                                         ('participant', int),
                                                         ('trial', int),
                                                         ('measurement', np.float64)])

    result = kendalltau(adjusted_records['day'], adjusted_records['measurement'])

    folder_logger.log('Mann Kendall Correlation:' + str(result.correlation))
    folder_logger.log('Mann Kendall p value:' + str(result.pvalue))

    folder_logger.log('Descriptive statistics by day:')
    # adjusted_records.sort(order=['day', 'participant',  'trial'])

    # days maps onto a dictionary that maps participant numbers to their measurements for that day.
    days = {}
    for record in adjusted_records:
      if record['day'] in days:
        if record['participant'] in days[record['day']]:
          days[record['day']][record['participant']].append(record['measurement'])
        else:
          days[record['day']].update({record['participant']:
                                        [record['measurement'], ]})
      else:
        days.update({record['day']:
                       {record['participant']: [record['measurement'], ]}})

    statistic_logger = Logger(Logger(folder_logger))
    for day in days:
      # for each participant for each day, average all their measurements and assign it back.
      # now days maps to a dictionary that maps participants to their average measurement for that day.
      Logger(folder_logger).log('Day ' + str(day) + ':')
      for participant in days[day]:
        days[day][participant] = np.average(days[day][participant])

      statistic_logger.log('Number of participants: ' + str(len(days[day])))
      description = descriptive_statistics(list(days[day].values()), nan_policy='omit')

      statistic_logger.log('Average measurement: ' + str(description.mean))
      statistic_logger.log('Variance measurement: ' + str(description.variance))
      statistic_logger.log('Standard Deviation measurement: ' + str(np.sqrt(description.variance)))


def calc_exp_obs_pre_posture(data_path, logger=Logger()):
  logger.log('Postural behavior')
  sub_logger = Logger(logger)

  for folder, files in get_csv_files_from_folders(data_path):
    sub_logger.log('Category: \"' + os.path.basename(folder) + '\"')
    folder_logger = Logger(sub_logger)
    folder_logger.log('Found files')

    # participants maps the participant numbers onto a
    # tuple with each rater's rating of that participant.
    participants = {}

    # Concatenate records from all .csv files
    for file in files:
      file_path = os.path.join(folder, file)
      Logger(folder_logger).log(file_path)
      table = get_csv_table(file_path)
      # First row in table is header row.
      for row in table[1:]:
        for item in range(1, len(row)):
          if not is_int(row[item]):
            row[item] = -1  # Negative integers used to represent data that's absent.

        participants.update({int(row[0]): (int(row[1]), int(row[2]),)})



    filled_table = [rater for rater in participants.values() if ((rater[0] >= 0) and (rater[1] >= 0))]
    amount_agreed = len([row for row in filled_table if (row[0] == row[1])])
    percent_agreement = amount_agreed / len(filled_table)

    folder_logger.log('Number of participants: ' +
                                str(len(participants)))
    folder_logger.log('Number of participants with both raters: ' +
                                str(len(filled_table)))


    folder_logger.log('Percent Agreement: ' +
                      str(percent_agreement))

    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      folder_logger.log('Krippendorff\'s Alpha: ' +
                        str(krippendorff.alpha(filled_table)))

      x_values, y_values = [x for x in zip(*filled_table)]
      folder_logger.log('Cohen\'s Kappa: ' +
                        str(kappa(x_values, y_values)))

    valid_ratings = [] # A list of valid ratings
    for participant_values in participants.values():
      for value in participant_values:
        if value >= 0:
          valid_ratings.append(value)

    folder_logger.log('Frequency of risky behavior: ' +
                      str(np.average(valid_ratings)))


def calc_exp_obs_pre_typing(data_path, logger=Logger()):
  logger.log('Typing behavior')

  sub_logger = Logger(logger)

  for folder, files in get_csv_files_from_folders(data_path):
    sub_logger.log('Category: \"' + os.path.basename(folder) + '\"')
    folder_logger = Logger(sub_logger)
    folder_logger.log('Found files')

    # participants maps the participant numbers onto their
    # average measurement on the experimental pre observation.
    participants = {}

    # Concatenate records from all .csv files
    for file in files:
      file_path = os.path.join(folder, file)
      Logger(folder_logger).log(file_path)
      table = get_csv_table(file_path)
      # First row in table is header row.
      for row in table[1:]:
        for item in range(1, len(row)):
          if not is_float(row[item]):
            row[item] = -1  # Negative integers used to represent data that's absent.

        participants.update({int(row[0]):
                               [float(measurement) for measurement in row[1:] if float(measurement) >= 0]})

    for participant in participants:
      participants[participant] = np.average(participants[participant])

    folder_logger.log('Number of participants: ' + str(len(participants)))
    description = descriptive_statistics(list(participants.values()), nan_policy='omit')

    folder_logger.log('Average measurement: ' + str(description.mean))
    folder_logger.log('Variance measurement: ' + str(description.variance))
    folder_logger.log('Standard Deviation measurement: ' + str(np.sqrt(description.variance)))


def main():
  parser = argparse.ArgumentParser(
    description='Calculate all the things for the SERP Keyboard Group.',
    epilog='Read README.md for more')
  parser.add_argument('-w', '--write', help='write data to DIR',
                      metavar='DIR', default='')
  parser.add_argument('-c', '--charts', help='make some Pie charts.',
                      metavar='DIR', default='')

  args = vars(parser.parse_args())
  output_file = args['write']
  if output_file == '':
    logger = Logger()
  else:
    logger = Logger(open(output_file, 'w'))

  output_dir = args['charts']
  if output_dir != '':
    print('Not Implemented yet!')

    return



  calc_pre_obs(DATA_PRE_OBS, logger=logger)

  exp_pre_sur_output = os.path.join(OUTPUT_DIR, 'exp_pre_sur')
  os.makedirs(exp_pre_sur_output, exist_ok=True)

  logger.log('exp-pre-sur')
  calc_survey(DATA_EXP_PRE_SUR, exp_pre_sur_output, logger=Logger(logger))
  logger.log('exp-pos-sur')
  calc_survey(DATA_EXP_POS_SUR, exp_pre_sur_output, logger=Logger(logger), num_choices=5)

  logger.log('exp-obs-pre')
  calc_exp_obs_pre_posture(DATA_EXP_OBS_PRE_POSTURE, Logger(logger))
  calc_exp_obs_pre_typing(DATA_EXP_OBS_PRE_TYPING, Logger(logger))

  logger.log('exp-obs')
  calc_exp_obs_posture(DATA_EXP_OBS_POSTURE, Logger(logger))
  calc_exp_obs_typing(DATA_EXP_OBS_TYPING, Logger(logger))


if __name__ == '__main__':
  main()
