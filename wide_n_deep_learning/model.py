import argparse
import os
import sys
import tempfile
import tensorflow as tf
import urllib

COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'martial_status',
           'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss',
           'hours_per_week', 'native_country', 'income_bracket']
LABEL_COLUMN = 'income_bracket'
CATEGORICAL_COLUMNS = ['workclass', 'education', 'martial_status', 'occupation',
                       'relationship', 'race', 'gender', 'native_country']
CONTINUOUS_COLUMNS = ['age', 'education_num', 'capital_gain', 'capital_loss',
                      'hours_per_week']
COLUMN_DEFAULTS = [[0.0], [''], [0.0], [''], [0.0], [''], [''], [''], [''], [''], [0.0], [0.0], [0.0], [''], ['']]


def maybe_download(train_data, test_data):
    """Maybe downloads training data and returns train and test file names."""
    if train_data:
        train_file_name = train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve('http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data', train_file.name)
        train_file_name = train_file.name
        train_file.close()
        print('Training data is downloaded to %s' % train_file_name)

    if test_data:
        test_file_name = test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve('http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test', test_file.name)
        test_file_name = test_file.name
        test_file.close()
        print('Test data is downloaded to %s' % test_file_name)

    return train_file_name, test_file_name

def clean(train_file_name, test_file_name):
    with open(train_file_name + '.clean', 'w') as wf:
        with open(train_file_name, 'r') as rf:
            for line in rf.readlines():
                if line.strip() == '': continue
                line = line.replace(', ', ',')
                wf.write(line)
    os.remove(train_file_name)
    os.rename(train_file_name + '.clean', train_file_name)

    with open(test_file_name + '.clean', 'w') as wf:
        with open(test_file_name, 'r') as rf:
            for i, line in enumerate(rf.readlines()):
                if i == 0: continue
                if line.strip() == '': continue
                line = line.replace(', ', ',')
                wf.write(line[:-2] + '\n')
    os.remove(test_file_name)
    os.rename(test_file_name + '.clean', test_file_name)


def build_model_columns():
    # Sparse base columns.
    workclass = tf.feature_column.categorical_column_with_hash_bucket(
        key='workclass', hash_bucket_size=100)
    education = tf.feature_column.categorical_column_with_hash_bucket(
        key='education', hash_bucket_size=1000)
    occupation = tf.feature_column.categorical_column_with_hash_bucket(
        key='occupation', hash_bucket_size=1000)
    relationship = tf.feature_column.categorical_column_with_hash_bucket(
        key='relationship', hash_bucket_size=100)
    gender = tf.feature_column.categorical_column_with_vocabulary_list(
        key='gender', vocabulary_list=['female', 'male'])
    native_country = tf.feature_column.categorical_column_with_hash_bucket(
        key='native_country', hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.feature_column.numeric_column(key='age')
    education_num = tf.feature_column.numeric_column(key='education_num')
    capital_gain = tf.feature_column.numeric_column(key='capital_gain')
    capital_loss = tf.feature_column.numeric_column(key='capital_loss')
    hours_per_week = tf.feature_column.numeric_column(key='hours_per_week')

    # Transformations.
    bucketized_age = tf.feature_column.bucketized_column(
        age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # Crossed columns.
    crossed_columns = [tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=int(1e4)),
                       tf.feature_column.crossed_column([bucketized_age, 'education', 'occupation'], hash_bucket_size=int(1e6)),
                       tf.feature_column.crossed_column(['native_country', 'occupation'], hash_bucket_size=int(1e4))]

    # Wide and deep columns
    base_columns = [workclass, education, occupation, relationship, gender, native_country, bucketized_age]
    wide_columns = base_columns + crossed_columns
    deep_columns = [tf.feature_column.embedding_column(workclass, dimension=8),
                    tf.feature_column.embedding_column(education, dimension=8),
                    tf.feature_column.embedding_column(relationship, dimension=8),
                    tf.feature_column.embedding_column(gender, dimension=8),
                    tf.feature_column.embedding_column(occupation, dimension=8),
                    age, education_num, capital_gain, capital_loss, hours_per_week]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
    """Build an estimator appropriate for the given model type."""
    wide_columns, deep_columns = build_model_columns()
    print(wide_columns)
    print(deep_columns)
    hidden_units = [256, 128, 64]

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns,
                                          hidden_units=hidden_units)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir,
                                                        linear_feature_columns=wide_columns,
                                                        dnn_feature_columns=deep_columns,
                                                        dnn_hidden_units=hidden_units)


def input_fn(data_file, shuffle, batch_size):
    """Input builder function."""

    def parse_data_file(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=COLUMN_DEFAULTS, field_delim=',')
        features = dict(zip(COLUMNS, columns))
        labels = features.pop(LABEL_COLUMN)
        return features, tf.equal(labels, '>50K')

    # Extract lines from input files using the Dataset API.
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_data_file)
    if shuffle:
        dataset.shuffle(buffer_size=10000)
        dataset = dataset.repeat()

    # We call repeat after shuffling, rather than before, to prevent seperate
    # epochs from blending together.
    dataset = dataset.batch(batch_size)
    return dataset


def main(_):
    print(FLAGS)
    train_file_name, test_file_name = maybe_download(FLAGS.train_data, FLAGS.test_data)
    clean(train_file_name, test_file_name)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print('model directory = %s' % model_dir)

    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(
        train_file_name, True, FLAGS.batch_size), max_steps=FLAGS.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(
        test_file_name, False, FLAGS.batch_size))
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    # Evaluate accuracy.
    print('Evaluating...')
    results = model.evaluate(input_fn=lambda: input_fn(
        test_file_name, False, FLAGS.batch_size))
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    print('Train WDL End.')


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--model_dir', type=str, default='./wdl_data/model_save',
                        help='Base directory for output models.')
    parser.add_argument('--model_type', type=str, default='wide_deep',
                        help="Valid model types:{'wide', 'deep', 'wide_deep'}.")
    parser.add_argument('--train_steps', type=int, default=2000,
                        help='Number of training steps.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples per batch')
    parser.add_argument('--train_data', type=str, default='./wdl_data/adult.data',
                        help='Path to the training data.')
    parser.add_argument('--test_data', type=str, default='./wdl_data/adult.test',
                        help='Path to the test data.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
