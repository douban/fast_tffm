import sys
file_path = sys.path[0]
sys.path.append(file_path + '/../py')
import tensorflow as tf
from tensorflow.python.platform import googletest
from fm_ops import fm_ops


class FmParserOpTest(tf.test.TestCase):
    TRAIN_STRING = "1 1:1 2:2 3:3 4:4"
    VOCAB_SIZE = 100
    TARGET_LABEL = 1
    TARGET_FEATURE_IDS = [1, 2, 3, 4]
    TARGET_FEATURE_VALS = [1, 2, 3, 4]

    def testNoHash(self):
        parser_op = fm_ops.fm_parser(self.TRAIN_STRING, self.VOCAB_SIZE, False)
        with self.test_session() as sess:
            label, feature_ids, feature_vals = sess.run(parser_op)
            self.assertEqual(label, self.TARGET_LABEL)
            self.assertAllEqual(feature_ids, self.TARGET_FEATURE_IDS)
            self.assertAllClose(feature_vals, self.TARGET_FEATURE_VALS)

    def testWithHash(self):
        parser_op = fm_ops.fm_parser(self.TRAIN_STRING, self.VOCAB_SIZE, False)
        with self.test_session() as sess:
            label, feature_ids, feature_vals = sess.run(parser_op)
            self.assertEqual(label, self.TARGET_LABEL)
            self.assertAllEqual(feature_ids, self.TARGET_FEATURE_IDS)
            self.assertAllClose(feature_vals, self.TARGET_FEATURE_VALS)

    def testError(self):
        with self.test_session() as sess:
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "Label could not be read in example: "):
                fm_ops.fm_parser(
                    "one 1:1 2:2 3:3 4:4",
                    self.VOCAB_SIZE,
                    False).label.eval()

            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "Invalid feature id "):
                fm_ops.fm_parser(
                    "1 one:1 2:2 3:3 4:4",
                    self.VOCAB_SIZE,
                    False).feature_ids.eval()

            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, "Invalid feature value: "):
                fm_ops.fm_parser(
                    "1 1:one 2:2 3:3 4:4",
                    self.VOCAB_SIZE,
                    False).feature_vals.eval()


if __name__ == "__main__":
    googletest.main()
